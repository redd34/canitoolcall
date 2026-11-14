"""SGLang adapter: replays fixtures through SGLang's real serving-layer code.

Runs inside ``.venvs/sglang`` (built by ``scripts/engines/sglang.sh``: the
SGLang 0.5.20 cp312 manylinux wheel unzipped under ``.engines/sglang/`` and
added through a ``.pth`` file). This module imports only the standard library
at module level; every SGLang import happens inside methods.

What runs is SGLang's own code, pinned to 0.5.20, in the order the OpenAI
server runs it for ``/v1/chat/completions``:

1. **Detokenization.** Token-id groups become text with
   ``DetokenizerManager._decode_batch_token_id_output``: SGLang's incremental
   detokenizer, which keeps a 5-token surrogate prefix, holds back incomplete
   UTF-8, and trims the matched stop token at finish (keeping ``<|call|>``
   when the tool-call parser is ``gpt-oss``). The request's
   ``skip_special_tokens`` is decided the way ``_process_messages`` decides it:
   ``False`` when tools are present, for gpt-oss/Gemma 4, and for the parsers
   ``_patch_reasoning_skip_special_tokens`` lists.
2. **Reasoning first.** ``ReasoningParser`` separates reasoning from normal
   text. ``force_reasoning`` comes from SGLang's template detection
   (``detect_reasoning_pattern`` on the model's chat template) combined with
   ``OpenAIServingChat._get_reasoning_from_request``.
3. **Then tool calls** on the normal text. Non-streaming:
   ``OpenAIServingChat._process_tool_calls`` (``has_tool_call`` +
   ``parse_non_stream``). Streaming: ``_process_reasoning_stream`` and
   ``_process_tool_call_stream`` per step (``parse_stream_chunk``, with
   ``parse_stream_end`` on the finishing step), then
   ``_check_for_unstreamed_tool_args``. The SSE chunks the server would send
   are decoded and accumulated OpenAI-client style.

The serving methods are called on an ``OpenAIServingChat`` built with
``object.__new__``. Only the attributes those methods read are set: the parser
names, the tokenizer, the model's ``architectures``/``model_type`` from
``config.json``, and the template-detection results. No server, scheduler or
model weights are involved.

Stop tokens: fixtures end *before* the stop token. The adapter re-appends the
stop token id that ended generation to the finishing step and lets SGLang's
``trim_matched_stop`` decide what survives. For gpt-oss (Harmony) the stop is
``<|call|>`` when the last message has a recipient (``to=``), else
``<|return|>``. What was appended and what SGLang kept is recorded in
``parser_config["stop"]``. Fixtures tagged ``truncated`` were cut by
``max_tokens``: nothing is appended and every finish-aware call gets SGLang's
``FINISH_LENGTH`` reason (``{"type": "length", ...}``).
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, ClassVar

from canitoolcall.adapters.base import (
    PLACEHOLDER_USER_MESSAGE,
    Adapter,
    AdapterUnavailable,
    ReplayInput,
    Support,
    ToolSpec,
    trusts_remote_code,
)
from canitoolcall.results import ParsedToolCall, ParseResult
from canitoolcall.results import StreamAccumulator as BaseStreamAccumulator

# --------------------------------------------------------------------------- parser map


@dataclass(frozen=True)
class ParserRule:
    """SGLang ``--tool-call-parser`` / ``--reasoning-parser`` for matching models.

    ``match`` is a case-insensitive substring of the HF repo id, or ``"*"``.
    ``tool_call_parser=None`` means SGLang has no parser for these models.
    """

    match: str
    tool_call_parser: str | None
    reasoning_parser: str | None
    unsupported_reason: str | None = None

    def matches(self, model: str) -> bool:
        return self.match == "*" or self.match.lower() in model.lower()


PARSERS: dict[str, tuple[ParserRule, ...]] = {
    # First matching rule wins. Every name is checked against
    # FunctionCallParser.ToolCallParserEnum / ReasoningParser.DetectorMap of
    # SGLang 0.5.20 by tests/adapters/test_sglang.py.
    "qwen3-hermes": (
        # qwen25 == the "qwen" detector SGLang's template auto-detection picks.
        ParserRule("Qwen2.5", "qwen25", None),
        ParserRule("Instruct-2507", "qwen25", None),  # non-thinking Qwen3 update
        ParserRule("*", "qwen25", "qwen3"),  # hybrid and Thinking-2507 (template forces reasoning)
    ),
    "qwen3-xml": (
        ParserRule("Qwen3-Coder", "qwen3_coder", None),
        ParserRule("*", "qwen3_coder", "qwen3"),
    ),
    "gpt-oss": (ParserRule("*", "gpt-oss", "gpt-oss"),),
    "deepseek": (
        ParserRule(
            "DeepSeek-V4.1",
            None,
            None,
            "SGLang 0.5.20 has no DeepSeek-V4.1 parser: V4.1's spaced DSML markers ('DSML' + U+FF5C + ' calls') "
            "appear nowhere in sglang/srt; the deepseekv4 detector only knows V4's 'DSML' + U+FF5C + 'tool_calls'",
        ),
        ParserRule("DeepSeek-V4", "deepseekv4", "deepseek-v4"),
        ParserRule("DeepSeek-V3.2", "deepseekv32", "deepseek-v3"),
        ParserRule("DeepSeek-V3.1", "deepseekv31", "deepseek-v3"),
        ParserRule("DeepSeek-R1", "deepseekv3", "deepseek-r1"),
        ParserRule("DeepSeek-V3", "deepseekv3", None),
    ),
    "kimi": (
        ParserRule("Kimi-K3", "kimi_k3", "kimi_k3"),
        ParserRule("Kimi-K2-Instruct", "kimi_k2", None),
        ParserRule("Kimi-K2", "kimi_k2", "kimi_k2"),  # K2-Thinking, K2.5, K2.6
    ),
    "glm": (
        ParserRule("GLM-4.5", "glm45", "glm45"),
        ParserRule("GLM-4.6", "glm45", "glm45"),
        ParserRule("GLM-4.7", "glm47", "glm45"),
        ParserRule("GLM-5", "glm47", "glm45"),
    ),
    "llama": (
        ParserRule("Llama-4", "pythonic", None),
        ParserRule("Llama-3", "llama3", None),
    ),
    "mistral": (
        ParserRule("Magistral", "mistral", "mistral"),
        ParserRule("Mistral-Medium-3.5", "mistral", "mistral"),
        ParserRule("Mistral-Small-4", "mistral", "mistral"),
        ParserRule("*", "mistral", None),
    ),
    "gemma4": (ParserRule("*", "gemma4", "gemma4"),),
}

HARMONY_CALL = "<|call|>"
HARMONY_RETURN = "<|return|>"
REQUEST_ID = "chatcmpl-canitoolcall-replay"


def resolve_parsers(family: str, model: str) -> ParserRule | None:
    """The first :class:`ParserRule` of ``family`` matching ``model``, if any."""
    return next((r for r in PARSERS.get(family, ()) if r.matches(model)), None)


def exception_text(exc: BaseException) -> str:
    """``"Type: message"`` as recorded in :attr:`ParseResult.exception`."""
    return f"{type(exc).__name__}: {exc}"


def harmony_stop_token(text: str) -> str:
    """The Harmony stop token that ends ``text``'s last message.

    Tool calls are messages with a recipient (``to=...`` in the role or channel
    part of the header) and end with ``<|call|>``; everything else ends the
    turn with ``<|return|>`` (openai/harmony ``docs/format.md``).
    """
    start = text.rfind("<|start|>")
    last = text[start:] if start >= 0 else text
    header = last.split("<|message|>", 1)[0]
    return HARMONY_CALL if " to=" in f" {header}" else HARMONY_RETURN


def _mod(name: str) -> Any:
    return importlib.import_module(name)


# --------------------------------------------------------------------------- stream accumulation


class StreamAccumulator(BaseStreamAccumulator):
    """The shared OpenAI-client accumulator (DESIGN.md rule 8), fed SGLang's SSE chunks."""

    def add_sse(self, chunk: str) -> None:
        """Decode one ``data: {...}`` server-sent event from SGLang and accumulate it."""
        for line in chunk.splitlines():
            if not line.startswith("data: "):
                continue
            body = line[len("data: ") :].strip()
            if not body or body == "[DONE]":
                continue
            payload = json.loads(body)
            for choice in payload.get("choices") or ():
                self.add_openai_delta(choice.get("delta") or {})


# --------------------------------------------------------------------------- per-model state


@dataclass
class _ModelContext:
    """Tokenizer, config and template-detection results for one reference model."""

    model: str
    tokenizer: Any
    tokenizer_repo: str
    tokenizer_revision: str | None
    hf_config: Any
    """``SimpleNamespace(architectures, model_type)`` from config.json, or None."""
    config_error: str | None
    template: str | None
    force_reasoning: bool
    reasoning_config: Any
    auto_tool_call_parser: str | None
    auto_reasoning_parser: str | None


@dataclass(frozen=True)
class _StopPlan:
    token: str | None
    token_id: int | None
    appended: bool
    rule: str
    length: int | None = None
    """Output length when generation hit ``max_tokens`` (``FINISH_LENGTH``); None for a stop."""

    @property
    def finish_type(self) -> str:
        return "length" if self.length is not None else "stop"

    def finish_reason(self) -> dict[str, Any]:
        """A fresh ``finish_reason`` dict (``FINISH_LENGTH``/``FINISH_MATCHED_TOKEN.to_json()``).

        Fresh on every call: ``_process_tool_calls`` mutates the dict it gets.
        """
        if self.length is not None:
            return {"type": "length", "length": self.length}
        return {"type": "stop", "matched": self.token_id}


@dataclass
class _Replay:
    """Everything one replay needs: serving stub, request and detokenizer inputs."""

    ctx: _ModelContext
    rule: ParserRule
    serving: Any
    request: Any
    sampling: dict[str, Any]
    prompt_tail: list[int]
    prompt_tail_source: str
    stop: _StopPlan
    notes: list[str]


class SglangAdapter(Adapter):
    name: ClassVar[str] = "sglang"
    pinned_version: ClassVar[str] = "0.5.20"
    supports_text_deltas: ClassVar[bool] = True

    def __init__(self) -> None:
        self._contexts: dict[tuple[str, str, str | None], _ModelContext] = {}
        self._tools: dict[str, list[dict[str, Any]]] = {}
        """Tools of the latest replay per fixture id (``parser_config`` gets no tools)."""
        self._tails: dict[tuple[str, str], tuple[list[int], str]] = {}

    # ------------------------------------------------------------------ identity

    def version(self) -> str:
        try:
            sglang = _mod("sglang")
        except ImportError as e:
            raise AdapterUnavailable(f"sglang is not importable here ({e}); run scripts/engines/sglang.sh") from e
        return str(sglang.__version__)

    def engine_details(self) -> dict[str, Any]:
        from importlib import metadata
        from pathlib import Path

        sglang = _mod("sglang")
        fcp = _mod("sglang.srt.function_call.function_call_parser")
        rp = _mod("sglang.srt.parser.reasoning_parser")
        root = Path(sglang.__file__).resolve().parent.parent
        dist_info = sorted(p.name for p in root.glob("sglang-*.dist-info"))
        deps: dict[str, str | None] = {}
        for dist in ("transformers", "tokenizers", "torch", "xgrammar", "pydantic", "partial-json-parser", "orjson"):
            try:
                deps[dist] = metadata.version(dist)
            except metadata.PackageNotFoundError:
                deps[dist] = None
        return {
            "install": "cp312 manylinux wheel unpacked and added via .pth (scripts/engines/sglang.sh)",
            "dist_info": dist_info[0] if dist_info else None,
            "tool_call_parsers": len(fcp.FunctionCallParser.ToolCallParserEnum),
            "reasoning_parsers": len(rp.ReasoningParser.DetectorMap),
            "deps": deps,
        }

    # ------------------------------------------------------------------ contract

    def supports(self, family: str, model: str) -> Support:
        rule = resolve_parsers(family, model)
        if rule is None:
            if family not in PARSERS:
                return Support(False, f"no SGLang parser mapping for family {family!r}")
            return Support(False, f"no SGLang parser mapping for model {model!r} in family {family!r}")
        if rule.tool_call_parser is None:
            return Support(False, rule.unsupported_reason or f"SGLang has no parser for {model!r}")
        return Support(True)

    def units(self, raw: ReplayInput) -> list[int]:
        if raw.token_ids is not None:
            return list(raw.token_ids)
        ctx = self._context(raw)
        return [int(i) for i in ctx.tokenizer.encode(raw.text, add_special_tokens=False)]

    def special_token_ids(self, raw: ReplayInput) -> Collection[int] | None:
        tok = self._context(raw).tokenizer
        ids = {int(i) for i in getattr(tok, "all_special_ids", ())}
        # HF tokenizers expose added tokens as a dict; mistral-common's backend
        # only lists its control tokens in all_special_ids.
        decoder = getattr(tok, "added_tokens_decoder", None)
        if isinstance(decoder, dict):
            ids.update(int(i) for i, t in decoder.items() if getattr(t, "special", False))
        return frozenset(ids)

    def supports_replay(self, raw: ReplayInput) -> Support:
        # _stop_plan handles explicit termination for both parse paths.
        if raw.finish is not None and not raw.assumes:
            return self.supports(raw.family, raw.model)
        return super().supports_replay(raw)

    def parser_config(self, raw: ReplayInput) -> dict[str, Any]:
        tools = self._tools.get(raw.fixture_id)
        rp = self._replay(raw, tools, stream=False)
        srv, req, ctx = rp.serving, rp.request, rp.ctx
        kept = self._kept_stop_ids(rp)
        tool_detector = None
        reasoning_detector = None
        if rp.rule.tool_call_parser:
            fcp = _mod("sglang.srt.function_call.function_call_parser")
            tool_detector = fcp.FunctionCallParser.ToolCallParserEnum[rp.rule.tool_call_parser].__name__
        if rp.rule.reasoning_parser:
            rpm = _mod("sglang.srt.parser.reasoning_parser")
            reasoning_detector = rpm.ReasoningParser.DetectorMap[rp.rule.reasoning_parser].__name__
        return {
            "engine": "sglang",
            "version": self.version(),
            "tool_call_parser": rp.rule.tool_call_parser,
            "reasoning_parser": rp.rule.reasoning_parser,
            "tool_call_detector": tool_detector,
            "reasoning_detector": reasoning_detector,
            "auto_detected": {
                "tool_call_parser": ctx.auto_tool_call_parser,
                "reasoning_parser": ctx.auto_reasoning_parser,
            },
            "model": raw.model,
            "tokenizer": {
                "repo": ctx.tokenizer_repo,
                "revision": ctx.tokenizer_revision,
                "class": type(ctx.tokenizer).__name__,
                "loader": "sglang.srt.utils.hf_transformers_utils.get_tokenizer(revision=, tokenizer_revision=)",
                "trust_remote_code": trusts_remote_code(ctx.tokenizer_repo, ctx.tokenizer_revision),
            },
            "tokenizer_mode": raw.tokenizer_mode,
            "hf_config": (
                {"architectures": ctx.hf_config.architectures, "model_type": ctx.hf_config.model_type}
                if ctx.hf_config is not None
                else None
            ),
            "hf_config_error": ctx.config_error,
            "chat_template_sha256": (
                hashlib.sha256(ctx.template.encode("utf-8")).hexdigest() if ctx.template is not None else None
            ),
            "template_force_reasoning": ctx.force_reasoning,
            "template_reasoning_config": repr(ctx.reasoning_config) if ctx.reasoning_config is not None else None,
            "chat_encoding_spec": srv.chat_encoding_spec,
            "thinking": raw.thinking,
            "chat_template_kwargs": req.chat_template_kwargs,
            "reasoning_effort": req.reasoning_effort,
            "reasoning_enabled": bool(srv._get_reasoning_from_request(req)),
            "tools_offered": None if tools is None else len(tools),
            "tool_choice": req.tool_choice if isinstance(req.tool_choice, str) else "named",
            "separate_reasoning": req.separate_reasoning,
            "stream_reasoning": req.stream_reasoning,
            "skip_special_tokens": rp.sampling["skip_special_tokens"],
            "spaces_between_special_tokens": rp.sampling["spaces_between_special_tokens"],
            "no_stop_trim": bool(rp.sampling["no_stop_trim"]),
            "detokenizer": "DetokenizerManager._decode_batch_token_id_output",
            "prompt_tail": {"source": rp.prompt_tail_source, "ids": rp.prompt_tail},
            "stop": {
                "token": rp.stop.token,
                "id": rp.stop.token_id,
                "rule": rp.stop.rule,
                "appended": rp.stop.appended,
                "finish_reason": rp.stop.finish_type,
                "kept_by_engine": bool(kept),
            },
            "notes": rp.notes,
        }

    def parse(self, raw: ReplayInput, tools: Sequence[ToolSpec]) -> ParseResult:
        rp = self._replay(raw, tools, stream=False)
        steps = [self.units(raw) + ([rp.stop.token_id] if rp.stop.appended and rp.stop.token_id is not None else [])]
        try:
            (text,) = self._detokenize(rp, steps)
            return self._nonstream(rp, text)
        except Exception as e:  # engine outcome (rule 6)
            return ParseResult(exception=exception_text(e))

    def parse_stream(self, raw: ReplayInput, chunks: Sequence[Sequence[int]], tools: Sequence[ToolSpec]) -> ParseResult:
        rp = self._replay(raw, tools, stream=True)
        steps = [list(c) for c in chunks] or [[]]
        if rp.stop.appended and rp.stop.token_id is not None:
            steps[-1] = [*steps[-1], rp.stop.token_id]
        acc = StreamAccumulator()
        try:
            deltas = self._detokenize(rp, steps)
            asyncio.run(self._stream(rp, deltas, acc))
        except Exception as e:  # engine outcome (rule 6)
            return acc.result(exception=exception_text(e))
        return acc.result()

    def parse_stream_text(self, raw: ReplayInput, deltas: Sequence[str], tools: Sequence[ToolSpec]) -> ParseResult:
        """Synthetic text deltas (``char:<seed>``), bypassing the detokenizer.

        The stop token text SGLang keeps (only ``<|call|>`` for gpt-oss) is
        appended to the last delta, as the detokenizer would have emitted it.
        """
        rp = self._replay(raw, tools, stream=True)
        texts = list(deltas) or [""]
        kept = self._kept_stop_ids(rp)
        if kept:
            texts[-1] += rp.ctx.tokenizer.decode(kept, skip_special_tokens=rp.sampling["skip_special_tokens"])
        acc = StreamAccumulator()
        try:
            asyncio.run(self._stream(rp, texts, acc))
        except Exception as e:
            return acc.result(exception=exception_text(e))
        return acc.result()

    # ------------------------------------------------------------------ setup

    def _revision(self, raw: ReplayInput) -> tuple[str, str | None]:
        """Tokenizer repo and revision: the fixture's pin, else the family's reference model."""
        if raw.tokenizer is not None:
            return raw.tokenizer.repo, raw.tokenizer.revision
        if raw.revision:
            return raw.model, raw.revision
        from canitoolcall.fixtures import load_family

        try:
            ref = load_family(raw.family).reference_for(raw.model)
        except (OSError, ValueError, KeyError):
            ref = None
        if ref is None:
            raise ValueError(
                f"{raw.fixture_id}: no pinned revision for {raw.model!r} "
                "(no tokenizer pin and no reference-model entry in the family)"
            )
        return raw.model, ref.revision

    def _context(self, raw: ReplayInput) -> _ModelContext:
        repo, revision = self._revision(raw)
        key = (raw.model, repo, revision)
        if key in self._contexts:
            return self._contexts[key]
        self.version()  # AdapterUnavailable if SGLang is missing
        get_tokenizer = _mod("sglang.srt.utils.hf_transformers_utils").get_tokenizer
        td = _mod("sglang.srt.parser.template_detection")
        # The server passes only ``revision=`` (``--revision``); its post-load
        # fixes (e.g. _fix_v5_tokenizer_components, which restores DeepSeek's
        # ByteLevel decoder) read tokenizer.json at ``tokenizer_revision``, i.e.
        # "main" from the local cache. Pinning both models a server whose
        # "main" is the pinned revision, and keeps the fixes working offline.
        tok = get_tokenizer(
            repo, trust_remote_code=trusts_remote_code(repo, revision), revision=revision, tokenizer_revision=revision
        )
        hf_config, config_error = self._hf_config(raw.model, revision if repo == raw.model else None)
        template = getattr(tok, "chat_template", None)
        if not isinstance(template, str):
            template = None
        force, reasoning_config = td.detect_reasoning_pattern(template)
        ctx = _ModelContext(
            model=raw.model,
            tokenizer=tok,
            tokenizer_repo=repo,
            tokenizer_revision=revision,
            hf_config=hf_config,
            config_error=config_error,
            template=template,
            force_reasoning=bool(force),
            reasoning_config=reasoning_config,
            auto_tool_call_parser=td.detect_tool_call_parser(template, tok, reasoning_config, force),
            auto_reasoning_parser=td.detect_reasoning_parser(template, tok, reasoning_config, force),
        )
        self._contexts[key] = ctx
        return ctx

    @staticmethod
    def _hf_config(model: str, revision: str | None) -> tuple[Any, str | None]:
        """``architectures``/``model_type`` from the model's config.json (no weights)."""
        try:
            hf_hub_download = _mod("huggingface_hub").hf_hub_download
            path = hf_hub_download(model, "config.json", revision=revision)
            with open(path, encoding="utf-8") as fh:
                cfg = json.load(fh)
        except Exception as e:  # gated or missing repos: record, don't guess
            return None, exception_text(e)
        return SimpleNamespace(architectures=cfg.get("architectures"), model_type=cfg.get("model_type")), None

    def _serving(self, ctx: _ModelContext, rule: ParserRule) -> Any:
        """An ``OpenAIServingChat`` with just the state its parsing methods read."""
        serving_chat = _mod("sglang.srt.entrypoints.openai.serving_chat")
        rpm = _mod("sglang.srt.parser.reasoning_parser")
        srv = object.__new__(serving_chat.OpenAIServingChat)
        srv.tool_call_parser = rule.tool_call_parser
        srv.reasoning_parser = rule.reasoning_parser
        srv.default_chat_template_kwargs = {}
        hf_config = ctx.hf_config or SimpleNamespace(architectures=None, model_type=None)
        srv.tokenizer_manager = SimpleNamespace(
            tokenizer=ctx.tokenizer, model_config=SimpleNamespace(hf_config=hf_config)
        )
        srv.template_manager = SimpleNamespace(
            force_reasoning=ctx.force_reasoning, reasoning_config=ctx.reasoning_config
        )
        srv._reasoning_detector = None
        if rule.reasoning_parser:
            try:  # as OpenAIServingChat.__init__
                srv._reasoning_detector = rpm.ReasoningParser(
                    model_type=rule.reasoning_parser, stream_reasoning=True, tokenizer=ctx.tokenizer
                ).detector
            except ValueError:
                srv._reasoning_detector = None
        srv.is_gpt_oss = hf_config.model_type == "gpt_oss"
        srv.is_gemma4 = hf_config.model_type in ("gemma4", "gemma4_unified")
        srv.chat_encoding_spec = srv._resolve_chat_encoding_spec()
        return srv

    def _replay(self, raw: ReplayInput, tools: Sequence[ToolSpec] | None, stream: bool) -> _Replay:
        rule = resolve_parsers(raw.family, raw.model)
        if rule is None or rule.tool_call_parser is None:
            raise ValueError(f"{raw.fixture_id}: unsupported by SGLang ({self.supports(raw.family, raw.model).reason})")
        ctx = self._context(raw)
        srv = self._serving(ctx, rule)
        protocol = _mod("sglang.srt.entrypoints.openai.protocol")
        notes: list[str] = []
        if tools is not None:
            self._tools[raw.fixture_id] = [dict(t) for t in tools]
        req = protocol.ChatCompletionRequest(
            model="canitoolcall-replay",
            messages=[{"role": "user", "content": PLACEHOLDER_USER_MESSAGE}],
            tools=[dict(t) for t in tools] if tools else None,
            stream=stream,
        )
        if tools is None:
            notes.append("parser_config computed before this fixture was replayed: request built without tools")
        if raw.thinking is not None:
            try:  # the engine's own mapping of "reasoning on/off" to template kwargs / effort
                srv.apply_reasoning_enabled(req, raw.thinking)
            except ValueError as e:
                notes.append(f"thinking={raw.thinking} not applied: {e}")
        # OpenAIServingChat._process_messages: which special tokens survive detokenization.
        if srv.is_gpt_oss or srv.is_gemma4:
            req.skip_special_tokens = False
        srv._patch_reasoning_skip_special_tokens(req)
        if srv._effective_tools(req) and req.tool_choice != "none":
            req.skip_special_tokens = False
        sampling = req.to_sampling_params(stop=[], model_generation_config={})
        tail_key = (raw.fixture_id, raw.model)
        if tail_key not in self._tails:
            self._tails[tail_key] = self._prompt_tail(ctx, raw, tools, req)
        tail, tail_source = self._tails[tail_key]
        return _Replay(
            ctx=ctx,
            rule=rule,
            serving=srv,
            request=req,
            sampling=sampling,
            prompt_tail=tail,
            prompt_tail_source=tail_source,
            stop=self._stop_plan(ctx, raw, self.units(raw)),
            notes=notes,
        )

    @staticmethod
    def _prompt_tail(
        ctx: _ModelContext, raw: ReplayInput, tools: Sequence[ToolSpec] | None, req: Any
    ) -> tuple[list[int], str]:
        """The last prompt ids SGLang's detokenizer uses as its surrogate prefix."""
        n = _mod("sglang.srt.managers.schedule_batch").INIT_INCREMENTAL_DETOKENIZATION_OFFSET
        tok = ctx.tokenizer
        if raw.generation_prompt:
            ids = [int(i) for i in tok.encode(raw.generation_prompt, add_special_tokens=False)]
            return ids[-n:], "generation_prompt"
        if ctx.template is not None:
            try:
                out = tok.apply_chat_template(
                    [{"role": "user", "content": PLACEHOLDER_USER_MESSAGE}],
                    tools=[dict(t) for t in tools] if tools else None,
                    add_generation_prompt=True,
                    tokenize=True,
                    **(req.chat_template_kwargs or {}),
                )
                ids = list(out["input_ids"] if hasattr(out, "keys") else out)
                return [int(i) for i in ids[-n:]], "chat_template"
            except Exception:
                pass
        return [], "none"

    @staticmethod
    def _stop_plan(ctx: _ModelContext, raw: ReplayInput, units: Sequence[int]) -> _StopPlan:
        if raw.finished_by_length:
            # Cut by max_tokens: the model never emitted a stop token, and the
            # scheduler finishes the request with FINISH_LENGTH.
            return _StopPlan(
                None, None, appended=False, rule="truncated fixture: finish_reason length", length=len(units)
            )
        tok = ctx.tokenizer

        def tid(token: str) -> int | None:
            i = tok.convert_tokens_to_ids(token)
            return int(i) if isinstance(i, int) and i != tok.unk_token_id and i >= 0 else None

        if raw.finish is not None:
            token = raw.finish.token
            assert token is not None  # length finishes returned above
            token_id = tid(token)
            if token_id is None:
                raise ValueError("finish.token is not a known token for this tokenizer")
            return _StopPlan(token, token_id, appended=True, rule="explicit fixture finish.token")
        stops = [s for s in raw.stop_tokens if s]
        for s in stops:
            if raw.text.endswith(s):
                return _StopPlan(s, tid(s), appended=False, rule="raw_output already ends with the stop token")
        if HARMONY_CALL in stops and HARMONY_RETURN in stops:
            token = harmony_stop_token(raw.text)
            rule = "harmony: <|call|> after a message with a recipient, else <|return|>"
        elif stops:
            token, rule = stops[0], "first stop token of the reference model"
        else:
            eos = tok.eos_token
            if not eos:
                return _StopPlan(None, None, appended=False, rule="no stop token known")
            token, rule = str(eos), "tokenizer eos_token (reference model lists no stop tokens)"
        token_id = tid(token)
        return _StopPlan(token, token_id, appended=token_id is not None, rule=rule)

    # ------------------------------------------------------------------ engine calls

    def _detokenizer(self, rp: _Replay) -> Any:
        """A ``DetokenizerManager`` with the state ``_decode_batch_token_id_output`` reads."""
        dm = _mod("sglang.srt.managers.detokenizer_manager")
        tok = rp.ctx.tokenizer
        d = object.__new__(dm.DetokenizerManager)
        d.tokenizer = tok
        try:
            d.vocab_size = len(tok)
        except TypeError:
            d.vocab_size = getattr(tok, "vocab_size", None)
        d.decode_status = {}
        d.disable_tokenizer_batch_decode = False  # server default
        d.is_tool_call_parser_gpt_oss = rp.rule.tool_call_parser == "gpt-oss"
        return d

    def _detokenize(self, rp: _Replay, steps: Sequence[Sequence[int]]) -> list[str]:
        """Text delta per scheduler step, exactly as the detokenizer process emits it.

        Mirrors ``Req.init_incremental_detokenize`` + the output streamer: the
        first step sends the surrogate prompt tail plus the new ids, later
        steps only their new ids; the last step carries the finish reason.
        """
        detok = self._detokenizer(rp)
        all_ids = list(rp.prompt_tail)
        sent = 0
        out: list[str] = []
        for i, step in enumerate(steps):
            all_ids.extend(step)
            last = i == len(steps) - 1
            recv = SimpleNamespace(
                rids=["canitoolcall-replay"],
                finished_reasons=[rp.stop.finish_reason() if last else None],
                decoded_texts=[""],
                decode_ids=[all_ids[sent:]],
                read_offsets=[len(rp.prompt_tail)],
                skip_special_tokens=[rp.sampling["skip_special_tokens"]],
                spaces_between_special_tokens=[rp.sampling["spaces_between_special_tokens"]],
                no_stop_trim=[bool(rp.sampling["no_stop_trim"])],
            )
            sent = len(all_ids)
            out.append(detok._decode_batch_token_id_output(recv)[0])
        return out

    def _kept_stop_ids(self, rp: _Replay) -> list[int]:
        """Stop ids SGLang keeps in the output (``trim_matched_stop``), e.g. gpt-oss ``<|call|>``."""
        if rp.stop.token_id is None:
            return []
        kept = self._detokenizer(rp).trim_matched_stop(
            [rp.stop.token_id], rp.stop.finish_reason(), bool(rp.sampling["no_stop_trim"])
        )
        return [int(i) for i in kept]

    @staticmethod
    def _nonstream(rp: _Replay, text: str) -> ParseResult:
        """``OpenAIServingChat._build_chat_response`` for one choice."""
        srv, req = rp.serving, rp.request
        rpm = _mod("sglang.srt.parser.reasoning_parser")
        reasoning_text = None
        if srv.reasoning_parser and req.separate_reasoning:
            force_reasoning = srv.template_manager.force_reasoning or srv._get_reasoning_from_request(req)
            parser = rpm.ReasoningParser(
                model_type=srv.reasoning_parser,
                stream_reasoning=False,
                force_reasoning=force_reasoning,
                request=req,
                tokenizer=rp.ctx.tokenizer,
                tool_call_parser_active=srv._tool_call_parsing_active(req),
            )
            reasoning_text, text = parser.parse_non_stream(text)
        calls: tuple[ParsedToolCall, ...] = ()
        if srv._tool_call_parsing_active(req):
            tool_calls, text, _ = srv._process_tool_calls(
                text,
                srv._effective_tools(req),
                rp.stop.finish_reason(),
                req.tool_choice,
                srv._get_history_tool_calls_cnt(req),
            )
            calls = tuple(ParsedToolCall(tc.function.name, tc.function.arguments) for tc in tool_calls or ())
        return ParseResult(content=text or None, reasoning_content=reasoning_text or None, tool_calls=calls)

    @staticmethod
    async def _stream(rp: _Replay, deltas: Sequence[str], acc: StreamAccumulator) -> None:
        """``OpenAIServingChat._generate_stream_content`` over each step, for choice 0."""
        srv, req = rp.serving, rp.request
        reasoning_parsers: dict[int, Any] = {}
        tool_parsers: dict[int, Any] = {}
        has_tool_calls: dict[int, bool] = {}
        content: dict[str, Any] = {"meta_info": {"id": REQUEST_ID}, "text": ""}
        for i, step_text in enumerate(deltas):
            finish_type = rp.stop.finish_type if i == len(deltas) - 1 else None
            content["text"] += step_text
            delta: Any = step_text
            if srv.reasoning_parser and req.separate_reasoning:
                reasoning_text, delta = srv._process_reasoning_stream(
                    0, delta, reasoning_parsers, content, req, finish_type
                )
                acc.add_reasoning(reasoning_text)
            if srv._tool_call_parsing_active(req):
                async for chunk in srv._process_tool_call_stream(
                    0, delta, tool_parsers, content, req, has_tool_calls, False, flush=finish_type is not None
                ):
                    if chunk:
                        acc.add_sse(chunk)
                if finish_type is not None and 0 in tool_parsers:
                    remaining = srv._check_for_unstreamed_tool_args(tool_parsers[0], content, req, 0)
                    if remaining:
                        acc.add_sse(remaining)
            else:
                acc.add_content(delta)

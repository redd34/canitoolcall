"""vLLM adapter: replays fixtures through vLLM's real tool/reasoning parsers.

Runs in ``.venvs/vllm`` (built by ``scripts/engines/vllm.sh``: the vLLM 0.30.0
manylinux wheel unzipped under ``.engines/vllm/`` and added via ``.pth``, CPU
torch, no model weights). vLLM is imported lazily, inside methods, so this
module loads with the standard library only.

Every step mirrors what vLLM 0.30.0's OpenAI server does for
``/v1/chat/completions`` with ``--enable-auto-tool-choice``
(``vllm/entrypoints/openai/chat_completion/serving.py`` and
``vllm/renderers/online_renderer.py``):

1. **Tokenizer:** ``vllm.tokenizers.get_tokenizer(repo, revision=...,
   trust_remote_code=True, tokenizer_mode=...)``. The mode is ``mistral`` for
   Mistral fixtures, the DeepSeek V3.2+/Kimi K3 mode that ``ModelConfig``
   derives from ``config.json`` architectures, else ``auto``.
2. **Parser class:** ``ParserManager.get_parser(tool_parser_name,
   reasoning_parser_name, enable_auto_tools=True, model_name=repo,
   is_harmony=(model_type == "gpt_oss"))``.
3. **Request:** a ``ChatCompletionRequest`` with the fixture's tools and
   ``tool_choice="auto"`` (both omitted when the fixture offers no tools, as
   a client would), and ``chat_template_kwargs`` from the fixture's
   ``thinking`` flag. A throwaway parser instance runs
   ``adjust_request(request)`` first, under the same condition as
   ``OnlineRenderer.preprocess_chat``. Many parsers force
   ``skip_special_tokens=False`` there.
4. **Prompt ids:** the renderer vLLM would pick for the tokenizer mode renders
   one user turn plus the tools with ``add_generation_prompt=True``. The ids are
   passed to ``parse_delta(prompt_token_ids=...)``, as serving does; without
   them ``reasoning_ended`` is never set.
5. **Text:** each id group goes through the v1 ``IncrementalDetokenizer``
   (never ``tok.decode``), built from ``request.to_sampling_params(...)``.
6. **Parse:** non-streaming ``p.parse(text, request, enable_auto_tools=True,
   model_output_token_ids=ids)``; streaming ``p.parse_delta(delta_text,
   delta_ids, request, prompt_token_ids=..., finished=last)``. The parser
   instances get ``chat_template_kwargs`` exactly as serving builds them.

Stream deltas are accumulated like openai-python's ``accumulate_delta``.
Content and reasoning are concatenated, and tool calls are merged by
``index``, with ``name`` and ``arguments`` fragments concatenated. Exceptions
raised by vLLM code are returned as ``ParseResult.exception``.

The stop token is **not** re-appended. ``raw_output`` ends before it, and
vLLM's server never puts it in the text (``IncrementalDetokenizer.update(...,
stop_terminated=True)`` drops it). A real ``finish_reason="stop"`` would still
carry its id in the last delta's ``token_ids``, but a fixture does not say
whether the model stopped or hit ``max_tokens``. So the final delta is
replayed as ``finished=True`` with the fixture's ids only, which is recorded
as ``stop_token_in_final_delta: false``. Re-appending the id changed no result
for the gpt-oss and qwen3-hermes corpora, apart from fixtures that end
truncated or already contain the stop token.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
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
from canitoolcall.results import ParsedToolCall, ParseResult, StreamAccumulator

# --------------------------------------------------------------------------- parser map


@dataclass(frozen=True)
class ParserChoice:
    """The ``vllm serve`` flags for one model: ``--tool-call-parser`` / ``--reasoning-parser``."""

    tool_parser: str
    reasoning_parser: str | None
    source: str
    """Where this choice comes from (model card, docs/formats, engine source)."""


@dataclass(frozen=True)
class ModelRule:
    """Maps reference-model repos (regex, case-insensitive ``search``) to a parser choice."""

    pattern: str
    choice: ParserChoice | None
    reason: str | None = None
    """Why the model is unsupported (when ``choice`` is None)."""

    def matches(self, model: str) -> bool:
        return re.search(self.pattern, model, flags=re.IGNORECASE) is not None


def _rule(pattern: str, tool: str, reasoning: str | None, source: str) -> ModelRule:
    return ModelRule(pattern, ParserChoice(tool, reasoning, source))


# Every name below is registered in vLLM 0.30.0 (ToolParserManager /
# ReasoningParserManager.list_registered(); checked by the engine tests).
# Rules are tried in order and the first match wins, so specific rules go
# before general ones.
PARSERS: dict[str, tuple[ModelRule, ...]] = {
    "qwen3-hermes": (
        _rule(
            r"Qwen3-.*Instruct-2507",
            "hermes",
            None,
            "Qwen3-2507 Instruct: non-thinking model; docs/formats/qwen3-hermes.md",
        ),
        _rule(
            r"Qwen3-.*Thinking-2507",
            "hermes",
            "deepseek_r1",
            "Qwen3-2507 Thinking model cards (no opening <think>); docs/formats/qwen3-hermes.md",
        ),
        _rule(r"Qwen2\.5", "hermes", None, "Qwen2.5 model card (hermes, no reasoning)"),
        _rule(r"Qwen3", "hermes", "qwen3", "Qwen3 model card: --tool-call-parser hermes --reasoning-parser qwen3"),
    ),
    "qwen3-xml": (
        _rule(r"Qwen3-Coder", "qwen3_coder", None, "Qwen3-Coder model cards: --tool-call-parser qwen3_coder"),
        _rule(
            r"Qwen3\.[5-9]",
            "qwen3_coder",
            "qwen3",
            "Qwen3.5+ model cards: --tool-call-parser qwen3_coder --reasoning-parser qwen3",
        ),
    ),
    "gpt-oss": (_rule(r"gpt-oss", "openai", "openai_gptoss", "vLLM gpt-oss recipe; HarmonyParser (is_harmony)"),),
    "deepseek": (
        _rule(r"DeepSeek-V4\.1", "deepseek_v41", "deepseek_v41", "docs/formats/deepseek.md (vLLM v0.30.0 row)"),
        _rule(r"DeepSeek-V4", "deepseek_v4", "deepseek_v4", "docs/formats/deepseek.md (vLLM v0.30.0 row)"),
        _rule(r"DeepSeek-V3\.2", "deepseek_v32", "deepseek_v3", "docs/formats/deepseek.md (vLLM v0.30.0 row)"),
        _rule(r"DeepSeek-V3\.1", "deepseek_v31", "deepseek_v3", "docs/formats/deepseek.md (vLLM v0.30.0 row)"),
        _rule(r"DeepSeek-R1", "deepseek_v3", "deepseek_r1", "docs/formats/deepseek.md (vLLM v0.30.0 row)"),
        _rule(r"DeepSeek-V3", "deepseek_v3", None, "docs/formats/deepseek.md (V3: no reasoning)"),
    ),
    "kimi": (
        _rule(r"Kimi-K3", "kimi_k3", "kimi_k3", "docs/formats/kimi.md (vLLM v0.30.0 row)"),
        _rule(
            r"Kimi-K2-Instruct",
            "kimi_k2",
            None,
            "Kimi-K2-Instruct model card: non-thinking; --tool-call-parser kimi_k2",
        ),
        _rule(r"Kimi-K2", "kimi_k2", "kimi_k2", "docs/formats/kimi.md (vLLM v0.30.0 row; thinking K2.x)"),
    ),
    "glm": (
        _rule(
            r"GLM-4\.[56]",
            "glm45",
            "glm45",
            "GLM-4.5/4.6 model cards: --tool-call-parser glm45 --reasoning-parser glm45",
        ),
        _rule(r"GLM-(4\.7|5)", "glm47", "glm47", "docs/formats/glm.md (vLLM v0.30.0 row)"),
    ),
    "minimax": (
        _rule(
            r"MiniMax-M2",
            "minimax_m2",
            "minimax_m2",
            "docs/formats/minimax.md (vLLM v0.30.0 row); the model cards use minimax_m2_append_think, "
            "which keeps <think> in content",
        ),
    ),
    "llama": (
        _rule(r"Llama-4", "llama4_pythonic", None, "docs/formats/llama.md (vLLM v0.30.0 row)"),
        _rule(r"Llama-3", "llama3_json", None, "docs/formats/llama.md (vLLM v0.30.0 row)"),
    ),
    "mistral": (
        _rule(
            r"Magistral|Mistral-Medium-3\.5|Mistral-Small-4|Ministral-3-.*Reasoning",
            "mistral",
            "mistral",
            "reasoning models ([THINK] v13+, <think> v11): --reasoning-parser mistral; docs/formats/mistral.md",
        ),
        _rule(r"mistral|ministral|devstral", "mistral", None, "Mistral model cards: --tool-call-parser mistral"),
    ),
    "gemma4": (_rule(r"gemma-4", "gemma4", "gemma4", "docs/formats/gemma4.md (vLLM v0.30.0 row)"),),
}
"""family slug -> ordered model rules (see :func:`resolve_parsers`)."""

# vllm/config/model.py (ModelConfig.__post_init__, "Set default tokenizer modes
# based on model architecture"): tokenizer_mode="auto" is replaced by these.
ARCH_TOKENIZER_MODES: dict[str, str] = {
    "KimiK3ForConditionalGeneration": "kimi_k3",
    "DeepseekV32ForCausalLM": "deepseek_v32",
    "DeepseekV4ForCausalLM": "deepseek_v4",
    "DeepseekV4ForConditionalGeneration": "deepseek_v4",
    "DeepseekV41ForCausalLM": "deepseek_v41",
}

MAX_TOKENS = 4096
"""``max_tokens`` for ``request.to_sampling_params`` (only affects detokenizer settings)."""


def resolve_parsers(family: str, model: str) -> ModelRule | None:
    """The first rule for ``family`` whose pattern matches ``model``."""
    return next((r for r in PARSERS.get(family, ()) if r.matches(model)), None)


def chat_template_kwargs_for(thinking: bool | None) -> dict[str, Any]:
    """Request ``chat_template_kwargs`` for a fixture's ``thinking`` flag.

    ``None`` means the request sent none (template and parser defaults). A
    bool is sent under both spellings: ``enable_thinking`` (Qwen3, GLM,
    Gemma 4) and ``thinking`` (DeepSeek, Kimi). vLLM's parsers read either
    one, and the renderer drops variables the template does not declare.
    """
    return {} if thinking is None else {"enable_thinking": thinking, "thinking": thinking}


def _imp(name: str) -> Any:
    """Import an engine module lazily (typed ``Any``: vLLM ships no stubs)."""
    return importlib.import_module(name)


def _dist_version(dist: str) -> str | None:
    """Installed version of a distribution in this interpreter, or None."""
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _class_name(obj: Any) -> str:
    """Qualified name of the first non-local class in ``obj``'s MRO (vLLM wraps HF tokenizers locally)."""
    cls = next((c for c in type(obj).__mro__ if "<locals>" not in c.__qualname__), type(obj))
    return f"{cls.__module__}.{cls.__qualname__}"


def _sha256(text: str | bytes) -> str:
    return hashlib.sha256(text.encode("utf-8") if isinstance(text, str) else text).hexdigest()


def _as_ids(x: Any) -> list[int]:
    """``apply_chat_template``/``tokenizer(...)`` output -> flat list of ids."""
    if isinstance(x, Mapping) or hasattr(x, "keys"):
        x = x["input_ids"]
    return [int(i) for i in x]


# --------------------------------------------------------------------------- engine setup


@dataclass
class _Model:
    """A loaded tokenizer plus everything derived from it (cached per repo/revision/mode)."""

    repo: str
    revision: str | None
    resolved_revision: str | None
    requested_mode: str
    tok: Any
    tokenizer_class: str
    model_type: str | None
    architectures: tuple[str, ...]
    detokenizer_class: str | None = None


@dataclass
class _Setup:
    """Per (model, tools, thinking) replay context."""

    model: _Model
    rule: ModelRule
    choice: ParserChoice
    parser_cls: Any
    is_harmony: bool
    tools: list[dict[str, Any]]
    chat_template_kwargs: dict[str, Any]
    prompt_ids: list[int]
    renderer: str
    template: dict[str, Any] = field(default_factory=dict)


class VllmAdapter(Adapter):
    """vLLM 0.30.0 through its unified ``Parser`` (``ParserManager``) path."""

    name: ClassVar[str] = "vllm"
    pinned_version: ClassVar[str] = "0.30.0"

    def __init__(self) -> None:
        self._models: dict[tuple[str, str | None, str], _Model] = {}
        self._setups: dict[str, _Setup] = {}
        self._templates: dict[tuple[str, str | None, str], dict[str, Any]] = {}
        self._prompt_texts: dict[str, str] = {}

    # ------------------------------------------------------------------ identity

    def version(self) -> str:
        try:
            vllm = _imp("vllm")
        except Exception as e:  # ImportError, or torch failing to load
            raise AdapterUnavailable(f"vllm is not importable here: {type(e).__name__}: {e}") from e
        return str(vllm.__version__)

    def engine_details(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            dist: _dist_version(dist)
            for dist in ("transformers", "tokenizers", "torch", "openai-harmony", "mistral-common", "huggingface-hub")
        }
        try:
            vllm_dir = Path(_imp("vllm").__file__).resolve().parent.parent
            sha = vllm_dir / "WHEEL.sha256"
            if sha.is_file():
                out["wheel_sha256"] = sha.read_text(encoding="utf-8").split()[0]
                out["wheel"] = sha.read_text(encoding="utf-8").split()[1]
        except Exception:
            pass
        return out

    # ------------------------------------------------------------------ cheap, engine-free

    def supports(self, family: str, model: str) -> Support:
        if family not in PARSERS:
            return Support(False, f"vLLM {self.pinned_version}: no parser mapping for family {family!r}")
        rule = resolve_parsers(family, model)
        if rule is None:
            return Support(False, f"vLLM {self.pinned_version}: no parser mapping for model {model!r} in {family!r}")
        if rule.choice is None:
            return Support(False, rule.reason or "unsupported")
        return Support(True)

    def static_parser_config(self, raw: ReplayInput) -> dict[str, Any]:
        """The parser configuration that needs no engine: flags, pins, request settings."""
        rule = resolve_parsers(raw.family, raw.model)
        if rule is None or rule.choice is None:
            raise ValueError(f"unsupported: {raw.family} / {raw.model}")
        repo, revision = self._tokenizer_pin(raw)
        return {
            "engine": self.name,
            "pinned_version": self.pinned_version,
            "tool_parser": rule.choice.tool_parser,
            "reasoning_parser": rule.choice.reasoning_parser,
            "parser_source": rule.choice.source,
            "model_rule": rule.pattern,
            "entrypoint": "vllm.parser.parser_manager.ParserManager.get_parser",
            "enable_auto_tools": True,
            "tool_choice": "auto (tools and tool_choice omitted when the fixture offers no tools)",
            "chat_template_kwargs": chat_template_kwargs_for(raw.thinking),
            "tokenizer": {
                "repo": repo,
                "revision": revision,
                "requested_mode": self._requested_mode(raw),
            },
            "model": raw.model,
            "units_source": "fixture.output_token_ids" if raw.token_ids is not None else "engine tokenizer encode",
            "detokenizer": "vllm.v1.engine.detokenizer.IncrementalDetokenizer.from_new_request",
            "stop_token_in_final_delta": False,
            "prompt_messages": [{"role": "user", "content": PLACEHOLDER_USER_MESSAGE}],
            "generation_prompt": raw.generation_prompt,
        }

    # ------------------------------------------------------------------ contract

    def parser_config(self, raw: ReplayInput) -> dict[str, Any]:
        cfg = self.static_parser_config(raw)
        if importlib.util.find_spec("vllm") is None:
            cfg["resolved"] = None  # engine not importable here: nothing was resolved
            return cfg
        model = self._model(raw)
        cfg["tokenizer"].update(
            resolved_revision=model.resolved_revision,
            mode=self._resolved_mode(model),
            tokenizer_class=model.tokenizer_class,
        )
        if model.detokenizer_class is not None:
            cfg["detokenizer_class"] = model.detokenizer_class
        cfg["model_type"] = model.model_type
        cfg["is_harmony"] = model.model_type == "gpt_oss"
        cfg["template"] = self._template_info(model)
        prompt_text = self._prompt_texts.get(self._prompt_key(raw))
        if prompt_text is not None:
            cfg["prompt_tail"] = prompt_text[-80:]
            cfg["generation_prompt_match"] = (
                None if raw.generation_prompt is None else prompt_text.endswith(raw.generation_prompt)
            )
        return cfg

    def units(self, raw: ReplayInput) -> list[int]:
        if raw.token_ids is not None:
            return list(raw.token_ids)
        tok = self._model(raw).tok
        return [int(i) for i in tok.encode(raw.text, add_special_tokens=False)]

    def special_token_ids(self, raw: ReplayInput) -> Collection[int] | None:
        """``tok.all_special_ids`` plus added tokens flagged ``special`` (HF), from vLLM's tokenizer."""
        tok = self._model(raw).tok
        ids = {int(i) for i in tok.all_special_ids}
        decoder = getattr(tok, "added_tokens_decoder", None)
        if isinstance(decoder, Mapping):
            ids.update(int(i) for i, t in decoder.items() if getattr(t, "special", False))
        return frozenset(ids)

    def parse(self, raw: ReplayInput, tools: Sequence[ToolSpec]) -> ParseResult:
        setup = self._setup(raw, tools)
        ids = self.units(raw)
        try:
            request, parser = self._request_and_parser(setup, stream=False)
            (text,) = self._detokenize(setup, request, [ids])
            reasoning, content, tool_calls = parser.parse(
                text, request, enable_auto_tools=True, model_output_token_ids=ids
            )
        except Exception as e:
            return ParseResult(exception=f"{type(e).__name__}: {e}")
        return ParseResult(
            content=content or None,
            reasoning_content=reasoning or None,
            tool_calls=tuple(ParsedToolCall(tc.name, tc.arguments) for tc in (tool_calls or [])),
        )

    def parse_stream(self, raw: ReplayInput, chunks: Sequence[Sequence[int]], tools: Sequence[ToolSpec]) -> ParseResult:
        setup = self._setup(raw, tools)
        groups = [list(c) for c in chunks]
        acc = _StreamAccumulator()
        try:
            request, parser = self._request_and_parser(setup, stream=True)
            texts = self._detokenize(setup, request, groups)
            for i, (delta_text, delta_ids) in enumerate(zip(texts, groups, strict=True)):
                if not delta_text and not delta_ids:
                    continue  # serving.py skips empty outputs
                delta = parser.parse_delta(
                    delta_text,
                    delta_ids,
                    request,
                    prompt_token_ids=setup.prompt_ids,
                    finished=i == len(groups) - 1,
                )
                acc.add(delta)
        except Exception as e:
            return acc.result(exception=f"{type(e).__name__}: {e}")
        return acc.result()

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _tokenizer_pin(raw: ReplayInput) -> tuple[str, str | None]:
        """Tokenizer repo and revision: the fixture's pin, else the family's reference-model pin."""
        if raw.tokenizer is not None:
            return raw.tokenizer.repo, raw.tokenizer.revision
        return raw.model, raw.revision

    def _prompt_key(self, raw: ReplayInput) -> str:
        """Prompt text cache key: tokenizer pin + thinking (the tail does not depend on the tools)."""
        return json.dumps([*self._tokenizer_pin(raw), self._requested_mode(raw), raw.thinking])

    @staticmethod
    def _requested_mode(raw: ReplayInput) -> str:
        return "mistral" if raw.tokenizer_mode == "mistral" else "auto"

    def _model(self, raw: ReplayInput) -> _Model:
        repo, revision = self._tokenizer_pin(raw)
        requested = self._requested_mode(raw)
        key = (repo, revision, requested)
        if key in self._models:
            return self._models[key]
        config, resolved_revision = self._hf_config(repo, revision)
        architectures = tuple(config.get("architectures") or ())
        mode = requested
        if mode == "auto":
            mode = next((ARCH_TOKENIZER_MODES[a] for a in architectures if a in ARCH_TOKENIZER_MODES), "auto")
        get_tokenizer = _imp("vllm.tokenizers").get_tokenizer
        tok = get_tokenizer(
            repo, revision=revision, trust_remote_code=trusts_remote_code(repo, revision), tokenizer_mode=mode
        )
        model = _Model(
            repo=repo,
            revision=revision,
            resolved_revision=resolved_revision,
            requested_mode=mode,
            tok=tok,
            tokenizer_class=_class_name(tok),
            model_type=config.get("model_type"),
            architectures=architectures,
        )
        self._models[key] = model
        return model

    @staticmethod
    def _hf_config(repo: str, revision: str | None) -> tuple[dict[str, Any], str | None]:
        """``config.json`` (architectures, model_type) and the snapshot commit it came from."""
        hf_hub_download = _imp("huggingface_hub").hf_hub_download
        for name in ("config.json", "params.json", "tokenizer_config.json"):
            try:
                path = Path(hf_hub_download(repo, name, revision=revision))
            except Exception:
                continue
            commit = path.parent.name if path.parent.parent.name == "snapshots" else None
            if name != "config.json":
                return {}, commit
            return json.loads(path.read_text(encoding="utf-8")), commit
        return {}, None

    @staticmethod
    def _resolved_mode(model: _Model) -> str:
        """The renderer/tokenizer mode vLLM ends up with (``auto`` resolves to hf or mistral)."""
        if model.requested_mode != "auto":
            return model.requested_mode
        return "mistral" if "MistralTokenizer" in model.tokenizer_class else "hf"

    def _template_info(self, model: _Model) -> dict[str, Any]:
        """Which chat template renders prompts for ``model``, with its sha256."""
        key = (model.repo, model.revision, model.requested_mode)
        if key in self._templates:
            return self._templates[key]
        mode = self._resolved_mode(model)
        info: dict[str, Any]
        if model.model_type == "gpt_oss":
            info = {
                "source": "openai_harmony via vLLM OnlineRenderer._make_request_with_harmony (no Jinja template)",
                "openai_harmony": _dist_version("openai-harmony"),
            }
        elif mode == "mistral":
            inner = getattr(model.tok, "instruct", None)
            version = getattr(getattr(inner, "tokenizer", None), "version", None)
            info = {
                "source": "mistral_common via vLLM MistralTokenizer (no Jinja template)",
                "mistral_common": _dist_version("mistral-common"),
                "tokenizer_version": str(getattr(version, "value", version)) if version is not None else None,
            }
        elif mode in ("deepseek_v32", "deepseek_v4", "deepseek_v41", "kimi_k3"):
            module = {
                "deepseek_v32": "vllm.tokenizers.deepseek_v32_encoding",
                "deepseek_v4": "vllm.tokenizers.deepseek_v4_encoding",
                "deepseek_v41": "vllm.tokenizers.deepseek_v41_encoding",
            }.get(mode)
            if module is not None:
                src = Path(_imp(module).__file__).read_bytes()
                info = {"source": f"vLLM tokenizer_mode={mode} encoder ({module})", "sha256": _sha256(src)}
            else:
                tmpl = getattr(model.tok, "chat_template", None)
                info = {
                    "source": f"vLLM tokenizer_mode={mode} (tokenizer.apply_chat_template, repo encoder)",
                    "sha256": _sha256(tmpl) if isinstance(tmpl, str) else None,
                }
        else:
            try:
                tmpl = model.tok.get_chat_template(None, tools=[{"type": "function", "function": {"name": "x"}}])
            except Exception:
                tmpl = getattr(model.tok, "chat_template", None)
            info = {
                "source": "tokenizer chat template (HF repo)",
                "sha256": _sha256(tmpl) if isinstance(tmpl, str) else None,
            }
        self._templates[key] = info
        return info

    def _setup(self, raw: ReplayInput, tools: Sequence[ToolSpec]) -> _Setup:
        rule = resolve_parsers(raw.family, raw.model)
        if rule is None or rule.choice is None:
            raise ValueError(f"vLLM adapter does not support {raw.family} / {raw.model}")
        ctk = chat_template_kwargs_for(raw.thinking)
        tool_list = [json.loads(json.dumps(t)) for t in tools]
        repo, revision = self._tokenizer_pin(raw)
        key = json.dumps([repo, revision, self._requested_mode(raw), rule.pattern, tool_list, ctk], sort_keys=True)
        if key in self._setups:
            return self._setups[key]
        model = self._model(raw)
        is_harmony = model.model_type == "gpt_oss"
        parser_cls = _imp("vllm.parser.parser_manager").ParserManager.get_parser(
            tool_parser_name=rule.choice.tool_parser,
            reasoning_parser_name=rule.choice.reasoning_parser,
            enable_auto_tools=True,
            model_name=model.repo,
            is_harmony=is_harmony,
        )
        setup = _Setup(
            model=model,
            rule=rule,
            choice=rule.choice,
            parser_cls=parser_cls,
            is_harmony=is_harmony,
            tools=tool_list,
            chat_template_kwargs=ctk,
            prompt_ids=[],
            renderer=self._resolved_mode(model),
            template=self._template_info(model),
        )
        setup.prompt_ids = self._render_prompt(setup)
        self._prompt_texts[self._prompt_key(raw)] = model.tok.decode(setup.prompt_ids, skip_special_tokens=False)
        self._setups[key] = setup
        return setup

    def _make_request(self, setup: _Setup, *, stream: bool) -> Any:
        """The client request. With no tools, ``tools``/``tool_choice`` are omitted (vLLM rejects
        ``tools=[]``), so ``tool_choice`` takes its default ``"none"``, as for a real client."""
        protocol = _imp("vllm.entrypoints.openai.chat_completion.protocol")
        tool_fields: dict[str, Any] = {"tools": setup.tools, "tool_choice": "auto"} if setup.tools else {}
        return protocol.ChatCompletionRequest(
            model=setup.model.repo,
            messages=[{"role": "user", "content": PLACEHOLDER_USER_MESSAGE}],
            stream=stream,
            chat_template_kwargs=setup.chat_template_kwargs or None,
            **tool_fields,
        )

    @staticmethod
    def _parser_kwargs(request: Any) -> dict[str, Any]:
        """``OpenAIServingChat._effective_chat_template_kwargs`` (no server defaults)."""
        return dict(request.build_chat_params(None, "auto").chat_template_kwargs)

    def _request_and_parser(self, setup: _Setup, *, stream: bool) -> tuple[Any, Any]:
        """A fresh request adjusted as the renderer does, and the parser instance serving uses."""
        request = self._make_request(setup, stream=stream)
        tok = setup.model.tok
        ctk = self._parser_kwargs(request)
        if setup.is_harmony:
            # OnlineRenderer.render_chat (harmony branch): always adjusts, without chat_template_kwargs.
            setup.parser_cls(tok, request.tools, model_config=None).adjust_request(request=request)
        elif self._should_adjust_request(setup, request):
            request = setup.parser_cls(tok, request.tools, model_config=None, chat_template_kwargs=ctk).adjust_request(
                request=request
            )
        parser = setup.parser_cls(tok, request.tools, chat_template_kwargs=ctk, model_config=None)
        return request, parser

    @staticmethod
    def _should_adjust_request(setup: _Setup, request: Any) -> bool:
        """``OnlineRenderer.preprocess_chat``'s condition for calling ``adjust_request``."""
        mistral = _imp("vllm.utils.mistral")
        tok = setup.model.tok
        mistral_grammar = (
            setup.parser_cls.tool_parser_cls is not None
            and mistral.is_mistral_tool_parser(setup.parser_cls.tool_parser_cls)
            and mistral.is_mistral_tokenizer(tok)
            and bool(tok.supports_grammar)
        )
        tool_choice = getattr(request, "tool_choice", "none")
        return setup.parser_cls.reasoning_parser_cls is not None or tool_choice != "none" or mistral_grammar

    def _render_prompt(self, setup: _Setup) -> list[int]:
        """Prompt ids exactly as the vLLM renderer for this tokenizer mode builds them."""
        request = self._make_request(setup, stream=False)
        tok = setup.model.tok
        # OnlineRenderer.render_chat: tool dicts are the pydantic tools' model_dump(), or None.
        tool_dicts = None if request.tools is None else [t.model_dump() for t in request.tools]
        params = request.build_chat_params(None, "auto").with_defaults(
            {"tools": tool_dicts, "tokenize": setup.renderer == "mistral"}
        )
        kwargs = params.get_apply_chat_template_kwargs()
        messages: list[dict[str, Any]] = [{"role": "user", "content": PLACEHOLDER_USER_MESSAGE}]
        conversation = [dict(m) for m in messages]
        mode = setup.renderer
        raw_prompt: Any
        if setup.is_harmony:
            # OnlineRenderer.render_chat (harmony branch): openai_harmony preamble + render_for_completion.
            renderer_cls = _imp("vllm.renderers.online_renderer").OnlineRenderer
            fake_self = SimpleNamespace(supports_browsing=False, supports_code_interpreter=False)
            _, (engine_input,) = renderer_cls._make_request_with_harmony(fake_self, request, tool_dicts is not None)
            return _as_ids(engine_input["prompt_token_ids"])
        if mode == "mistral":
            raw_prompt = _imp("vllm.renderers.mistral").safe_apply_chat_template(tok, messages, **kwargs)
        elif mode in ("deepseek_v32", "deepseek_v4", "deepseek_v41"):
            renderer_mod = "vllm.renderers.deepseek_v32" if mode == "deepseek_v32" else "vllm.renderers.deepseek_v4"
            cls_name = "DeepseekV32Renderer" if mode == "deepseek_v32" else "DeepseekV4Renderer"
            renderer_cls = getattr(_imp(renderer_mod), cls_name)
            raw_prompt = renderer_cls._apply_chat_template(
                SimpleNamespace(get_tokenizer=lambda: tok), conversation=conversation, messages=messages, **kwargs
            )
        elif mode == "kimi_k3":
            k3 = _imp("vllm.renderers.kimi_k3")
            raw_prompt = k3.KimiK3Renderer._apply_chat_template(
                SimpleNamespace(get_tokenizer=lambda: tok), k3._normalize_k3_tool_messages(conversation), params
            )
        else:
            hf = _imp("vllm.renderers.hf")
            model_config = SimpleNamespace(
                hf_config=SimpleNamespace(model_type=setup.model.model_type),
                revision=setup.model.revision,
                code_revision=None,
                trust_remote_code=trusts_remote_code(setup.model.repo, setup.model.revision),
            )
            raw_prompt = hf.safe_apply_chat_template(model_config, tok, conversation, **kwargs)
        if isinstance(raw_prompt, str):
            # BaseRenderer._tokenize_prompt: tokenizer(prompt, add_special_tokens=request.add_special_tokens)
            return _as_ids(tok(raw_prompt, add_special_tokens=request.add_special_tokens))
        return _as_ids(raw_prompt)

    @staticmethod
    def _detokenize(setup: _Setup, request: Any, groups: Sequence[Sequence[int]]) -> list[str]:
        """Text delta per id group from vLLM's v1 IncrementalDetokenizer (as OutputProcessor)."""
        detok_mod = _imp("vllm.v1.engine.detokenizer")
        sampling_params = request.to_sampling_params(MAX_TOKENS, {})
        engine_request = SimpleNamespace(
            request_id="canitoolcall",
            sampling_params=sampling_params,
            prompt_token_ids=list(setup.prompt_ids),
            prompt_embeds=None,
        )
        det = detok_mod.IncrementalDetokenizer.from_new_request(setup.model.tok, engine_request)
        setup.model.detokenizer_class = _class_name(det)
        out: list[str] = []
        for i, g in enumerate(groups):
            det.update(list(g), False)
            out.append(det.get_next_output_text(i == len(groups) - 1, True))
        return out


class _StreamAccumulator(StreamAccumulator):
    """The shared OpenAI-client accumulator (DESIGN.md rule 8), fed vLLM ``DeltaMessage``s."""

    def add(self, delta: Any) -> None:
        if delta is None:
            return
        self.add_content(getattr(delta, "content", None))
        self.add_reasoning(getattr(delta, "reasoning", None))
        for tc in getattr(delta, "tool_calls", None) or []:
            fn = tc.function
            self.add_tool_call(int(tc.index), fn.name if fn else None, fn.arguments if fn else None)

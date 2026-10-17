"""llama.cpp adapter: replays fixtures through llama.cpp's real chat parser
(``common/chat.h``) via a small compiled C++ harness, the way ``llama-server``
drives it.

Owner: llama.cpp adapter builder (see docs/DESIGN.md). The harness source lives
in ``harnesses/llamacpp/`` and is built by ``scripts/engines/llamacpp.sh``
against llama.cpp pinned at :attr:`LlamaCppAdapter.pinned_version` into
``.engines/llamacpp/`` (gitignored). No model weights are involved: the parser
comes from the chat template, and text comes from a vocab-only GGUF made by
``scripts/engines/gguf_vocab.sh`` (``convert_hf_to_gguf.py --vocab-only``).

What the harness does for every fixture (one long-lived process per run,
JSON lines over stdin/stdout, see ``harnesses/llamacpp/replay.cpp``):

* ``common_chat_templates_init(vocab_only_model, template)`` and
  ``common_chat_templates_apply(inputs{messages, tools, reasoning_format,
  enable_thinking, ...})`` build ``common_chat_params`` (PEG parser, preserved
  tokens, generation prompt, additional stops), as ``oaicompat_chat_params_parse``
  does.
* Token ids are detokenized EXACTLY like the server:
  ``common_token_to_piece(vocab, id, special = id in preserved_tokens)``. GGUF
  CONTROL tokens are never approximated with HF ``special`` flags (wrong for
  Gemma 4's ``<|"|>``).
* The server slot's text handling is emulated per token: incomplete UTF-8 is
  held back (``validate_utf8``), partial stop strings are held back, a full stop
  string cuts the text, an end-of-generation token stops generation. The stop
  token that ended the real generation (``ReplayInput.stop_tokens``) is fed as
  the final server step, except for ``truncated`` fixtures (cut by
  ``max_tokens``, so no stop token was ever generated).
* Non-streaming: ``common_chat_parse(generated_text, is_partial=false)``.
* Streaming, exactly like ``task_result_state::update_chat_msg``: after each
  chunk, re-parse the accumulated sent text with ``is_partial=true`` and keep
  ``compute_diffs``; the final response re-parses with ``is_partial=false``.
  The deltas are accumulated OpenAI-client style here (:func:`accumulate`).

Template source (contract rule 7). ``llama-server`` uses the template embedded
in the GGUF, which ``convert_hf_to_gguf.py`` copies from the HF repo at the
converted revision: source ``gguf`` (the default). llama.cpp also ships
curated copies in ``models/templates/`` that users pass with
``--chat-template-file``: source ``llamacpp``. They can disagree (the HF
DeepSeek-V3.1 template yields a parser that rejects the model's own tool calls,
the llama.cpp copy parses them), so both are recorded: the primary source (set
with ``$CANITOOLCALL_LLAMACPP_TEMPLATE``) produces the results, and
``parser_config["template_alternatives"]`` holds the other source's
non-streaming outcome for the same fixture.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, ClassVar

from canitoolcall.adapters.base import (
    PLACEHOLDER_USER_MESSAGE,
    Adapter,
    AdapterUnavailable,
    ReplayInput,
    Support,
    ToolSpec,
)
from canitoolcall.chunking import TokensPerStep
from canitoolcall.results import ParsedToolCall, ParseResult, StreamAccumulator

HARNESS_PROTOCOL = 1
HARNESS_ENV = "CANITOOLCALL_LLAMACPP_HARNESS"
"""Path to the compiled harness (default ``.engines/llamacpp/build/canitoolcall-llamacpp``)."""
SOURCE_ENV = "CANITOOLCALL_LLAMACPP_SRC"
"""llama.cpp checkout (default ``.engines/llamacpp/llama.cpp``); used for ``models/templates``."""
GGUF_DIR_ENV = "CANITOOLCALL_GGUF_DIR"
"""Directory of vocab-only GGUFs (default ``.engines/gguf``)."""
TEMPLATE_ENV = "CANITOOLCALL_LLAMACPP_TEMPLATE"
"""Primary template source: ``gguf`` (default) or ``llamacpp``."""

TEMPLATE_SOURCES = ("gguf", "llamacpp")

REASONING_FORMAT = "deepseek"
"""``llama-server``'s default ``--reasoning-format`` (``common_params::reasoning_format``)."""

PROMPT_MESSAGES: tuple[Mapping[str, Any], ...] = ({"role": "user", "content": PLACEHOLDER_USER_MESSAGE},)
"""The request's messages. Only the tail of the rendered prompt (the generation
prompt) reaches the parser; the fixture's tools are passed as the request's tools."""

FAMILIES: Mapping[str, str] = {
    # llama.cpp has no per-family parser names: common_chat_templates_apply derives
    # the parser from the chat template, via a specialized handler when the template
    # matches one (common/chat.cpp common_chat_try_specialized_template) and the
    # generic autoparser otherwise. The recorded `format` + template sha256 pin it.
    "qwen3-hermes": "autoparser (from the chat template)",
    "qwen3-xml": "specialized handler 'Qwen3-Coder'",
    "gpt-oss": "specialized handler 'GPT-OSS'",
    "deepseek": "specialized handler 'DeepSeek V3.2/V4' for V3.2/V4 templates, autoparser otherwise",
    "kimi": "specialized handlers 'Kimi K2 Thinking'/'Kimi K3', autoparser otherwise",
    "glm": "autoparser (from the chat template)",
    "llama": "autoparser (from the chat template)",
    "mistral": "specialized handler 'Ministral/Magistral Large 3', autoparser otherwise",
    "gemma4": "specialized handler (format peg-gemma4)",
}
"""Family slug -> how llama.cpp (at the pin) builds that family's parser."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def gguf_stem(repo: str) -> str:
    """``org/model`` -> ``org--model`` (file stem used by ``scripts/engines/gguf_vocab.sh``)."""
    return repo.replace("/", "--")


def llamacpp_template_file(src_dir: Path, repo: str) -> Path | None:
    """llama.cpp's curated copy of ``repo``'s template, when one is named after it.

    Only exact file-name matches count (``<org>-<model>.jinja`` or
    ``<model>.jinja`` in ``models/templates``): a template of a sibling model is
    never substituted.
    """
    org, _, name = repo.partition("/")
    tdir = src_dir / "models" / "templates"
    for cand in (f"{org}-{name}.jinja", f"{name}.jinja"):
        p = tdir / cand
        if p.is_file():
            return p
    return None


def accumulate(deltas: Sequence[Mapping[str, Any]], exception: str | None = None) -> ParseResult:
    """Merge ``common_chat_msg_diff`` deltas OpenAI-client style (DESIGN.md rule 8).

    Uses the shared :class:`~canitoolcall.results.StreamAccumulator`: content and
    reasoning deltas are concatenated; tool-call deltas are merged by ``index``
    with name and argument fragments concatenated, as openai-python does. The
    server sends the name again when a call's id changes (``compute_diffs``),
    and a client then sees it repeated. Fields that never received a delta stay
    ``None``.
    """
    acc = StreamAccumulator()
    for d in deltas:
        acc.add_reasoning(d.get("reasoning_content"))
        acc.add_content(d.get("content"))
        if "index" in d:
            acc.add_tool_call(int(d["index"]), d.get("name"), d.get("arguments"))
    return acc.result(exception)


def message_result(msg: Mapping[str, Any]) -> ParseResult:
    """A non-streaming ``common_chat_msg`` as the OpenAI response carries it.

    ``common_chat_msg::to_json_oaicompat`` always sends ``content`` (``""`` when
    empty) and omits an empty ``reasoning_content``.
    """
    return ParseResult(
        content=msg.get("content", ""),
        reasoning_content=msg.get("reasoning_content") or None,
        tool_calls=tuple(ParsedToolCall(t["name"], t["arguments"]) for t in msg.get("tool_calls", [])),
    )


class HarnessError(RuntimeError):
    """The compiled harness failed (a harness problem, never a parse outcome)."""


class Harness:
    """One long-lived ``canitoolcall-llamacpp`` process speaking JSON lines."""

    def __init__(self, binary: Path) -> None:
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise AdapterUnavailable(f"llama.cpp harness not built: {binary} (run scripts/engines/llamacpp.sh)")
        self.binary = binary
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None

    def _start(self) -> subprocess.Popen[str]:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                [str(self.binary)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,  # harness diagnostics go to the worker's stderr log
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        return self._proc

    def request(self, req: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            proc = self._start()
            stdin: IO[str] | None = proc.stdin
            stdout: IO[str] | None = proc.stdout
            assert stdin is not None and stdout is not None
            try:
                stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
                stdin.flush()
                line = stdout.readline()
            except (BrokenPipeError, OSError) as e:
                self._proc = None
                raise HarnessError(f"llama.cpp harness died: {e}") from e
            if not line:
                code = proc.wait()
                self._proc = None
                raise HarnessError(f"llama.cpp harness exited with code {code}")
        reply: dict[str, Any] = json.loads(line)
        if not reply.get("ok"):
            raise HarnessError(f"llama.cpp harness: {reply.get('error')}")
        return reply

    def close(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            assert proc.stdin is not None
            proc.stdin.write('{"op":"shutdown"}\n')
            proc.stdin.flush()
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
            proc.wait()


@dataclass
class _Vocab:
    """A vocab-only GGUF and its provenance sidecar (from gguf_vocab.sh)."""

    repo: str
    path: Path
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TemplateChoice:
    """Which chat template the parser is built from, pinned by sha256."""

    source: str
    """``gguf`` (the GGUF's tokenizer.chat_template, i.e. the HF template at the
    converted revision) or ``llamacpp`` (llama.cpp's ``models/templates`` copy)."""
    text: str | None
    """Override passed to ``common_chat_templates_init``; ``None`` = the GGUF's own."""
    sha256: str
    path: str
    reason: str | None = None
    """Why this source was used instead of the requested one, if it was."""


@dataclass
class _Case:
    """Harness facts cached per fixture, shared by parse/parse_stream/parser_config."""

    config: dict[str, Any]
    text: str | None
    tools_sha256: str
    alternatives: dict[str, Any] | None = None


class LlamaCppAdapter(Adapter):
    name: ClassVar[str] = "llamacpp"
    pinned_version: ClassVar[str] = "a25c9865fe03c954c93fd755b5d79ae86ba99750"
    supports_text_deltas: ClassVar[bool] = True
    tokens_per_step: ClassVar[TokensPerStep] = "one"
    """llama-server sends one partial response per sampled token (``process_token``)."""

    def __init__(
        self,
        *,
        harness: Path | None = None,
        source_dir: Path | None = None,
        gguf_dir: Path | None = None,
        template_source: str | None = None,
    ) -> None:
        root = _repo_root()
        self.harness_path = harness or Path(
            os.environ.get(HARNESS_ENV) or root / ".engines" / "llamacpp" / "build" / "canitoolcall-llamacpp"
        )
        self.source_dir = source_dir or Path(os.environ.get(SOURCE_ENV) or root / ".engines" / "llamacpp" / "llama.cpp")
        self.gguf_dir = gguf_dir or Path(os.environ.get(GGUF_DIR_ENV) or root / ".engines" / "gguf")
        self.template_source = template_source or os.environ.get(TEMPLATE_ENV) or "gguf"
        if self.template_source not in TEMPLATE_SOURCES:
            raise ValueError(f"template source must be one of {TEMPLATE_SOURCES}, not {self.template_source!r}")
        self._harness: Harness | None = None
        self._hello: dict[str, Any] | None = None
        self._cases: dict[str, _Case] = {}
        self._gguf_templates: dict[Path, tuple[bool, str]] = {}
        self._template_index: dict[str, str] | None = None
        self._special: dict[Path, frozenset[int]] = {}

    # -- engine identity ---------------------------------------------------------------

    def _h(self) -> Harness:
        if self._harness is None:
            self._harness = Harness(self.harness_path)
        return self._harness

    def _hello_reply(self) -> dict[str, Any]:
        if self._hello is None:
            reply = self._h().request({"op": "hello"})
            if reply.get("protocol") != HARNESS_PROTOCOL:
                raise AdapterUnavailable(
                    f"harness protocol {reply.get('protocol')} != {HARNESS_PROTOCOL}; "
                    "rebuild with scripts/engines/llamacpp.sh"
                )
            self._hello = reply
        return self._hello

    def version(self) -> str:
        """The llama.cpp commit the harness was built from (verified against the pin)."""
        commit = str(self._hello_reply().get("commit", ""))
        if not commit or not self.pinned_version.startswith(commit):
            raise AdapterUnavailable(
                f"harness built from llama.cpp {commit or '?'}, expected {self.pinned_version}; "
                "rebuild with scripts/engines/llamacpp.sh"
            )
        return self.pinned_version[:8]

    def commit(self) -> str | None:
        return self.pinned_version

    def engine_details(self) -> dict[str, Any]:
        hello = self._hello_reply()
        return {
            "build_info": hello.get("build_info"),
            "harness": "harnesses/llamacpp/replay.cpp",
            "harness_protocol": hello.get("protocol"),
            "template_source": self.template_source,
            "reasoning_format": REASONING_FORMAT,
            "detokenizer": "common_token_to_piece(vocab-only GGUF, special = id in preserved_tokens)",
        }

    # -- support / configuration -------------------------------------------------------

    def supports(self, family: str, model: str) -> Support:
        if family not in FAMILIES:
            return Support(False, f"no llama.cpp configuration for family {family!r}")
        if family == "llama" and "Llama-4" in model:
            # llama.cpp derives its parser from the chat template. The only ungated Llama 4
            # tokenizer (the fixtures' pin) ships a chat template that is not Meta's, and
            # llama.cpp's models/templates has no Llama 4 copy (fixtures/llama/family.json).
            return Support(
                False,
                "no authoritative Llama 4 chat template: meta-llama repos are gated and the ungated "
                "mirror's template is not Meta's; llama.cpp derives its parser from the template",
            )
        err = self.gguf_dir / f"{gguf_stem(model)}.vocab.error.json"
        if err.is_file():
            try:
                info = json.loads(err.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                info = {}
            if info.get("stage") == "convert":
                first = str(info.get("error", "")).strip().splitlines()[-1:] or ["unknown error"]
                return Support(False, f"llama.cpp convert_hf_to_gguf.py cannot convert {model}: {first[0]}")
        return Support(True)

    def _vocab(self, raw: ReplayInput) -> _Vocab:
        repo = raw.tokenizer.repo if raw.tokenizer is not None else raw.model
        path = self.gguf_dir / f"{gguf_stem(repo)}.vocab.gguf"
        if not path.is_file():
            err = path.with_name(f"{gguf_stem(repo)}.vocab.error.json")
            detail = f" ({err.read_text(encoding='utf-8')[:500]})" if err.is_file() else ""
            raise AdapterUnavailable(
                f"no vocab-only GGUF for {repo} at {path}; run scripts/engines/gguf_vocab.sh --all{detail}"
            )
        meta_path = path.with_name(f"{gguf_stem(repo)}.vocab.json")
        meta: Mapping[str, Any] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        return _Vocab(repo, path, meta)

    def _gguf_template(self, vocab: _Vocab) -> tuple[bool, str]:
        """(GGUF embeds a chat template, sha256 of the template llama.cpp uses without an override)."""
        if vocab.path not in self._gguf_templates:
            reply = self._h().request(
                {
                    "op": "replay",
                    "vocab": str(vocab.path),
                    "template": None,
                    "messages": list(PROMPT_MESSAGES),
                    "tools": [],
                    "ids": [],
                    "nonstream": False,
                    "streams": [],
                    "return_template": True,
                }
            )
            cfg = reply.get("config", {})
            stored = cfg.get("gguf_template")
            # sha256 of the template bytes as stored in the GGUF (comparable with the HF file
            # and models/templates); without one, of the chatml fallback llama.cpp uses
            text = stored if isinstance(stored, str) else str(cfg.get("template_source", ""))
            self._gguf_templates[vocab.path] = (bool(cfg.get("gguf_has_template")), _sha256(text.encode()))
        return self._gguf_templates[vocab.path]

    def _candidates(self, raw: ReplayInput, vocab: _Vocab) -> dict[str, TemplateChoice | str]:
        """Every template source for this fixture: a choice, or why it is unavailable."""
        out: dict[str, TemplateChoice | str] = {}
        has_template, sha = self._gguf_template(vocab)
        if has_template:
            out["gguf"] = TemplateChoice("gguf", None, sha, f"{vocab.path.name}: tokenizer.chat_template")
        else:
            out["gguf"] = f"{vocab.path.name} has no tokenizer.chat_template (llama.cpp would fall back to chatml)"
        copies = [p for p in (llamacpp_template_file(self.source_dir, r) for r in (raw.model, vocab.repo)) if p]
        if copies:
            text = copies[0].read_text(encoding="utf-8")
            out["llamacpp"] = TemplateChoice(
                "llamacpp", text, _sha256(text.encode()), f"models/templates/{copies[0].name}"
            )
        else:
            out["llamacpp"] = f"llama.cpp models/templates has no copy named after {raw.model}"
        return out

    def _llamacpp_template_by_sha(self, sha: str) -> str | None:
        """``models/templates/<file>`` whose bytes hash to ``sha``, if any.

        convert_hf_to_gguf.py embeds some of these itself (e.g. Mistral community
        templates, DeepSeek-V4), so a GGUF template can be one of them verbatim.
        """
        if self._template_index is None:
            tdir = self.source_dir / "models" / "templates"
            files = sorted(tdir.glob("*.jinja")) if tdir.is_dir() else []
            self._template_index = {_sha256(f.read_bytes()): f"models/templates/{f.name}" for f in files}
        return self._template_index.get(sha)

    def _choose(self, raw: ReplayInput, vocab: _Vocab) -> TemplateChoice:
        """The primary template: the requested source when it exists, else the other one.

        A GGUF without a chat template makes llama-server fall back to chatml; for such
        repos (e.g. DeepSeek-V3.2, whose HF repo ships a Python encoder instead of a
        Jinja template) llama.cpp's docs point to ``--chat-template-file
        models/templates/...``, so that copy is used and the reason recorded.
        """
        cands = self._candidates(raw, vocab)
        want = cands[self.template_source]
        if isinstance(want, TemplateChoice):
            return want
        for src in TEMPLATE_SOURCES:
            alt = cands[src]
            if isinstance(alt, TemplateChoice):
                return dataclasses.replace(alt, reason=want)
        # Neither exists: exactly what llama-server does with this GGUF (chatml fallback).
        _, sha = self._gguf_template(vocab)
        return TemplateChoice("gguf", None, sha, "chatml fallback (common_chat_templates_init)", reason=want)

    def _request(
        self,
        raw: ReplayInput,
        tools: Sequence[ToolSpec],
        *,
        template: TemplateChoice,
        nonstream: bool,
        streams: Sequence[Sequence[int] | Mapping[str, Any]] = (),
        units: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        vocab = self._vocab(raw)
        req: dict[str, Any] = {
            "op": "replay",
            "vocab": str(vocab.path),
            "template": template.text,
            "messages": list(PROMPT_MESSAGES),
            "tools": list(tools),
            "reasoning_format": REASONING_FORMAT,
            "chat_template_kwargs": self._kwargs(raw),
            "ids": list(units if units is not None else self.units(raw)),
            # A truncated generation ended at max_tokens, with no stop token.
            "end_tokens": list(raw.replay_end_tokens),
            "nonstream": nonstream,
            "streams": [dict(s) if isinstance(s, Mapping) else list(s) for s in streams],
        }
        response = self._h().request(req)
        if raw.finish is not None and response.get("ok") and "engine_error" not in response:
            config = response.get("config", {})
            if "end_token" not in config or config["end_token"] != raw.finish.token:
                raise ValueError("harness did not replay the explicit finish.token")
        return response

    @staticmethod
    def _kwargs(raw: ReplayInput) -> dict[str, Any]:
        """Request ``chat_template_kwargs``: the fixture's thinking mode, if pinned.

        Sent under both spellings templates use, like the vLLM adapter:
        ``enable_thinking`` (Qwen3, GLM, Gemma 4; llama-server also maps it to
        ``inputs.enable_thinking``) and ``thinking`` (DeepSeek, Kimi K3). ``None``
        leaves llama-server's default (thinking on iff the template supports it).
        """
        return {} if raw.thinking is None else {"enable_thinking": raw.thinking, "thinking": raw.thinking}

    def _remember(self, raw: ReplayInput, tools: Sequence[ToolSpec], reply: Mapping[str, Any]) -> None:
        tools_sha = _sha256(json.dumps(list(tools), sort_keys=True, ensure_ascii=False).encode())
        prev = self._cases.get(raw.fixture_id)
        case = _Case(dict(reply.get("config", {})), reply.get("text"), tools_sha)
        if reply.get("engine_error"):
            case.config["engine_error"] = reply["engine_error"]
        if prev is not None and prev.tools_sha256 == tools_sha:
            case.alternatives = prev.alternatives
        if case.alternatives is None:
            case.alternatives = self._alternatives(raw, tools)
        self._cases = {raw.fixture_id: case}  # only the fixture being replayed

    def _alternatives(self, raw: ReplayInput, tools: Sequence[ToolSpec]) -> dict[str, Any]:
        """Non-streaming outcome with every other template source (rule 7: record both)."""
        vocab = self._vocab(raw)
        primary = self._choose(raw, vocab)
        out: dict[str, Any] = {}
        for source, cand in self._candidates(raw, vocab).items():
            if source == primary.source:
                continue
            if not isinstance(cand, TemplateChoice):
                out[source] = {"available": False, "reason": cand}
                continue
            reply = self._request(raw, tools, template=cand, nonstream=True)
            out[source] = {
                "available": True,
                "path": cand.path,
                "sha256": cand.sha256,
                "format": reply.get("config", {}).get("format"),
                "nonstream": self._nonstream_result(reply).to_dict(),
            }
        return out

    def supports_replay(self, raw: ReplayInput) -> Support:
        # _request sends the exact terminal token (or none for a length finish).
        if raw.finish is not None and not raw.assumes:
            return self.supports(raw.family, raw.model)
        return super().supports_replay(raw)

    def parser_config(self, raw: ReplayInput) -> dict[str, Any]:
        vocab = self._vocab(raw)
        template = self._choose(raw, vocab)
        case = self._cases.get(raw.fixture_id)
        cfg: dict[str, Any] = {
            "engine_commit": self.pinned_version,
            "chat_parser": "common_chat_templates_apply -> common_chat_parse (PEG)",
            "family_handler": FAMILIES.get(raw.family),
            "reasoning_format": REASONING_FORMAT,
            "chat_template_kwargs": self._kwargs(raw),
            "template_source": template.source,
            "template_source_requested": self.template_source,
            "template_source_reason": template.reason,
            "template_path": template.path,
            "template_sha256": template.sha256,
            "template_identical_to": self._llamacpp_template_by_sha(template.sha256),
            "model": raw.model,
            "tokenizer": f"{vocab.repo}@{vocab.meta.get('revision', '?')}",
            "tokenizer_mode": "gguf-vocab-only",
            "vocab_gguf": vocab.path.name,
            "vocab_gguf_sha256": vocab.meta.get("sha256"),
            "vocab_gguf_llama_cpp_commit": vocab.meta.get("llama_cpp_commit"),
            "vocab_gguf_converter_env": vocab.meta.get("converter_env"),
            "vocab_gguf_converter_patches": vocab.meta.get("converter_patches") or None,
            "fixture_tokenizer": f"{raw.tokenizer.repo}@{raw.tokenizer.revision}" if raw.tokenizer else None,
            "fixture_generation_prompt": raw.generation_prompt,
        }
        if case is None:
            cfg["harness"] = None  # no parse ran yet for this fixture
            return cfg
        hc = case.config
        for key in (
            "format",
            "generation_prompt",
            "enable_thinking",
            "template_supports_thinking",
            "parallel_tool_calls",
            "template_variant",
            "preserved_tokens",
            "additional_stops",
            "thinking_start_tag",
            "thinking_end_tags",
            "end_token",
            "eog_positions",
            "engine_error",
        ):
            if key in hc:
                cfg[key] = hc[key]
        cfg["tools_sha256"] = case.tools_sha256
        if case.text is not None:
            cfg["detokenized_matches_raw_output"] = case.text == raw.text
        cfg["template_alternatives"] = case.alternatives
        return cfg

    # -- replay ------------------------------------------------------------------------

    def units(self, raw: ReplayInput) -> list[int]:
        if raw.token_ids is not None:
            return list(raw.token_ids)
        vocab = self._vocab(raw)
        reply = self._h().request({"op": "tokenize", "vocab": str(vocab.path), "text": raw.text})
        return [int(i) for i in reply["ids"]]

    def special_token_ids(self, raw: ReplayInput) -> frozenset[int]:
        """llama.cpp's special tokens: vocab attrs CONTROL | USER_DEFINED | UNKNOWN.

        These are the tokens ``llama_vocab`` caches as special (matched atomically
        by ``common_tokenize(..., parse_special=true)``), read from the vocab-only GGUF.
        """
        vocab = self._vocab(raw)
        if vocab.path not in self._special:
            reply = self._h().request({"op": "special_ids", "vocab": str(vocab.path)})
            self._special[vocab.path] = frozenset(int(i) for i in reply["ids"])
        return self._special[vocab.path]

    @staticmethod
    def _nonstream_result(reply: Mapping[str, Any]) -> ParseResult:
        if reply.get("engine_error"):
            return ParseResult(exception=f"LlamaCppError: {reply['engine_error']}")
        ns = reply["nonstream"]
        if "exception" in ns:
            return ParseResult(exception=f"LlamaCppError: {ns['exception']}")
        return message_result(ns["msg"])

    def parse(self, raw: ReplayInput, tools: Sequence[ToolSpec]) -> ParseResult:
        reply = self._request(raw, tools, template=self._choose(raw, self._vocab(raw)), nonstream=True)
        self._remember(raw, tools, reply)
        return self._nonstream_result(reply)

    def parse_stream(self, raw: ReplayInput, chunks: Sequence[Sequence[int]], tools: Sequence[ToolSpec]) -> ParseResult:
        units = [i for c in chunks for i in c]
        reply = self._request(
            raw,
            tools,
            template=self._choose(raw, self._vocab(raw)),
            nonstream=False,
            streams=[[len(c) for c in chunks]],
            units=units,
        )
        self._remember(raw, tools, reply)
        return self._stream_result(reply)

    @staticmethod
    def _stream_result(reply: Mapping[str, Any]) -> ParseResult:
        if reply.get("engine_error"):
            return ParseResult(exception=f"LlamaCppError: {reply['engine_error']}")
        (stream,) = reply["streams"]
        exc = stream.get("exception")
        return accumulate(stream.get("deltas", []), f"LlamaCppError: {exc}" if exc else None)

    def parse_stream_text(self, raw: ReplayInput, deltas: Sequence[str], tools: Sequence[ToolSpec]) -> ParseResult:
        """Synthetic stress path: each text delta is one server step (``char:<seed>``).

        Mirrors llama.cpp's own tests/test-chat.cpp, which streams per character.
        Everything else (hold-back of incomplete UTF-8 and partial stops, the stop
        token step, re-parse + ``compute_diffs``) is the same as :meth:`parse_stream`.
        """
        reply = self._request(
            raw, tools, template=self._choose(raw, self._vocab(raw)), nonstream=False, streams=[{"text": list(deltas)}]
        )
        self._remember(raw, tools, reply)
        return self._stream_result(reply)

    def close(self) -> None:
        if self._harness is not None:
            self._harness.close()
            self._harness = None

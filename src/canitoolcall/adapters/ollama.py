"""Ollama adapter: replays fixtures through Ollama's real built-in output parsers
(``model/parsers``, including the ``harmony`` handler) via two small compiled
harnesses, reproducing Ollama's serving path at the pinned commit.

Owner: Ollama adapter builder (see docs/DESIGN.md). The harness sources live in
``harnesses/ollama/`` and ``scripts/engines/ollama.sh`` builds them into
``.engines/ollama/bin/``. The adapter itself uses only the standard library.

How Ollama serves a chat at the pin, and what each part of this adapter mirrors:

1. **Runner.** Every GGUF model is served by an upstream ``llama-server``
   subprocess built from the llama.cpp tag in Ollama's ``LLAMA_CPP_VERSION``
   (``llm/server.go``). Ollama sends the parser's ``PreservedTokens()`` as
   ``preserved_tokens``. llama-server renders each sampled token with
   ``common_token_to_piece(vocab, id, special = id in preserved_tokens)``,
   holds back an incomplete UTF-8 tail, and streams one SSE event per token.
   Its final ``stop`` event carries no text (``tools/server/server-context.cpp``).
   ``ctc-detok`` renders the pieces from the shared vocab-only GGUF at that
   llama.cpp tag, and :func:`runner_events` replays the per-token hold-back.
2. **Runner client** (``llm/llama_server.go``). Each non-empty event becomes
   ``Add(content, done=false)``, and the stop event becomes ``Add("", done=true)``.
   A run of more than 100 events with the same trimmed text aborts the request.
3. **Chat handler** (``server/routes.go``). It calls ``ParserForName`` and
   ``Init(tools, lastMessage, think)`` once, then ``Add`` per event. ``think``
   defaults to true when the parser supports thinking. A parser error ends the
   response with an error. ``ctcreplay`` runs this step with the pinned
   ``model/parsers`` package and serializes tool calls with
   ``openai.ToToolCalls``, as Ollama's OpenAI-compatible endpoint does.
4. **Non-streaming** (``writeChatResponse``). There is no separate
   non-streaming parse: with ``stream=false`` the same per-token events go
   through ``Add``, and content, thinking and tool calls are concatenated.
   :meth:`OllamaAdapter.parse` does exactly that. Tool calls are kept only
   when the request has tools. Streams are accumulated OpenAI-client style,
   merging tool calls by index.

Parser selection: see :data:`PARSER_RULES`. Models with no built-in parser at
the pin (Kimi, Llama, GLM-4.5/4.6, older DeepSeek formats, most Mistral
models) are served through a Go template with the legacy ``tools`` parser, or
through llama-server's own chat parser (the ``llamacpp`` engine). Neither path
is covered here, so those pairs are ``unsupported``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, ClassVar

from canitoolcall.adapters.base import Adapter, AdapterUnavailable, ReplayInput, Support, ToolSpec
from canitoolcall.chunking import TokensPerStep
from canitoolcall.fixtures import repo_root
from canitoolcall.results import ParsedToolCall, ParseResult, StreamAccumulator

BIN_DIR_ENV = "CANITOOLCALL_OLLAMA_BIN_DIR"
"""Directory holding ``ctcreplay`` and ``ctc-detok`` (default ``.engines/ollama/bin``)."""

GGUF_DIR_ENV = "CANITOOLCALL_GGUF_DIR"
"""Directory of shared vocab-only GGUFs (default ``.engines/gguf``, see ``scripts/engines/gguf_vocab.sh``)."""

TOKEN_REPEAT_LIMIT = 100
"""``llm/llama_server.go``: the request aborts when ``tokenRepeat > 100``."""

TOKEN_REPEAT_ERROR = "prediction aborted, token repeat limit reached"


@dataclass(frozen=True)
class ParserRule:
    """Selects Ollama's built-in parser for a family's reference model."""

    family: str
    pattern: str
    """Regex searched (case-insensitively) in the model repo id."""
    parser: str
    source: str
    """Why this parser: the Ollama library config or pinned source that assigns it."""
    parser_if_thinking_off: str | None = None
    """Parser to use when the fixture disables thinking (hybrid-thinking models)."""


_LIB = "Ollama library model config (registry.ollama.ai, fetched 2026-09-25)"

PARSER_RULES: tuple[ParserRule, ...] = (
    ParserRule(
        "qwen3-hermes",
        r"Qwen3-.*Thinking-2507",
        "qwen3-thinking",
        "model/parsers/parsers.go: qwen3-thinking = Qwen3Parser{thinking support, thinking by default}",
    ),
    ParserRule(
        "qwen3-hermes",
        r"Qwen3-.*Instruct-2507",
        "qwen3",
        "create/metadata.go parserNameForIdentifier: qwen3 architectures -> qwen3 (no thinking)",
    ),
    ParserRule(
        "qwen3-hermes",
        r"Qwen3-",
        "qwen3-thinking",
        "hybrid Qwen3: qwen3-thinking when thinking is on (the Qwen3 parser that handles <think>), "
        "qwen3 when the request disables thinking",
        parser_if_thinking_off="qwen3",
    ),
    ParserRule("qwen3-xml", r"Qwen3-Coder", "qwen3-coder", f"{_LIB}: qwen3-coder, qwen3-coder-next"),
    ParserRule("qwen3-xml", r"Qwen3\.[5-9]", "qwen3.5", f"{_LIB}: qwen3.5, qwen3.6; create/metadata.go qwen3_5"),
    ParserRule("gpt-oss", r"gpt-oss", "harmony", "server/routes.go shouldUseHarmony: gptoss family -> harmony"),
    ParserRule(
        "deepseek",
        r"DeepSeek-V3\.1|DeepSeek-V3\.2-Exp",
        "deepseek3",
        "create/metadata.go: deepseek architectures -> deepseek3, which parses the V3.1 tool format",
    ),
    ParserRule("glm", r"GLM-4\.7|GLM-5", "glm-4.7", f"{_LIB}: glm-4.7-flash; create/metadata.go glm4 -> glm-4.7"),
    ParserRule("gemma4", r"gemma-4", "gemma4", f"{_LIB}: gemma4"),
    ParserRule("mistral", r"Ministral-3|/Devstral-2-", "ministral", f"{_LIB}: ministral-3, devstral-2"),
    # Both names construct Nemotron3NanoParser (model/parsers/parsers.go). Nemotron-3.5-Bolt is Gemma 4 based.
    ParserRule(
        "nemotron",
        r"Nemotron-3-Ultra",
        "nemotron-3.5-nano",
        "create/metadata.go nemotronRendererParserNameFromTemplate: nemotron_h + '{reasoning effort: efficient}' "
        "in the template -> nemotron-3.5-nano",
    ),
    ParserRule(
        "nemotron",
        r"Nemotron-3(\.5)?-(Nano|Super|Lightning)",
        "nemotron-3-nano",
        "create/metadata.go nemotronRendererParserNameFromTemplate: nemotron_h architectures -> nemotron-3-nano",
    ),
)
"""First matching rule wins. Principle: the built-in parser Ollama ships for the
model's output format, as the Ollama library configures it where a published
model exists, else as ``create/metadata.go`` or ``ParserForName`` define it."""

UNSUPPORTED_REASON = (
    "no built-in Ollama parser for this model at the pin; Ollama serves it through a Go template "
    "with the legacy tools parser or through llama-server's native chat (see the llamacpp engine); "
    "legacy template parser not covered"
)


def select_rule(family: str, model: str) -> ParserRule | None:
    """The first :data:`PARSER_RULES` entry for ``family`` whose pattern matches ``model``."""
    for rule in PARSER_RULES:
        if rule.family == family and re.search(rule.pattern, model, re.IGNORECASE):
            return rule
    return None


def parser_name(rule: ParserRule, thinking: bool | None) -> str:
    if thinking is False and rule.parser_if_thinking_off:
        return rule.parser_if_thinking_off
    return rule.parser


def utf8_complete_len(data: bytes) -> int:
    """Port of llama-server's ``validate_utf8``: bytes up to the last complete UTF-8 sequence."""
    n = len(data)
    for i in range(1, min(4, n) + 1):
        c = data[n - i]
        if (c & 0xE0) == 0xC0 and i < 2:
            return n - i
        if (c & 0xF0) == 0xE0 and i < 3:
            return n - i
        if (c & 0xF8) == 0xF0 and i < 4:
            return n - i
    return n


def runner_events(pieces: Sequence[bytes]) -> list[str | None]:
    """Text each token's llama-server SSE event carries (``process_token``).

    ``None`` means no event was sent for that token: the generated text ended
    inside a UTF-8 sequence, so it was held back until a later token completed it.
    Stop strings are not emulated, because fixtures end before the stop token.
    """
    generated = b""
    sent = 0
    out: list[str | None] = []
    for piece in pieces:
        generated += piece
        if utf8_complete_len(generated) < len(generated):
            out.append(None)
            continue
        out.append(generated[sent:].decode("utf-8", errors="replace"))
        sent = len(generated)
    return out


def eog_stop_index(eog: Sequence[bool]) -> int | None:
    """Number of tokens llama-server processes before stopping at an end-of-generation token.

    ``server_context::process_token`` still handles the EOG token itself (its text
    is sent when it is preserved) and then stops, so later ids never reach Ollama.
    ``None`` when no id is an EOG token (fixtures normally end before the stop token).
    """
    for i, is_eog in enumerate(eog):
        if is_eog:
            return i + 1
    return None


def repeat_abort_index(events: Sequence[str | None]) -> int | None:
    """Index of the token whose event trips Ollama's token-repeat guard, if any.

    The runner client compares ``strings.TrimSpace`` of each event's content
    with the previous one and aborts once 100 repeats have been seen. The
    end-of-generation token's empty event and the final stop event are included.
    """
    last = ""
    repeat = 0
    trailing: list[str | None] = ["", ""]
    for i, text in enumerate([*events, *trailing]):
        if text is None:
            continue
        t = text.strip()
        if t == last:
            repeat += 1
        else:
            last, repeat = t, 0
        if repeat > TOKEN_REPEAT_LIMIT:
            return i
    return None


def _as_text(s: str) -> str | None:
    return s or None


def accumulate_nonstream(streams_events: Sequence[Mapping[str, Any]], has_tools: bool) -> ParseResult:
    """``writeChatResponse`` with ``stream=false`` plus ``openai.ToChatCompletion``."""
    content = "".join(e["content"] for e in streams_events)
    thinking = "".join(e["thinking"] for e in streams_events)
    calls: list[ParsedToolCall] = []
    if has_tools:
        for e in streams_events:
            calls.extend(ParsedToolCall(tc["function"]["name"], tc["function"]["arguments"]) for tc in e["tool_calls"])
    return ParseResult(content=_as_text(content), reasoning_content=_as_text(thinking), tool_calls=tuple(calls))


def accumulate_stream(streams_events: Sequence[Mapping[str, Any]]) -> ParseResult:
    """An OpenAI client reading Ollama's streamed chunks: deltas concatenated, tool calls merged by index."""
    acc = StreamAccumulator()
    for e in streams_events:
        acc.add_content(e["content"])
        acc.add_reasoning(e["thinking"])
        for tc in e["tool_calls"]:
            fn = tc["function"]
            acc.add_tool_call(int(tc.get("index", 0)), fn["name"], fn["arguments"])
    return acc.result()


class HarnessError(RuntimeError):
    """A compiled harness failed (a harness problem, not an engine outcome)."""


class _JsonLinesProcess:
    """A long-lived harness speaking one JSON object per line on stdin/stdout."""

    def __init__(self, argv: Sequence[str]) -> None:
        self._argv = list(argv)
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def request(self, msg: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._proc = subprocess.Popen(
                    self._argv,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=None,  # inherit: harness diagnostics go to the worker's stderr
                    text=True,
                    encoding="utf-8",
                    bufsize=1,
                )
            stdin: IO[str] | None = self._proc.stdin
            stdout: IO[str] | None = self._proc.stdout
            assert stdin is not None and stdout is not None
            try:
                stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
                stdin.flush()
                line = stdout.readline()
            except (BrokenPipeError, OSError) as e:
                raise HarnessError(f"{Path(self._argv[0]).name}: {e}") from e
            if not line:
                code = self._proc.wait()
                raise HarnessError(f"{Path(self._argv[0]).name} exited with code {code}")
        reply: dict[str, Any] = json.loads(line)
        if not reply.get("ok"):
            raise HarnessError(f"{Path(self._argv[0]).name}: {reply.get('error')}")
        return reply

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.stdin:
            proc.stdin.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        if proc.stdout:
            proc.stdout.close()


def _root() -> Path:
    root = repo_root()
    if root is None:
        raise AdapterUnavailable("the Ollama adapter runs from a source checkout (harnesses live in .engines/)")
    return root


def bin_dir() -> Path:
    env = os.environ.get(BIN_DIR_ENV)
    return Path(env) if env else _root() / ".engines" / "ollama" / "bin"


def gguf_dir() -> Path:
    env = os.environ.get(GGUF_DIR_ENV)
    return Path(env) if env else _root() / ".engines" / "gguf"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class _ParserInfo:
    preserved_tokens: tuple[str, ...]
    has_tool_support: bool
    has_thinking_support: bool


class OllamaAdapter(Adapter):
    name: ClassVar[str] = "ollama"
    pinned_version: ClassVar[str] = "7af393188defd52d370464de0d2064649cab9b41"
    tokens_per_step: ClassVar[TokensPerStep] = "one"
    """llama-server sends one SSE event per token and Ollama calls ``parser.Add`` once per event."""

    def __init__(self) -> None:
        self._replay: _JsonLinesProcess | None = None
        self._detok: _JsonLinesProcess | None = None
        self._hello: dict[str, Any] | None = None
        self._detok_hello: dict[str, Any] | None = None
        self._parsers: dict[str, _ParserInfo] = {}
        self._gguf_sha: dict[Path, str] = {}
        self._events: dict[tuple[str, str, tuple[int, ...]], tuple[list[str | None], int | None]] = {}

    # ------------------------------------------------------------------ harnesses

    def _harness(self, which: str) -> _JsonLinesProcess:
        path = bin_dir() / which
        if not path.is_file():
            raise AdapterUnavailable(f"{path} not built; run scripts/engines/ollama.sh")
        return _JsonLinesProcess([str(path)])

    def _replayer(self) -> _JsonLinesProcess:
        if self._replay is None:
            self._replay = self._harness("ctcreplay")
        return self._replay

    def _detokenizer(self) -> _JsonLinesProcess:
        if self._detok is None:
            self._detok = self._harness("ctc-detok")
        return self._detok

    def _info(self) -> dict[str, Any]:
        if self._hello is None:
            self._hello = self._replayer().request({"op": "hello"})
            self._detok_hello = self._detokenizer().request({"op": "hello"})
        return self._hello

    def _parser_info(self, parser: str) -> _ParserInfo:
        if parser not in self._parsers:
            d = self._replayer().request({"op": "describe", "parser": parser})
            if not d.get("known"):
                raise HarnessError(f"Ollama at the pin has no parser named {parser!r}")
            self._parsers[parser] = _ParserInfo(
                tuple(d["preserved_tokens"]), bool(d["has_tool_support"]), bool(d["has_thinking_support"])
            )
        return self._parsers[parser]

    # ------------------------------------------------------------------ Adapter API

    def version(self) -> str:
        commit = str(self._info()["ollama_commit"])
        if commit != self.pinned_version:
            raise AdapterUnavailable(f"ctcreplay was built from {commit}, expected {self.pinned_version}")
        return commit[:8]

    def commit(self) -> str | None:
        return str(self._info()["ollama_commit"])

    def engine_details(self) -> dict[str, Any]:
        info = self._info()
        detok = self._detok_hello or {}
        return {
            "go_version": info.get("go_version"),
            "llama_cpp_build": f"b{detok.get('llama_cpp_build')}",
            "llama_cpp_commit": detok.get("llama_cpp_commit"),
            "runner": "llama-server token rendering (preserved_tokens) + model/parsers via server/routes.go flow",
        }

    def supports(self, family: str, model: str) -> Support:
        if select_rule(family, model) is None:
            return Support(False, UNSUPPORTED_REASON)
        return Support(True)

    def _rule(self, raw: ReplayInput) -> ParserRule:
        rule = select_rule(raw.family, raw.model)
        if rule is None:
            raise HarnessError(f"unsupported pair {raw.family}/{raw.model}")
        return rule

    def _gguf(self, raw: ReplayInput) -> tuple[Path, dict[str, Any]]:
        repo = raw.tokenizer.repo if raw.tokenizer else raw.model
        stem = repo.replace("/", "--")
        path = gguf_dir() / f"{stem}.vocab.gguf"
        if not path.is_file():
            raise HarnessError(f"missing vocab GGUF {path}; run scripts/engines/gguf_vocab.sh {repo}@<revision>")
        meta_path = path.with_name(f"{stem}.vocab.json")
        meta: dict[str, Any] = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        return path, meta

    def parser_config(self, raw: ReplayInput) -> dict[str, Any]:
        rule = self._rule(raw)
        parser = parser_name(rule, raw.thinking)
        info = self._parser_info(parser)
        gguf, meta = self._gguf(raw)
        if gguf not in self._gguf_sha:
            self._gguf_sha[gguf] = _sha256(gguf)
        tok_rev = raw.tokenizer.revision if raw.tokenizer else None
        think_effective: bool | None = raw.thinking
        if think_effective is None and info.has_thinking_support:
            think_effective = True
        detok = self._detok_hello or {}
        return {
            "parser": parser,
            "parser_source": rule.source,
            "ollama_commit": self.pinned_version,
            "think_requested": raw.thinking,
            "think_effective": think_effective,
            "preserved_tokens": list(info.preserved_tokens),
            "has_thinking_support": info.has_thinking_support,
            "detokenizer": "llama-server common_token_to_piece(special = id in preserved_tokens)",
            "llama_cpp_build": f"b{detok.get('llama_cpp_build')}" if detok else None,
            "vocab_gguf": gguf.name,
            "vocab_gguf_sha256": self._gguf_sha[gguf],
            "vocab_gguf_revision": meta.get("revision"),
            "vocab_gguf_revision_matches_tokenizer": (meta.get("revision") == tok_rev) if tok_rev else None,
            "vocab_gguf_converter_llama_cpp_commit": meta.get("llama_cpp_commit"),
            "chunking": "one llama-server event per token; Add(delta, false) per non-empty group, then Add('', true)",
            "eog_stop": "tokens after the first end-of-generation token are dropped (llama-server stops there)",
            "nonstream": "per-token events concatenated (routes.go writeChatResponse)",
            "tool_call_serialization": "openai.ToToolCalls",
            "legacy_template_parser": False,
        }

    def units(self, raw: ReplayInput) -> list[int]:
        if raw.token_ids is not None:
            return list(raw.token_ids)
        gguf, _ = self._gguf(raw)
        reply = self._detokenizer().request({"op": "tokenize", "gguf": str(gguf), "text": raw.text})
        return [int(i) for i in reply["ids"]]

    def special_token_ids(self, raw: ReplayInput) -> Collection[int] | None:
        """The fixture's ids whose type in the vocab GGUF is CONTROL (what llama-server hides unless preserved)."""
        gguf, _ = self._gguf(raw)
        reply = self._detokenizer().request({"op": "control", "gguf": str(gguf), "ids": sorted(set(self.units(raw)))})
        return frozenset(int(i) for i in reply["control"])

    def _token_events(self, raw: ReplayInput, parser: str, ids: Sequence[int]) -> tuple[list[str | None], int | None]:
        """Per-token llama-server events, and how many tokens are generated before an EOG stop."""
        key = (raw.fixture_id, parser, tuple(ids))
        if key not in self._events:
            gguf, _ = self._gguf(raw)
            info = self._parser_info(parser)
            reply = self._detokenizer().request(
                {"op": "pieces", "gguf": str(gguf), "preserved": list(info.preserved_tokens), "ids": list(ids)}
            )
            self._events = {
                key: (runner_events([bytes(p) for p in reply["pieces"]]), eog_stop_index(reply.get("eog", ())))
            }
        return self._events[key]

    def _run(
        self, raw: ReplayInput, tools: Sequence[ToolSpec], groups: Sequence[Sequence[int]], *, stream: bool
    ) -> ParseResult:
        parser = parser_name(self._rule(raw), raw.thinking)
        ids = [i for g in groups for i in g]
        events, eog_stop = self._token_events(raw, parser, ids)
        if eog_stop is not None:
            events = events[:eog_stop]
        abort = repeat_abort_index(events)
        # Tokens past `cut` are never produced: the request aborted (repeat guard) or
        # llama-server stopped at an end-of-generation token.
        cut = abort if abort is not None else eog_stop
        deltas: list[str] = []
        pos = 0
        for g in groups:
            end = pos + len(g)
            if cut is not None and end > cut:
                end = cut
            text = "".join(e for e in events[pos:end] if e)
            if text:
                deltas.append(text)
            pos += len(g)
            if cut is not None and pos >= cut:
                break
        # llama-server's stop event reaches the parser as Add("", done=true).
        # An aborted request returns its error before that final callback.
        finished = abort is None
        if finished:
            deltas.append("")
        reply = self._replayer().request(
            {
                "op": "replay",
                "parser": parser,
                "tools": list(tools),
                "think": raw.thinking,
                "streams": [deltas],
                "finish": finished,
            }
        )
        s = reply["streams"][0]
        events_out: list[Mapping[str, Any]] = s["events"]
        result = accumulate_stream(events_out) if stream else accumulate_nonstream(events_out, bool(tools))
        error = s.get("error") or (None if finished else TOKEN_REPEAT_ERROR)
        if error:
            if not stream:
                # A failed non-streaming request returns only the error (writeChatResponse).
                return ParseResult(exception=f"OllamaError: {error}")
            return ParseResult(
                content=result.content,
                reasoning_content=result.reasoning_content,
                tool_calls=result.tool_calls,
                exception=f"OllamaError: {error}",
            )
        return result

    def parse(self, raw: ReplayInput, tools: Sequence[ToolSpec]) -> ParseResult:
        ids = self.units(raw)
        return self._run(raw, tools, [[i] for i in ids], stream=False)

    def parse_stream(self, raw: ReplayInput, chunks: Sequence[Sequence[int]], tools: Sequence[ToolSpec]) -> ParseResult:
        return self._run(raw, tools, chunks, stream=True)

    def close(self) -> None:
        for proc in (self._replay, self._detok):
            if proc is not None:
                proc.close()
        self._replay = self._detok = None

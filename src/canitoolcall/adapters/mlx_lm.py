"""mlx-lm adapter: ``mlx_lm.server``'s own chat-completions loop, with no model.

Runs in ``.venvs/mlx_lm`` (mlx-lm==0.32.2, mlx==0.32.4), built by
``scripts/engines/mlx_lm.sh``. ``mlx_lm.server`` imports ``mlx.core`` at module
level, so the env exists only where mlx imports (macOS arm64 with Metal; Linux
through the ``mlx-cpu`` wheels). Elsewhere :meth:`MlxLmAdapter.version` raises
:class:`AdapterUnavailable`.

What runs, all from mlx-lm 0.32.2 (``ml-explore/mlx-lm@93e6d4a6``, tag v0.32.2):

* **Tokenizer and parser selection:** ``mlx_lm.utils.load_tokenizer`` on a local
  snapshot of the tokenizer/config files, as ``mlx_lm.server`` loads a model.
  It picks the tool parser from ``tool_parser_type`` (tokenizer_config.json or
  the model module's ``TOKENIZER_DEFAULTS``), else ``_infer_tool_parser`` reads
  the chat template and vocab. Thinking markers come from ``_infer_thinking``.
  mlx-lm has no reasoning-parser names: reasoning is the text between those
  markers.
* **Request handling:** the real ``APIHandler.do_POST`` for
  ``/v1/chat/completions``, so ``handle_completion`` runs unmodified: the
  ``TextStateMachine`` splits each step's text into normal/reasoning/tool
  segments, and ``ToolCallFormatter`` calls ``parse_tool_call`` once per
  complete tool block. The prompt (and so the initial ``reasoning``/``normal``
  state) comes from the real ``ResponseGenerator._tokenize``. Only the HTTP
  header writes are no-ops, and the body is read from ``wfile``.
* **Generation steps:** the one part replayed in place of a model. Each id group
  becomes one ``Response``, built the way ``ResponseGenerator._generate``
  (the batched path the server uses without a draft model) builds it:
  ``detokenizer.add_token`` per id, then ``last_segment``; at a ``stop`` finish
  the stop token is never decoded (``finalize`` only); at ``length`` the last
  token is added and then ``finalize``\\ d. The detokenizer is the tokenizer's own
  ``StreamingDetokenizer`` (``tokenizer.detokenizer``).

Coverage (:data:`FAMILY_PARSERS`) is what mlx-lm selects for the original Hub
repos that the fixtures pin. Users often serve ``mlx-community`` conversions of
those repos; their tokenizer files were not checked.

Non-streaming responses run the same per-token loop with ``stream: false``, as
the server does. The server sends one ``Response`` per generated token
(:attr:`MlxLmAdapter.tokens_per_step` ``"one"``), so multi-token chunkings are
synthetic for this engine.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

from canitoolcall.adapters.base import (
    PLACEHOLDER_USER_MESSAGE,
    REMOTE_CODE_PATTERNS,
    Adapter,
    AdapterUnavailable,
    ReplayInput,
    Support,
    ToolSpec,
    missing_relative_imports,
    trusts_remote_code,
)
from canitoolcall.chunking import TokensPerStep
from canitoolcall.results import ParsedToolCall, ParseResult, StreamAccumulator

TAG_COMMIT = "93e6d4a6447b3080861985947f0820e76fcb957d"
"""``ml-explore/mlx-lm`` commit of tag ``v0.32.2``; the PyPI wheel's sources match it."""

CHAT_PATH = "/v1/chat/completions"

FAMILY_PARSERS: dict[str, dict[str, str]] = {
    "qwen3-hermes": {
        "Qwen/Qwen3-0.6B": "json_tools",
        "Qwen/Qwen3-4B-Thinking-2507": "json_tools",
        "Qwen/Qwen3-4B-Instruct-2507": "json_tools",
    },
    "qwen3-xml": {
        "Qwen/Qwen3.8-27B": "qwen3_coder",
        "Qwen/Qwen3.6-35B-A3B": "qwen3_coder",
        "Qwen/Qwen3.5-9B": "qwen3_coder",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct": "qwen3_coder",
        "Qwen/Qwen3-Coder-480B-A35B-Instruct": "qwen3_coder",
        "Qwen/Qwen3-Coder-Next": "qwen3_coder",
    },
    "glm": {"zai-org/GLM-4.5": "glm47", "zai-org/GLM-4.7": "glm47", "zai-org/GLM-5.3": "glm47"},
    "gemma4": {"google/gemma-4-31B-it": "gemma4"},
    "kimi": {
        "moonshotai/Kimi-K2.6": "kimi_k2",
        "moonshotai/Kimi-K2-Instruct-0905": "kimi_k2",
        "moonshotai/Kimi-K3": "kimi_k3",
    },
    "mistral": {
        "mistralai/Mistral-7B-Instruct-v0.3": "mistral",
        "mistralai/Ministral-3-14B-Instruct-2512": "mistral",
        "mistralai/Ministral-3-14B-Reasoning-2512": "mistral",
        "mistralai/Mistral-Small-4-119B-2603": "mistral",
        "mistralai/Mistral-Medium-3.5-128B": "mistral",
        "mistralai/Devstral-Small-2-24B-Instruct-2512": "mistral",
    },
}
"""Family -> reference model -> the parser mlx-lm 0.32.2 selects for it.

Found on 2026-11-19 by running ``mlx_lm.utils.load_tokenizer`` on each
``family.json`` reference model at its pinned revision (Llama: the fixtures'
unsloth mirrors). The adapter re-checks the selection at load time and fails
as a harness error if it differs."""

NO_TEMPLATE: frozenset[str] = frozenset(
    {
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "mistralai/Magistral-Small-2509",
        "mistralai/Magistral-Small-2506",
        "mistralai/Devstral-Small-2507",
    }
)
"""Mistral reference repos with no HF chat template: mlx-lm infers no parser for them."""

TOKENIZER_FILES: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "chat_template.json",
    "*.jinja",
)
"""Snapshot files: what ``load_tokenizer`` reads (a subset of scripts/engines/prefetch_hf.py's list). Never weights."""


def _engine(module: str) -> Any:
    """Import an engine module lazily (adapter modules stay stdlib-only at import time)."""
    return importlib.import_module(module)


def exception_text(exc: BaseException) -> str:
    """``"Type: message"`` as recorded in :attr:`ParseResult.exception`."""
    return f"{type(exc).__name__}: {exc}"


def message_result(message: Mapping[str, Any]) -> ParseResult:
    """A non-streaming ``choices[0].message`` (mlx-lm names the reasoning field ``reasoning``)."""
    content, reasoning = message.get("content"), message.get("reasoning")
    return ParseResult(
        content=content if isinstance(content, str) and content else None,
        reasoning_content=reasoning if isinstance(reasoning, str) and reasoning else None,
        tool_calls=tuple(
            ParsedToolCall(c["function"]["name"], c["function"]["arguments"]) for c in message.get("tool_calls") or ()
        ),
    )


def sse_result(body: bytes, exception: str | None = None) -> ParseResult:
    """Accumulate a server-sent-events body (``data: {...}`` lines) OpenAI-client style."""
    acc = StreamAccumulator()
    for line in body.decode("utf-8").splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        for choice in json.loads(line[len("data: ") :]).get("choices") or ():
            acc.add_openai_delta(choice.get("delta") or {})
    return acc.result(exception=exception)


def snapshot(repo: str, revision: str) -> Path:
    """Local snapshot of :data:`TOKENIZER_FILES` (plus reviewed remote code) at ``revision``."""
    snapshot_download = _engine("huggingface_hub").snapshot_download
    remote = trusts_remote_code(repo, revision)
    allow = list(TOKENIZER_FILES) + (list(REMOTE_CODE_PATTERNS) if remote else [])
    path = Path(snapshot_download(repo, revision=revision, allow_patterns=allow))
    requested: set[str] = set()
    while remote and (todo := [f for f in missing_relative_imports(path) if f not in requested]):
        requested.update(todo)
        path = Path(snapshot_download(repo, revision=revision, allow_patterns=todo))
    return path


@dataclass(frozen=True)
class _Loaded:
    repo: str
    revision: str
    tok: Any
    """``mlx_lm.tokenizer_utils.TokenizerWrapper``."""
    parser: str
    trust_remote_code: bool
    chat_template_sha256: str | None


class _Replay:
    """Stands in for ``ResponseGenerator``: the real ``_tokenize`` and state machine, replayed tokens."""

    def __init__(self, server: Any, tok: Any, steps: Sequence[Sequence[int]], by_length: bool, max_tokens: int) -> None:
        self.server, self.tok, self.steps, self.by_length = server, tok, steps, by_length
        # Request defaults read by APIHandler.do_POST; none of them reaches the parsers.
        self.cli_args = SimpleNamespace(
            num_draft_tokens=0,
            max_tokens=max_tokens,
            temp=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            allowed_origins=["*"],
            chat_template_args={},
        )
        self.rg = object.__new__(server.ResponseGenerator)
        self.rg.model_provider = SimpleNamespace(
            processor=None, cli_args=self.cli_args, model_key=("replay", None, None)
        )
        self.rg._state_machine_cache = {}
        self.error: BaseException | None = None
        self.prompt: list[int] = []
        self.initial_state: str | None = None

    def generate(self, request: Any, args: Any, progress_callback: Any = None) -> tuple[Any, Iterator[Any]]:
        try:
            prompt, _segments, _types, initial_state, _vision = self.rg._tokenize(self.tok, request, args)
            _stops, text_sm = self.rg._make_state_machine(self.rg.model_provider.model_key, self.tok, args.stop_words)
        except Exception as e:  # handle_completion would turn this into a 404 body
            self.error = e
            raise
        self.prompt, self.initial_state = list(prompt), initial_state
        ctx = self.server.GenerationContext(
            has_tool_calling=self.tok.has_tool_calling,
            has_thinking=self.tok.has_thinking,
            tool_parser=self.tok.tool_parser,
            text_sm=text_sm,
            initial_state=initial_state,
            prompt=self.prompt,
            prompt_cache_count=0,
        )
        return ctx, self._responses()

    def _responses(self) -> Iterator[Any]:
        """``Response`` objects as ``ResponseGenerator._generate`` (batched path) makes them."""
        Response = self.server.Response
        detok = self.tok.detokenizer
        for i, group in enumerate(self.steps):
            for t in group:
                detok.add_token(int(t))
            if self.by_length and i == len(self.steps) - 1:
                detok.finalize()
                yield Response(detok.last_segment, int(group[-1]), 0.0, "length", ())
                return
            yield Response(detok.last_segment, int(group[-1]), 0.0, None, ())
        # The stop token itself is never decoded: finalize, then emit what was held back.
        detok.finalize()
        yield Response(detok.last_segment, min(self.tok.eos_token_ids, default=0), 0.0, "stop", ())


class MlxLmAdapter(Adapter):
    name: ClassVar[str] = "mlx_lm"
    pinned_version: ClassVar[str] = "0.32.2"
    tokens_per_step: ClassVar[TokensPerStep] = "one"
    """``mlx_lm.server`` puts one generated token in each ``Response`` (both generation paths)."""

    def __init__(self) -> None:
        self._loaded: dict[tuple[str, str], _Loaded] = {}
        self._handler: Any = None
        self._initial_state: dict[str, str | None] = {}

    # -- engine identity -------------------------------------------------

    def version(self) -> str:
        try:
            mlx_lm = _engine("mlx_lm")
            _engine("mlx.core")
        except Exception as e:  # ImportError, or a missing backend library
            raise AdapterUnavailable(
                f"mlx-lm/mlx cannot be imported here ({type(e).__name__}: {e}); run scripts/engines/mlx_lm.sh "
                "(macOS arm64, or Linux with the mlx-cpu wheels)"
            ) from e
        return str(mlx_lm.__version__)

    def commit(self) -> str | None:
        return TAG_COMMIT if self.version() == self.pinned_version else None

    def engine_details(self) -> dict[str, Any]:
        from importlib import metadata

        details: dict[str, Any] = {}
        for dist in ("mlx", "mlx-metal", "mlx-cpu", "transformers", "tokenizers", "huggingface-hub", "tiktoken"):
            try:
                details[dist] = metadata.version(dist)
            except metadata.PackageNotFoundError:
                details[dist] = None
        details["mlx_default_device"] = str(_engine("mlx.core").default_device())
        return details

    # -- coverage ----------------------------------------------------------

    def supports(self, family: str, model: str) -> Support:
        models = FAMILY_PARSERS.get(family)
        if models and model in models:
            return Support(True)
        if model in NO_TEMPLATE:
            return Support(
                False,
                f"mlx-lm 0.32.2 infers no tool parser for {model}: the repo ships no HF chat template "
                "(mlx_lm.tokenizer_utils._infer_tool_parser), so mlx_lm.server returns calls as content",
            )
        if models is None:
            return Support(
                False,
                f"mlx-lm 0.32.2 has no tool parser for the {family} format "
                "(mlx_lm/tool_parsers/; none is selected for its reference models)",
            )
        return Support(
            False,
            f"{model} is not a reference model whose mlx-lm parser was verified "
            f"(verified: {', '.join(sorted(models))})",
        )

    # -- loading -------------------------------------------------------------

    def _load(self, raw: ReplayInput) -> _Loaded:
        if raw.tokenizer is not None:
            repo, revision = raw.tokenizer.repo, raw.tokenizer.revision
        elif raw.revision:
            repo, revision = raw.model, raw.revision
        else:
            raise ValueError(f"no pinned revision for {raw.model!r} and the fixture has no tokenizer pin")
        key = (repo, revision)
        if key in self._loaded:
            return self._loaded[key]
        utils = _engine("mlx_lm.utils")
        path = snapshot(repo, revision)
        config: dict[str, Any] = utils.load_config(path) if (path / "config.json").exists() else {}
        trust = trusts_remote_code(repo, revision)
        # mlx_lm.server: load(..., tokenizer_config={"trust_remote_code": <--trust-remote-code>})
        tok = utils.load_tokenizer(path, {"trust_remote_code": trust}, eos_token_ids=config.get("eos_token_id"))
        parser = tok.init_kwargs.get("tool_parser_type")
        want = FAMILY_PARSERS.get(raw.family, {}).get(raw.model)
        if parser != want:
            # Harness problem: the static coverage table is stale for this revision.
            raise RuntimeError(f"mlx-lm selected tool parser {parser!r} for {repo}@{revision}, table says {want!r}")
        chat_template = tok.chat_template
        loaded = _Loaded(
            repo=repo,
            revision=revision,
            tok=tok,
            parser=parser,
            trust_remote_code=trust,
            chat_template_sha256=(
                hashlib.sha256(chat_template.encode("utf-8")).hexdigest() if isinstance(chat_template, str) else None
            ),
        )
        self._loaded[key] = loaded
        return loaded

    def _handler_class(self) -> Any:
        """``APIHandler`` with the HTTP header writes stubbed out (they never touch the body)."""
        if self._handler is None:
            server = _engine("mlx_lm.server")

            class Handler(server.APIHandler):  # type: ignore[name-defined,misc]
                status: int | None = None

                def send_response(self, code: int, message: str | None = None) -> None:
                    self.status = code

                def send_header(self, keyword: str, value: str) -> None:
                    pass

                def end_headers(self) -> None:
                    pass

            self._handler = Handler
        return self._handler

    def _body(self, raw: ReplayInput, tools: Sequence[ToolSpec], stream: bool, max_tokens: int) -> dict[str, Any]:
        body: dict[str, Any] = {
            "messages": [{"role": "user", "content": PLACEHOLDER_USER_MESSAGE}],
            "stream": stream,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = [dict(t) for t in tools]
        if raw.thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": raw.thinking}
        return body

    def _post(
        self, raw: ReplayInput, tools: Sequence[ToolSpec], steps: Sequence[Sequence[int]], stream: bool
    ) -> tuple[bytes, BaseException | None]:
        """One ``POST /v1/chat/completions`` through the real handler. Returns the body and any engine exception."""
        st = self._load(raw)
        server = _engine("mlx_lm.server")
        max_tokens = max(1, sum(len(g) for g in steps) + 1)
        replay = _Replay(server, st.tok, steps, raw.finished_by_length, max_tokens)
        data = json.dumps(self._body(raw, tools, stream, max_tokens)).encode("utf-8")
        Handler = self._handler_class()
        h = Handler.__new__(Handler)
        h.created, h.response_generator, h.system_fingerprint = 0, replay, "canitoolcall-replay"
        h.path, h.headers = CHAT_PATH, {"Content-Length": str(len(data))}
        h.rfile, h.wfile = io.BytesIO(data), io.BytesIO()
        engine_error: BaseException | None = None
        try:
            h.do_POST()
        except Exception as e:  # raised mid-response: the client keeps what was already sent
            engine_error = e
        if replay.error is not None:
            raise RuntimeError(f"mlx_lm.server could not build the prompt: {exception_text(replay.error)}")
        if h.status != 200:
            raise RuntimeError(f"mlx_lm.server answered {h.status}: {h.wfile.getvalue()[:500]!r}")
        self._initial_state[raw.fixture_id] = replay.initial_state
        if raw.generation_prompt:
            prompt_text = st.tok.decode(replay.prompt)
            if not prompt_text.endswith(raw.generation_prompt):
                raise RuntimeError(
                    f"rendered prompt does not end with the fixture's generation prompt {raw.generation_prompt!r}: "
                    f"...{prompt_text[-80:]!r}"
                )
        return h.wfile.getvalue(), engine_error

    # -- contract ------------------------------------------------------------

    def parser_config(self, raw: ReplayInput) -> dict[str, Any]:
        st = self._load(raw)
        tok = st.tok
        return {
            "tool_parser": st.parser,
            "tool_parser_module": f"mlx_lm.tool_parsers.{st.parser}",
            "tool_call_markers": [tok.tool_call_start, tok.tool_call_end],
            "reasoning_parser": None,
            "think": [tok.think_start, tok.think_end] if tok.has_thinking else None,
            "parser_selection": "mlx_lm.utils.load_tokenizer (tool_parser_type, else _infer_tool_parser)",
            "initial_state": self._initial_state.get(raw.fixture_id),
            "chat_template_sha256": st.chat_template_sha256,
            "tokenizer": f"{st.repo}@{st.revision}",
            "tokenizer_mode": "hf",
            "trust_remote_code": st.trust_remote_code,
            "detokenizer": type(tok._detokenizer).__name__,
            "entrypoint": f"mlx_lm.server.APIHandler.do_POST {CHAT_PATH} (handle_completion)",
            "generation_steps": "ResponseGenerator._generate batched path: add_token + last_segment per step",
            "prompt": {
                "user_message": PLACEHOLDER_USER_MESSAGE,
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": raw.thinking} if raw.thinking is not None else None,
            },
            "finish_reason": "length" if raw.finished_by_length else "stop",
            "stop_token_appended": None,
        }

    def units(self, raw: ReplayInput) -> list[int]:
        if raw.token_ids is not None:
            return list(raw.token_ids)
        return list(self._load(raw).tok.encode(raw.text, add_special_tokens=False))

    def special_token_ids(self, raw: ReplayInput) -> Collection[int] | None:
        tok = self._load(raw).tok
        ids = {int(i) for i in tok.all_special_ids}
        ids.update(int(i) for i, t in tok.added_tokens_decoder.items() if getattr(t, "special", False))
        return frozenset(ids)

    def parse(self, raw: ReplayInput, tools: Sequence[ToolSpec]) -> ParseResult:
        """``stream: false``: the server still runs its loop one token at a time."""
        body, error = self._post(raw, tools, [[i] for i in self.units(raw)], stream=False)
        if error is not None:
            return ParseResult(exception=exception_text(error))
        return message_result(json.loads(body)["choices"][0]["message"])

    def parse_stream(self, raw: ReplayInput, chunks: Sequence[Sequence[int]], tools: Sequence[ToolSpec]) -> ParseResult:
        body, error = self._post(raw, tools, chunks, stream=True)
        return sse_result(body, exception_text(error) if error is not None else None)

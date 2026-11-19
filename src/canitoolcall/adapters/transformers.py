"""HF transformers adapter: ``tokenizer.parse_response`` and the streaming
``ResponseParser``, driven by the ``response_template`` that a model repo ships
in its ``tokenizer_config.json``.

Owner: transformers adapter builder (see docs/DESIGN.md). Runs in
``.venvs/transformers`` (transformers==5.23.0, jinja2, jmespath; no torch),
built by ``scripts/engines/transformers.sh``.

Entry points (transformers 5.23.0), called exactly as below:

* non-streaming: ``tok.parse_response(output_ids, template, prefix=prompt_text, tools=tools)``
  (``output_ids`` are decoded inside transformers, as ``transformers serve`` does)
* streaming: ``ResponseParser(template, prefix=prompt_text, tools=tools)``;
  ``initial_events``, then ``.feed(delta)`` once per id group, then ``.finalize()``.
  Text deltas come from the Rust ``tokenizers.decoders.DecodeStream`` with
  ``skip_special_tokens=False``, stepped one id at a time. That is the
  detokenizer ``transformers serve`` uses when a response parser is active,
  and it never splits a multi-byte character.
* Parser events become OpenAI deltas through transformers' own
  ``transformers.cli.serving.utils.response_events_to_chunks``. The
  non-streaming reasoning is the first field of transformers'
  ``_REASONING_FIELDS`` (``reasoning_content``, ``thinking``, ``reasoning``)
  that is set, as ``serve``'s ``parse_response_with_tools`` reads it. Tool-call
  arguments are serialized with its ``_normalize_tool_call``
  (``json.dumps``, so non-ASCII is ``\\u``-escaped; it is still valid JSON).
* The legacy ``response_schema`` key is dropped by transformers 5.x and is
  never used here.

Coverage rule: a ``(family, model)`` pair is supported only when the model's
Hub repo itself ships a ``response_template`` at the revision pinned in
:data:`PINNED_REPOS`. The template is read from that revision; the tokenizer
and chat template come from the fixture's tokenizer pin, which produced the
token ids. When the two revisions differ, ``tokenizer.json`` must be
byte-identical and ``tokenizer_config.json`` identical apart from
``chat_template``/``response_template``, or the replay is a harness error. At
load time the adapter re-checks that the downloaded ``tokenizer_config.json``
contains ``response_template``. Response templates
copied from transformers' tests, and the ``transformers serve`` fallbacks
keyed on ``model_type``, are not what the Hub repo ships, so they are never
used for results.

Differences from ``transformers serve`` (recorded in ``parser_config``): the
prompt is rendered from
a fixed placeholder user message, because fixtures do not record the
conversation. Only the part after the template's ``start_anchor`` reaches
the parser.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

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

GEMMA4_REPOS: dict[str, str] = {
    "google/gemma-4-E2B-it": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    "google/gemma-4-E4B-it": "ee0ef6023621cff504d758262d4e04895a5af4a2",
    "google/gemma-4-12B-it": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    "google/gemma-4-26B-A4B-it": "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
    "google/gemma-4-31B-it": "842da3794eaa0b77d5f08bae87a17459d91ff475",
}
"""Verified 2026-09-25. All five ship the same template."""

QWEN3_XML_REPOS: dict[str, str] = {
    # "Add a response template and generation tags (#236)", 2026-11-07. The fixtures' pin,
    # 1d4bf0f2, has no response_template; tokenizer.json is unchanged between the two.
    "Qwen/Qwen3.8-27B": "37561816e66b5b9153b1b64c10968c18c1e6f4a0",
}
"""Verified 2026-11-19. Qwen/Qwen3.6-35B-A3B@345d467b ships the same template
(its #81), but no fixture leads with that repo, so it is not listed."""

PINNED_REPOS: dict[str, str] = {**GEMMA4_REPOS, **QWEN3_XML_REPOS}
"""Hub repos verified to ship ``response_template`` in ``tokenizer_config.json``,
with the revision the check was made at; the template is read from it."""

FAMILY_REPOS: dict[str, frozenset[str]] = {
    "gemma4": frozenset(GEMMA4_REPOS),
    "qwen3-xml": frozenset(QWEN3_XML_REPOS),
}
"""Family slug -> repos whose Hub-shipped ``response_template`` is the parser.

In transformers the tool parser and the reasoning parser are both the one
``response_template`` (fields ``tool_calls`` and ``thinking`` or
``reasoning_content``). On 2026-11-19 every other repo the fixtures list was
checked at its fixture pin and at its Hub ``main``: none ships one (the
Qwen3, Qwen3.5, Qwen3-Coder, DeepSeek, GLM, gpt-oss, Kimi, MiniMax, Mistral,
Seed-OSS and unsloth Llama repos). The meta-llama repos are gated and were not
checked."""

PARSER_NAME = "response_template"
"""transformers has no named parsers: the repo's ``response_template`` is the parser."""

TEMPLATE_FILE = "tokenizer_config.json"

SERVE_UTILS = "transformers.cli.serving.utils"
"""``transformers serve`` helpers reused for event -> delta translation and
tool-call serialization (importable without torch in 5.23.0)."""

TEMPLATE_KEYS = frozenset({"chat_template", "response_template"})
"""``tokenizer_config.json`` keys that may differ between the template revision and the tokenizer pin."""


def _engine(module: str) -> Any:
    """Import an engine module lazily (adapter modules stay stdlib-only at import time)."""
    return importlib.import_module(module)


def response_template_sha256(template: Mapping[str, Any]) -> str:
    """sha256 of the template as canonical JSON (sorted keys, compact, UTF-8)."""
    blob = json.dumps(template, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def same_tokenizer(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Two ``tokenizer_config.json`` dicts are equal apart from :data:`TEMPLATE_KEYS`."""
    return {k: v for k, v in a.items() if k not in TEMPLATE_KEYS} == {
        k: v for k, v in b.items() if k not in TEMPLATE_KEYS
    }


def resolve_tokenizer(raw: ReplayInput) -> tuple[str, str]:
    """``(repo, revision)`` whose tokenizer and ``response_template`` are used.

    The fixture's tokenizer pin wins, because it produced ``output_token_ids``.
    Otherwise the reference model is used at the family's pinned revision,
    falling back to :data:`PINNED_REPOS`.
    """
    if raw.tokenizer is not None:
        return raw.tokenizer.repo, raw.tokenizer.revision
    if raw.revision:
        return raw.model, raw.revision
    try:
        return raw.model, PINNED_REPOS[raw.model]
    except KeyError:
        raise ValueError(f"no pinned revision for {raw.model!r} and the fixture has no tokenizer pin") from None


def exception_text(exc: BaseException) -> str:
    """``"Type: message"`` as recorded in :attr:`ParseResult.exception`."""
    return f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class _Loaded:
    """One tokenizer at one revision, with the verified Hub-shipped template of its repo."""

    repo: str
    revision: str
    template_revision: str
    tok: Any
    template: dict[str, Any]
    template_sha256: str
    chat_template_sha256: str | None


class TransformersAdapter(Adapter):
    name: ClassVar[str] = "transformers"
    pinned_version: ClassVar[str] = "5.23.0"
    supports_text_deltas: ClassVar[bool] = True
    tokens_per_step: ClassVar[TokensPerStep] = "one"
    """``transformers serve`` streams from ``generate`` one sampled token at a time."""

    def __init__(self) -> None:
        self._loaded: dict[tuple[str, str], _Loaded] = {}

    # -- engine identity -------------------------------------------------

    def version(self) -> str:
        try:
            transformers = _engine("transformers")
        except ImportError as e:
            raise AdapterUnavailable(
                f"transformers is not installed here ({e}); run scripts/engines/transformers.sh"
            ) from e
        return str(transformers.__version__)

    def engine_details(self) -> dict[str, Any]:
        from importlib import metadata
        from importlib.util import find_spec

        details: dict[str, Any] = {}
        for dist in ("tokenizers", "huggingface_hub", "jinja2", "jmespath"):
            try:
                details[dist] = metadata.version(dist)
            except metadata.PackageNotFoundError:
                details[dist] = None
        details["torch"] = "installed" if find_spec("torch") is not None else None
        details["pinned_repos"] = dict(PINNED_REPOS)
        return details

    # -- coverage ----------------------------------------------------------

    def supports(self, family: str, model: str) -> Support:
        repos = FAMILY_REPOS.get(family)
        if repos is None:
            return Support(
                False,
                f"no Hub repo of family {family!r} ships a response_template in {TEMPLATE_FILE} "
                "(transformers only parses with a repo-shipped template)",
            )
        if model not in repos:
            return Support(
                False,
                f"{model} is not a repo verified to ship a response_template (verified: {', '.join(sorted(repos))})",
            )
        return Support(True)

    # -- loading -------------------------------------------------------------

    def _load(self, raw: ReplayInput) -> _Loaded:
        repo, revision = resolve_tokenizer(raw)
        key = (repo, revision)
        if key in self._loaded:
            return self._loaded[key]
        hf_hub_download = _engine("huggingface_hub").hf_hub_download
        AutoTokenizer = _engine("transformers").AutoTokenizer

        def read(rev: str, name: str) -> bytes:
            with open(hf_hub_download(repo, name, revision=rev), "rb") as f:
                return f.read()

        template_revision = PINNED_REPOS.get(repo, revision)
        template_config = json.loads(read(template_revision, TEMPLATE_FILE))
        shipped = template_config.get("response_template")
        if not isinstance(shipped, dict):
            # Harness problem: the static coverage table is stale for this revision.
            raise RuntimeError(f"{repo}@{template_revision} does not ship a response_template in {TEMPLATE_FILE}")
        tok = AutoTokenizer.from_pretrained(repo, revision=revision)
        if template_revision == revision:
            if getattr(tok, "response_template", None) != shipped:
                raise RuntimeError(f"{repo}@{revision}: tokenizer.response_template differs from {TEMPLATE_FILE}")
        elif read(revision, "tokenizer.json") != read(template_revision, "tokenizer.json") or not same_tokenizer(
            json.loads(read(revision, TEMPLATE_FILE)), template_config
        ):
            raise RuntimeError(f"{repo}: the tokenizer at {revision} differs from the one at {template_revision}")
        chat_template = getattr(tok, "chat_template", None)
        loaded = _Loaded(
            repo=repo,
            revision=revision,
            template_revision=template_revision,
            tok=tok,
            template=shipped,
            template_sha256=response_template_sha256(shipped),
            chat_template_sha256=(
                hashlib.sha256(chat_template.encode("utf-8")).hexdigest() if isinstance(chat_template, str) else None
            ),
        )
        self._loaded[key] = loaded
        return loaded

    def _prompt_text(self, st: _Loaded, raw: ReplayInput, tools: Sequence[ToolSpec]) -> str:
        """Rendered prompt (``add_generation_prompt=True``), decoded like ``serve``'s ``_decode_prefix``."""
        kwargs: dict[str, Any] = {}
        if raw.thinking is not None:
            kwargs["enable_thinking"] = raw.thinking
        out = st.tok.apply_chat_template(
            [{"role": "user", "content": PLACEHOLDER_USER_MESSAGE}],
            tools=[dict(t) for t in tools] or None,
            add_generation_prompt=True,
            tokenize=True,
            **kwargs,
        )
        ids = list(out["input_ids"] if hasattr(out, "keys") else out)
        prompt = st.tok.decode(ids)
        if not isinstance(prompt, str):
            raise TypeError(f"tokenizer.decode returned {type(prompt).__name__}")
        if raw.generation_prompt and not prompt.endswith(raw.generation_prompt):
            raise RuntimeError(
                f"rendered prompt does not end with the fixture's generation prompt {raw.generation_prompt!r}: "
                f"...{prompt[-80:]!r}"
            )
        return prompt

    # -- contract ------------------------------------------------------------

    def parser_config(self, raw: ReplayInput) -> dict[str, Any]:
        st = self._load(raw)
        return {
            "tool_parser": PARSER_NAME,
            "reasoning_parser": PARSER_NAME,
            "response_template_source": f"{st.repo}@{st.template_revision}:{TEMPLATE_FILE}",
            "response_template_sha256": st.template_sha256,
            "chat_template_sha256": st.chat_template_sha256,
            "tokenizer": f"{st.repo}@{st.revision}",
            "tokenizer_mode": "hf",
            "nonstream_entrypoint": "tokenizer.parse_response(output_ids, template, prefix=prompt, tools=tools)",
            "stream_entrypoint": "ResponseParser(template, prefix=prompt, tools=tools).feed/.finalize",
            "stream_detokenizer": "tokenizers.decoders.DecodeStream(skip_special_tokens=False), one id per step",
            "stream_events": "transformers.cli.serving.utils.response_events_to_chunks",
            "tools_passed_to_parser": True,
            "prompt": {
                "user_message": PLACEHOLDER_USER_MESSAGE,
                "add_generation_prompt": True,
                "enable_thinking": raw.thinking,
            },
            "stop_token_appended": None,
        }

    def units(self, raw: ReplayInput) -> list[int]:
        if raw.token_ids is not None:
            return list(raw.token_ids)
        st = self._load(raw)
        return list(st.tok.encode(raw.text, add_special_tokens=False))

    def parse(self, raw: ReplayInput, tools: Sequence[ToolSpec]) -> ParseResult:
        st = self._load(raw)
        prompt = self._prompt_text(st, raw, tools)
        ids = self.units(raw)
        _normalize_tool_call = _engine(SERVE_UTILS)._normalize_tool_call
        reasoning_fields = _engine("transformers.utils.chat_template_utils")._REASONING_FIELDS

        try:
            msg = st.tok.parse_response(ids, st.template, prefix=prompt, tools=[dict(t) for t in tools] or None)
            calls = [_normalize_tool_call(v) for v in msg.get("tool_calls") or []]
        except Exception as e:  # engine outcome, not a harness error
            return ParseResult(exception=exception_text(e))
        content = msg.get("content")
        reasoning = next((msg[f] for f in reasoning_fields if msg.get(f) is not None), None)
        return ParseResult(
            content=content if isinstance(content, str) else None,
            reasoning_content=reasoning if isinstance(reasoning, str) else None,
            tool_calls=tuple(ParsedToolCall(name=c.name, arguments_raw=c.arguments) for c in calls),
        )

    def special_token_ids(self, raw: ReplayInput) -> Collection[int] | None:
        """``all_special_ids`` plus every added token flagged ``special`` (Gemma 4: ``<|tool_call>``, ...)."""
        tok = self._load(raw).tok
        ids = {int(i) for i in tok.all_special_ids}
        ids.update(int(i) for i, t in tok.added_tokens_decoder.items() if getattr(t, "special", False))
        return frozenset(ids)

    def parse_stream(self, raw: ReplayInput, chunks: Sequence[Sequence[int]], tools: Sequence[ToolSpec]) -> ParseResult:
        """One ``feed`` per id group; the group's text comes from ``DecodeStream``, one id per step."""
        DecodeStream = _engine("tokenizers.decoders").DecodeStream

        st = self._load(raw)
        decoder = DecodeStream([], skip_special_tokens=False)
        rust_tok = st.tok._tokenizer

        def deltas() -> Iterator[str]:
            for group in chunks:
                pieces = (decoder.step(rust_tok, int(i)) for i in group)
                yield "".join(p for p in pieces if p is not None)

        return self._stream(st, raw, deltas(), tools)

    def parse_stream_text(self, raw: ReplayInput, deltas: Sequence[str], tools: Sequence[ToolSpec]) -> ParseResult:
        """Synthetic ``char:<seed>`` stress path: raw text deltas fed straight to ``ResponseParser``."""
        return self._stream(self._load(raw), raw, iter(deltas), tools)

    def _stream(self, st: _Loaded, raw: ReplayInput, deltas: Iterator[str], tools: Sequence[ToolSpec]) -> ParseResult:
        prompt = self._prompt_text(st, raw, tools)
        serve = _engine(SERVE_UTILS)
        ResponseParser = _engine("transformers.utils.chat_parsing.response_parser").ResponseParser

        acc = StreamAccumulator()

        def emit(events: Iterable[dict[str, Any]]) -> None:
            for item in serve.response_events_to_chunks(list(events)):
                if isinstance(item, serve.ToolCall):
                    # serve emits each call as one complete delta with a fresh index
                    acc.append_tool_call(item.name, item.arguments)
                elif isinstance(item, serve.ReasoningText):
                    acc.add_reasoning(str(item))
                elif isinstance(item, str):
                    acc.add_content(item)

        try:
            parser = ResponseParser(st.template, prefix=prompt, tools=[dict(t) for t in tools] or None)
            emit(parser.initial_events)
            for delta in deltas:
                if delta:  # serve skips steps that decode to nothing (incomplete UTF-8)
                    emit(parser.feed(delta))
            _msg, final_events = parser.finalize()
            emit(final_events)
        except Exception as e:  # engine outcome; deltas already streamed are kept
            return acc.result(exception=exception_text(e))
        return acc.result()

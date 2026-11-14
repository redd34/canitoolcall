# ruff: noqa: RUF001  (DeepSeek markers use U+FF5C and U+2581 on purpose)
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "tokenizers==0.23.2",
#     "mistral-common==1.12.0",
#     "huggingface-hub==1.33.0",
#     "jsonschema==4.25.1",
# ]
# ///
"""Derive ``max_tokens`` cuts of complete tool-call fixtures at structural boundaries.

Each derived fixture is a token prefix of an existing fixture (mostly ``engine_test``
ones): the first ``k`` of its ``output_token_ids``, with ``raw_output`` set to those
ids decoded with special tokens kept. That is exactly what ``max_tokens`` produces.
The cut sits right after a named piece of text (an open marker, a function name, a
key/value separator, a value before its inner close, ...), and the script refuses a
cut that does not fall on a token boundary.

The expectation follows section 2 of the recovery contract (``docs/formats/recovery.md``,
issue #36): the unfinished call must never be returned, and returning the text, dropping
it or raising are all accepted. A call that is complete before the cut is kept, as in the
``*-truncated-second-parallel-call`` fixtures. Whether text completed before the cut must
be kept is left open: cuts with no complete call use ``expected_error`` with all three
outcomes accepted, and cuts that keep a call are only taken from sources with no text
around the calls. A call whose arguments are all closed but whose close marker is cut off
is still unfinished, and its ``reason`` names the missing marker.

Every record keeps its source's provenance (``source_url``, ``revision``, ``license``,
``attribution``, ``template_sha256``) and says in ``notes`` which fixture it was cut from
and where. ``generator`` names this script, except for ``template_render`` sources: they
keep the generator that renders the source, and ``notes`` name this script instead.

Tags: every record is tagged ``truncated`` and ``x-derived-truncation``. A source tag is
carried when the cut still contains what it describes (see ``TAG_RULES``). Tags about
argument values are tested on the arguments that are complete in the cut: every argument
of a kept call, and the leading ``closed_args`` arguments of the cut call.

Only tokenizer files are downloaded, each at the fixture's pinned revision. Kimi's
tokenizer is tiktoken-based and only loads through repository code, so this script
decodes its ``tiktoken.model`` ranks directly instead of running that code.

Run from the repo root (deterministic; re-running leaves ``git diff`` clean)::

    uv run --script scripts/fixtures/derived_truncation.py
"""

from __future__ import annotations

import base64
import json
import re
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _shared.core import REPO_ROOT, validate_records, write_jsonl  # noqa: E402

GENERATOR = "scripts/fixtures/derived_truncation.py"
OUT_NAME = "derived-truncated.jsonl"
RULE = "docs/formats/recovery.md"
ACCEPT_ALL = ["no_tool_calls", "content_passthrough", "exception"]

# Format sub-variant tags always carry over, so that the matrix and triage group a derived cut
# with its source.
SUBFORMAT_TAG = re.compile(
    r"^x-(deepseek-v\d+|kimi-k\d|glm\d+|llama-(json|pythonic)|recipient-in-(role|channel)|v3-json-array|v11-args"
    r"|v13-compact|no-newlines|whitespace-variant-\d+|think-[a-z-]+|constrain-[a-z-]+)$"
)

Decoder = Callable[[Sequence[int]], str]


@dataclass(frozen=True)
class InCut:
    """What a cut still contains, for deciding which source tags it carries."""

    raw: str
    keep: int
    content: str | None
    """The source's expected content and reasoning, which a cut keeps only if they are in ``raw``."""
    reasoning: str | None
    nodes: tuple[Any, ...]
    """Every key and value, nested ones included, of the arguments complete in the cut."""


def _nodes(value: Any) -> Iterator[Any]:
    yield value
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _nodes(item)
    elif isinstance(value, list):
        for item in value:
            yield from _nodes(item)


def _any_str(c: InCut, test: Callable[[str], bool]) -> bool:
    return any(isinstance(n, str) and test(n) for n in c.nodes)


def _reasoning_kept(c: InCut) -> bool:
    return c.reasoning is not None and c.reasoning in c.raw


# Source tag -> (is it still in the cut?, why it is dropped otherwise). A source tag with neither a
# rule here nor a SUBFORMAT_TAG match stops the script, so that every dropped tag is a decision.
TAG_RULES: dict[str, tuple[Callable[[InCut], bool], str]] = {
    # Call counts describe the calls a parser returns, so they need a kept call. The corpus's other
    # truncated fixtures with no complete call carry no call-count tag either.
    "single-call": (lambda c: c.keep > 0, "the only call is unfinished, so the cut has no complete call"),
    "parallel-calls": (lambda c: c.keep > 0, "the cut ends inside the first call, so it has no complete call"),
    "text-before-call": (
        lambda c: c.content is not None and c.content.strip() in c.raw,
        "the text before the call is not in the cut",
    ),
    "reasoning": (_reasoning_kept, "the reasoning is not in the cut"),
    "reasoning-prefilled": (_reasoning_kept, "the reasoning is not in the cut"),
    # Argument values: only the arguments complete in the cut count.
    "numeric-arguments": (
        lambda c: any(isinstance(n, int | float) and not isinstance(n, bool) for n in c.nodes),
        "no argument value complete in the cut is a number",
    ),
    "unicode": (
        lambda c: _any_str(c, lambda s: not s.isascii()),
        "no argument complete in the cut has a non-ASCII character",
    ),
    "nested-json": (
        lambda c: any(isinstance(n, dict | list) for n in c.nodes),
        "no object or array argument is complete in the cut",
    ),
    "string-escapes": (
        lambda c: _any_str(c, lambda s: any(ch in s for ch in '"\\') or any(ord(ch) < 32 for ch in s)),
        "no string argument that needs a JSON escape is complete in the cut",
    ),
    "x-multiline-string": (lambda c: _any_str(c, lambda s: "\n" in s), "no multi-line string is complete in the cut"),
    "x-significant-whitespace": (
        lambda c: _any_str(c, lambda s: s != s.strip()),
        "no value with leading or trailing whitespace is complete in the cut",
    ),
    "x-pythonic-nested-list": (
        lambda c: any(isinstance(n, list) for n in c.nodes),
        "no list argument is complete in the cut",
    ),
    "x-schema-coercion": (
        lambda c: any(not isinstance(n, str) for n in c.nodes),
        "every value complete in the cut is a string, so none is coerced by the schema",
    ),
    "x-code-in-arguments": (
        lambda c: _any_str(c, lambda s: any(ch in s for ch in "(){};")),
        "no code argument is complete in the cut",
    ),
    # Format details of the call text itself.
    "x-parameters-key": (lambda c: '"parameters"' in c.raw, 'the cut ends before the "parameters" key'),
    "x-id-encodes-name": (
        lambda c: re.search(r"functions\.[\w.-]+:\d+<\|tool_call_argument_begin\|>", c.raw) is not None,
        "the cut ends before the call id functions.<name>:<idx> is complete",
    ),
    "x-typed-args": (lambda c: re.search(r' type="\w+"', c.raw) is not None, "the cut has no typed argument header"),
}


@dataclass(frozen=True)
class Cut:
    source: str
    """Id of the fixture that is cut."""
    slug: str
    """Suffix of the derived id: ``<source name>-cut-<slug>``."""
    after: str
    """The raw output is cut right after this text ..."""
    occurrence: int
    """... at its n-th occurrence."""
    where: str
    """Where the cut sits, completing "Output cut by max_tokens ..."."""
    keep: int = 0
    """Number of calls complete before the cut, which a correct parser keeps."""
    closed_args: int = 0
    """Number of leading arguments of the cut call whose value is closed before the cut. A JSON
    number with no delimiter after it is not closed: the next tokens could extend it."""


CUTS: tuple[Cut, ...] = (
    # ---- qwen3-hermes: <tool_call>{json}</tool_call>
    Cut(
        "qwen3-hermes/vllm-no-newlines-no-spaces",
        "after-name",
        '"name": "get_current_temperature","',
        1,
        "right after the complete function name, before the arguments key (its opening quote shares the token)",
    ),
    Cut(
        "qwen3-hermes/vllm-no-newlines-no-spaces",
        "before-outer-close",
        '"unit": "celsius"}}',
        1,
        "just before the outer close </tool_call>, with the call JSON itself complete",
        closed_args=2,
    ),
    Cut(
        "qwen3-hermes/sglang-text-before-call-with-space",
        "mid-arguments",
        '{"city":',
        1,
        "inside the arguments, at a key/value boundary (after the key, before the value), after text",
    ),
    Cut(
        "qwen3-hermes/vllm-two-calls-no-separator",
        "after-second-open",
        "<tool_call>",
        2,
        "right after the second call's open marker <tool_call>",
        keep=1,
    ),
    # ---- qwen3-xml: <tool_call><function=f><parameter=k>v</parameter></function></tool_call>
    Cut(
        "qwen3-xml/llamacpp-q35-parallel-no-think",
        "after-second-name",
        "<function=special_function_with_opt>",
        1,
        "right after the second call's function name",
        keep=1,
    ),
    Cut(
        "qwen3-xml/llamacpp-q35-reasoning-then-call",
        "before-outer-close",
        "</function>\n",
        1,
        "just before the outer close </tool_call>, with </function> already emitted, after reasoning",
        closed_args=1,
    ),
    Cut(
        "qwen3-xml/llamacpp-q35-reasoning-content-call",
        "after-name",
        "<function=run_in_terminal>\n",
        1,
        "right after the function name, before any parameter, after reasoning and text",
    ),
    Cut(
        "qwen3-xml/llamacpp-q35-python-multiline",
        "before-inner-close",
        "hello()\n",
        1,
        "inside the only parameter value, just before its inner close </parameter>",
    ),
    # ---- llama: Llama 3 JSON and Llama 4 pythonic
    Cut(
        "llama/l3-vllm-parameters-key",
        "after-name",
        '{"name": "searchTool",',
        1,
        "right after the complete function name and its comma, before the parameters key",
    ),
    Cut(
        "llama/l3-vllm-parameters-key",
        "after-last-value",
        '"limit": 10',
        1,
        "after the last argument value, before the two closing braces (one token, so no cut fits between them)",
        closed_args=1,
    ),
    Cut(
        "llama/l4-vllm-pythonic-more-types",
        "after-name",
        "[register_user",
        1,
        "right after the function name, before its open parenthesis",
    ),
    Cut(
        "llama/l4-vllm-pythonic-more-types",
        "mid-arguments",
        "age=9,",
        1,
        "inside the arguments, at a key/value boundary (after a complete keyword argument)",
        closed_args=2,
    ),
    # ---- kimi: K2 section tokens, K3 XTML
    Cut(
        "kimi/k2i-vllm-multiline-json",
        "mid-arguments",
        '"name": "test",\n',
        1,
        "inside the argument JSON, at a key/value boundary (after a complete pair), after text",
        closed_args=1,
    ),
    Cut(
        "kimi/k2i-single-call",
        "before-section-close",
        "<|tool_call_end|>",
        1,
        "after the call's <|tool_call_end|>, before <|tool_calls_section_end|>",
        keep=1,
    ),
    Cut(
        "kimi/k3-vllm-call-thinking-disabled",
        "before-inner-close",
        'type="number"<|sep|>1',
        1,
        "inside the only argument, just before its inner close <|close|>argument",
    ),
    Cut(
        "kimi/k3-vllm-call-thinking-disabled",
        "before-tools-close",
        "<|close|>call<|sep|>",
        1,
        "after the call's <|close|>call, before <|close|>tools",
        keep=1,
    ),
    # ---- glm: <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
    Cut(
        "glm/glm45-vllm-newline-format-two-calls",
        "after-second-name",
        "<tool_call>get_current_weather",
        2,
        "right after the second call's function name",
        keep=1,
    ),
    Cut(
        "glm/glm47-vllm-whitespace-preserved-in-value",
        "after-name",
        "<tool_call>get_weather",
        1,
        "right after the function name, before any <arg_key>",
    ),
    Cut(
        "glm/glm45-vllm-coerces-schema-types",
        "before-inner-close",
        "<arg_value>3.14",
        1,
        "inside an argument value, just before its inner close </arg_value>",
        closed_args=2,
    ),
    # ---- deepseek: V3 fenced JSON, V4.1 DSML
    Cut(
        "deepseek/vllm-v3-parallel-no-newline",
        "after-second-name",
        "search_hotels",
        1,
        "right after the second call's function name",
        keep=1,
    ),
    Cut(
        "deepseek/vllm-v3-single-call",
        "before-call-end",
        "```",
        2,
        "after the closing code fence, just before <｜tool▁call▁end｜>",
        closed_args=2,
    ),
    Cut(
        "deepseek/vllm-v3-single-call",
        "before-section-close",
        "<｜tool▁call▁end｜>",
        1,
        "after the call's <｜tool▁call▁end｜>, before <｜tool▁calls▁end｜>",
        keep=1,
    ),
    Cut(
        "deepseek/vllm-v41-text-before-parallel-calls",
        "between-parameters",
        "</｜DSML｜ parameter>\n",
        1,
        "inside the first invoke, between its two parameters, after text",
        closed_args=1,
    ),
    # ---- gemma4: <|tool_call>call:name{k:v}<tool_call|>
    Cut(
        "gemma4/vllm-two-calls",
        "after-second-name",
        "<|tool_call>call:get_time",
        1,
        "right after the second call's function name",
        keep=1,
    ),
    Cut(
        "gemma4/vllm-multiple-arguments",
        "before-outer-close",
        'celsius<|"|>}',
        1,
        "just before the outer close <tool_call|>, with the argument object closed",
        closed_args=2,
    ),
    Cut(
        "gemma4/vllm-text-before-call",
        "after-open",
        "<|tool_call>",
        1,
        "right after the outer open marker <|tool_call>, after text",
    ),
    # ---- mistral: v11 [TOOL_CALLS]name[ARGS]{json}, v3 [TOOL_CALLS][{json}]
    Cut(
        "mistral/vllm-v11-parallel",
        "after-second-args",
        "[ARGS]",
        2,
        "right after the second call's [ARGS], before its JSON",
        keep=1,
    ),
    Cut(
        "mistral/vllm-v11-single-call",
        "after-name",
        "[TOOL_CALLS]add_this_and_that",
        1,
        "right after the function name, before [ARGS]",
    ),
    Cut(
        "mistral/vllm-v3-no-space",
        "mid-arguments",
        '{"a": 3.5,',
        1,
        "inside the arguments, at a key/value boundary (after a complete pair)",
        closed_args=1,
    ),
    # ---- gpt-oss: Harmony
    Cut(
        "gpt-oss/vllm-channel-first-recipient",
        "mid-arguments",
        '{"city":',
        1,
        "inside the argument JSON, at a key/value boundary (after the key), after reasoning",
    ),
    Cut(
        "gpt-oss/llamacpp-edit-code-arguments",
        "mid-arguments",
        '"newString": "if (part < 4) {",',
        1,
        "inside the argument JSON, at a key/value boundary (after a complete pair), after reasoning",
        closed_args=2,
    ),
)


# --------------------------------------------------------------------------- tokenizers


def _hub_file(repo: str, revision: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, revision=revision))


def _kimi_decoder(repo: str, revision: str) -> Decoder:
    """tiktoken decoding without running tokenization_kimi.py: concatenate each id's bytes."""
    table: dict[int, bytes] = {}
    for line in _hub_file(repo, revision, "tiktoken.model").read_text(encoding="utf-8").splitlines():
        if line:
            token, rank = line.split()
            table[int(rank)] = base64.b64decode(token)
    config = json.loads(_hub_file(repo, revision, "tokenizer_config.json").read_text(encoding="utf-8"))
    for tid, spec in config["added_tokens_decoder"].items():
        table[int(tid)] = spec["content"].encode("utf-8")
    return lambda ids: b"".join(table[i] for i in ids).decode("utf-8", errors="replace")


def _mistral_decoder(repo: str, revision: str) -> Decoder:
    from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    raw = MistralTokenizer.from_file(str(_hub_file(repo, revision, "tekken.json"))).instruct_tokenizer.tokenizer
    return lambda ids: str(raw.decode(list(ids), special_token_policy=SpecialTokenPolicy.KEEP))


def _hf_decoder(repo: str, revision: str) -> Decoder:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(_hub_file(repo, revision, "tokenizer.json")))
    return lambda ids: str(tok.decode(list(ids), skip_special_tokens=False))


@cache
def decoder(repo: str, revision: str, mode: str) -> Decoder:
    if mode == "mistral":
        return _mistral_decoder(repo, revision)
    if repo.startswith("moonshotai/Kimi-"):
        return _kimi_decoder(repo, revision)
    return _hf_decoder(repo, revision)


# --------------------------------------------------------------------------- derivation


def load_corpus() -> dict[str, dict[str, Any]]:
    """Every fixture except the ones this script writes, by id."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted((REPO_ROOT / "fixtures").glob("*/*.jsonl")):
        if path.name == OUT_NAME:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["id"]] = rec
    return out


def cut_point(raw: str, after: str, occurrence: int) -> int:
    pos = -1
    for _ in range(occurrence):
        pos = raw.index(after, pos + 1)
    return pos + len(after)


def token_prefix(src: dict[str, Any], cut: Cut) -> list[int]:
    """The shortest id prefix whose decode is exactly the raw output up to the cut."""
    pin = src["tokenizer"]
    decode = decoder(pin["repo"], pin["revision"], pin["mode"])
    ids: list[int] = src["output_token_ids"]
    raw: str = src["raw_output"]
    if decode(ids) != raw:
        raise AssertionError(f"{src['id']}: the ids do not decode to raw_output")
    target = raw[: cut_point(raw, cut.after, cut.occurrence)]
    for k in range(1, len(ids)):
        text = decode(ids[:k])
        if text == target:
            return ids[:k]
        if len(text) > len(target) and not text.startswith("�"):
            break
    raise AssertionError(f"{src['id']}: no token boundary right after {cut.after!r} (occurrence {cut.occurrence})")


@cache
def family_markers(family: str) -> frozenset[str]:
    path = REPO_ROOT / "fixtures" / family / "family.json"
    return frozenset(json.loads(path.read_text(encoding="utf-8"))["markers"])


def in_cut(src: dict[str, Any], cut: Cut, raw: str) -> InCut:
    """The kept calls' arguments and the cut call's closed leading arguments, with the text kept."""
    calls = src["expected"]["tool_calls"]
    done = [dict(c["arguments"]) for c in calls[: cut.keep]]
    if cut.closed_args:
        items = list(calls[cut.keep]["arguments"].items())[: cut.closed_args]
        if len(items) < cut.closed_args or any(key not in raw for key, _ in items):
            raise AssertionError(f"{src['id']}: the cut does not hold {cut.closed_args} arguments of call {cut.keep}")
        done.append(dict(items))
    nodes = tuple(n for args in done for key, value in args.items() for n in (key, *_nodes(value)))
    exp = src["expected"]
    return InCut(raw, cut.keep, exp["content"], exp["reasoning_content"], nodes)


def tag_decisions(src: dict[str, Any], c: InCut) -> list[tuple[str, str | None]]:
    """Each source tag, with ``None`` when the cut carries it or else the reason it is dropped."""
    out: list[tuple[str, str | None]] = []
    for tag in src["tags"]:
        if SUBFORMAT_TAG.match(tag):
            out.append((tag, None))
        elif tag in TAG_RULES:
            present, reason = TAG_RULES[tag]
            out.append((tag, None if present(c) else reason))
        else:
            raise AssertionError(f"{src['id']}: source tag {tag!r} has no TAG_RULES entry")
    return out


def missing_close(src: dict[str, Any], cut: Cut, k: int, decode: Decoder) -> str | None:
    """The close marker cut off a call whose arguments are all closed, if the cut is there."""
    calls = src["expected"]["tool_calls"]
    if cut.keep >= len(calls) or not cut.closed_args or cut.closed_args < len(calls[cut.keep]["arguments"]):
        return None
    marker = decode(src["output_token_ids"][k : k + 1])
    if marker not in family_markers(src["family"]):
        raise AssertionError(f"{src['id']}: the token after a complete payload is {marker!r}, not a family marker")
    return marker


def derive(src: dict[str, Any], cut: Cut) -> dict[str, Any]:
    ids = token_prefix(src, cut)
    pin = src["tokenizer"]
    decode = decoder(pin["repo"], pin["revision"], pin["mode"])
    raw = decode(ids)
    exp = src["expected"]
    name = src["id"].split("/", 1)[1]
    provenance = dict(src["provenance"])
    notes = (
        f"Derived, not quoted: the first {len(ids)} of {len(src['output_token_ids'])} output tokens of {src['id']} "
        f"({provenance['kind']}), cut {cut.where}. Expectation by section 2 of the recovery contract ({RULE}): "
        "the unfinished call must not be returned."
    )
    if provenance["kind"] == "template_render":
        notes += (
            f" The source is a history render of the official chat template, made by {provenance['generator']}; "
            f"{GENERATOR} cut it."
        )
    else:
        provenance["generator"] = GENERATOR
    rec: dict[str, Any] = {
        "id": f"{src['family']}/{name}-cut-{cut.slug}",
        "family": src["family"],
        "models": src["models"],
        "spec_version": src["spec_version"],
        "provenance": provenance,
        "tools": src["tools"],
        "raw_output": raw,
        "output_token_ids": ids,
        "tokenizer": pin,
    }
    for key in ("generation_prompt", "thinking"):
        if key in src:
            rec[key] = src[key]
    if cut.keep:
        # Kept calls come only from sources with no text around the calls, so `expected` does not
        # decide whether completed text before the cut must be kept.
        if exp["content"] is not None or exp["reasoning_content"] is not None:
            raise AssertionError(f"{src['id']}: a cut that keeps a call needs a source with no text")
        rec["expected"] = {"content": None, "reasoning_content": None, "tool_calls": exp["tool_calls"][: cut.keep]}
        notes += f" The first {cut.keep} call(s) are complete before the cut, so a correct parser keeps them."
    else:
        marker = missing_close(src, cut, len(ids), decode)
        why = "no complete call exists"
        if marker:
            why = f"the call's close marker {marker} is missing, so the call is unfinished"
        rec["expected_error"] = {"reason": f"Output cut by max_tokens {cut.where}: {why}.", "accept": ACCEPT_ALL}
        if exp["content"] is not None or exp["reasoning_content"] is not None:
            notes += " Whether the text completed before the cut must be kept is not decided, so all outcomes pass."
    carried = [tag for tag, dropped in tag_decisions(src, in_cut(src, cut, raw)) if dropped is None]
    rec["tags"] = ["truncated", "x-derived-truncation", *carried]
    rec["notes"] = notes
    return rec


def main() -> None:
    corpus = load_corpus()
    existing = {(r["family"], r["raw_output"]): r["id"] for r in corpus.values()}
    by_family: dict[str, list[dict[str, Any]]] = {}
    for cut in CUTS:
        src = corpus[cut.source]
        if src["provenance"]["kind"] != "engine_test" and src["provenance"]["kind"] != "template_render":
            raise AssertionError(f"{cut.source}: unexpected source kind")
        rec = derive(src, cut)
        dup = existing.get((rec["family"], rec["raw_output"]))
        if dup is not None:
            raise AssertionError(f"{rec['id']} duplicates {dup}")
        by_family.setdefault(rec["family"], []).append(rec)
    validate_records([r for recs in by_family.values() for r in recs])
    for family, recs in sorted(by_family.items()):
        write_jsonl(REPO_ROOT / "fixtures" / family / OUT_NAME, recs)


if __name__ == "__main__":
    main()

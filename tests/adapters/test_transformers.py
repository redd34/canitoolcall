"""Tests for the HF transformers adapter.

* Fast unit tests run in the main dev env. They never import transformers.
* ``@pytest.mark.engine("transformers")`` tests drive the real worker inside
  ``.venvs/transformers`` (built by ``scripts/engines/transformers.sh``). They
  are skipped when that env is missing. They replay:

  - the core Qwen3 sample, which must come back ``unsupported`` (Qwen3 repos
    ship no ``response_template``);
  - Gemma 4 outputs rendered in the engine venv through the OFFICIAL chat
    template of ``google/gemma-4-E2B-it`` (``apply_chat_template(tokenize=True)``,
    prompt ids sliced off, cut at the first ``generation_config.json`` stop id),
    exactly as spec/README.md prescribes for ``template_render`` fixtures;
  - every record in ``fixtures/gemma4/*.jsonl``, once the fixture group has
    added some. Each must replay without harness errors.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from canitoolcall.adapters import adapter_class
from canitoolcall.adapters.base import ReplayInput
from canitoolcall.adapters.transformers import (
    FAMILY_REPOS,
    PINNED_REPOS,
    StreamAccumulator,
    TransformersAdapter,
    exception_text,
    resolve_tokenizer,
    response_template_sha256,
    same_tokenizer,
)
from canitoolcall.chunking import DEFAULT_STRATEGIES
from canitoolcall.fixtures import TokenizerPin, load_families, read_jsonl, repo_root
from canitoolcall.results import ParsedToolCall, ParseResult

ROOT = repo_root() or Path(__file__).resolve().parents[2]
GEMMA = "google/gemma-4-E2B-it"
GEMMA_REV = PINNED_REPOS[GEMMA]
QWEN38 = "Qwen/Qwen3.8-27B"
STRATEGIES = [s.id for s in DEFAULT_STRATEGIES]


# --------------------------------------------------------------------------
# fast unit tests (no engine)
# --------------------------------------------------------------------------


def test_registered_and_pinned() -> None:
    cls = adapter_class("transformers")
    assert cls is TransformersAdapter
    assert cls.pinned_version == "5.23.0"


@pytest.mark.parametrize(("family", "repo"), sorted((f, r) for f, repos in FAMILY_REPOS.items() for r in repos))
def test_supports_mapped_repos(family: str, repo: str) -> None:
    assert TransformersAdapter().supports(family, repo)


def test_qwen38_is_mapped_at_the_revision_that_ships_its_template() -> None:
    assert FAMILY_REPOS["qwen3-xml"] == {QWEN38}
    assert PINNED_REPOS[QWEN38] == "37561816e66b5b9153b1b64c10968c18c1e6f4a0"


@pytest.mark.parametrize(
    ("family", "model", "why"),
    [
        ("qwen3-hermes", "Qwen/Qwen3-0.6B", "no Hub repo of family"),
        ("gpt-oss", "openai/gpt-oss-20b", "no Hub repo of family"),
        ("gemma4", "unsloth/gemma-4-E2B-it", "not a repo verified"),
        ("gemma4", "google/gemma-4-E2B-it-qat-q4_0-gguf", "not a repo verified"),
        ("qwen3-xml", "Qwen/Qwen3.5-9B", "not a repo verified"),
        ("qwen3-xml", "Qwen/Qwen3-Coder-30B-A3B-Instruct", "not a repo verified"),
    ],
)
def test_supports_declines_honestly(family: str, model: str, why: str) -> None:
    sup = TransformersAdapter().supports(family, model)
    assert not sup
    assert sup.reason is not None and why in sup.reason


def test_family_map_only_lists_pinned_repos() -> None:
    assert set(FAMILY_REPOS) == {"gemma4", "qwen3-xml"}
    assert all(r in PINNED_REPOS for repos in FAMILY_REPOS.values() for r in repos)
    assert all(len(rev) == 40 for rev in PINNED_REPOS.values())


def test_resolve_tokenizer_prefers_fixture_pin() -> None:
    base = ReplayInput(fixture_id="gemma4/x", family="gemma4", model=GEMMA, text="")
    assert resolve_tokenizer(base) == (GEMMA, GEMMA_REV)
    pinned = ReplayInput(
        fixture_id="gemma4/x", family="gemma4", model=GEMMA, text="", tokenizer=TokenizerPin(GEMMA, "a" * 40)
    )
    assert resolve_tokenizer(pinned) == (GEMMA, "a" * 40)
    with pytest.raises(ValueError, match="no pinned revision"):
        resolve_tokenizer(ReplayInput(fixture_id="x/y", family="x", model="org/unknown", text=""))


def test_same_tokenizer_ignores_only_the_template_keys() -> None:
    a = {"model_max_length": 10, "chat_template": "x", "response_template": None, "added_tokens_decoder": {"1": {}}}
    b = {
        "model_max_length": 10,
        "chat_template": "y",
        "response_template": {"fields": {}},
        "added_tokens_decoder": {"1": {}},
    }
    assert same_tokenizer(a, b)
    assert not same_tokenizer(a, {**b, "model_max_length": 11})
    assert not same_tokenizer(a, {k: v for k, v in b.items() if k != "added_tokens_decoder"})


def test_template_sha_is_key_order_independent() -> None:
    a = {"fields": {"content": {"content": "text"}}, "defaults": {"role": "assistant"}}
    b = {"defaults": {"role": "assistant"}, "fields": {"content": {"content": "text"}}}
    assert response_template_sha256(a) == response_template_sha256(b)
    assert len(response_template_sha256(a)) == 64


def test_stream_accumulator_is_openai_client_style() -> None:
    acc = StreamAccumulator()
    assert acc.result() == ParseResult()
    acc.add_reasoning("think ")
    acc.add_reasoning("more")
    acc.add_content("hi")
    acc.append_tool_call("f", '{"a": 1}')
    acc.append_tool_call("g", "{}")
    assert acc.result(exception="ValueError: x") == ParseResult(
        content="hi",
        reasoning_content="think more",
        tool_calls=(ParsedToolCall("f", '{"a": 1}'), ParsedToolCall("g", "{}")),
        exception="ValueError: x",
    )


def test_exception_text() -> None:
    assert exception_text(ValueError("bad")) == "ValueError: bad"


# --------------------------------------------------------------------------
# engine tests (real transformers 5.23.0 in .venvs/transformers)
# --------------------------------------------------------------------------

# Renders Gemma 4 assistant turns through the official chat template and prints
# fixture records (spec/README.md "template_render"). Runs in the engine venv.
GEMMA4_GENERATOR = r"""
import hashlib, json, sys
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from transformers.utils import logging

logging.set_verbosity_error()
REPO, REV = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
with open(hf_hub_download(REPO, "generation_config.json", revision=REV)) as f:
    stop_ids = set(json.load(f)["eos_token_id"])

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
                   "required": ["city"]}}}
SEARCH = {"type": "function", "function": {"name": "search", "description": "Search the web.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"},
                   "safe": {"type": "boolean"}, "filters": {"type": "object"}}, "required": ["query"]}}}
CASES = [
    ("reasoning-single-unicode", [WEATHER], "Weather in Paris?", None, "User wants weather; call the tool.",
     [{"name": "get_weather", "arguments": {"city": "Paris", "unit": "°C 🌤"}}],
     ["single-call", "reasoning", "unicode"]),
    ("parallel-nested-numeric", [WEATHER, SEARCH], "Weather in Zürich, and search cafés.", None, None,
     [{"name": "get_weather", "arguments": {"city": "Zürich"}},
      {"name": "search", "arguments": {"query": "best café", "limit": 3, "safe": True,
                                       "filters": {"tags": ["a", "b"], "max": 2.5}}}],
     ["parallel-calls", "nested-json", "numeric-arguments", "unicode"]),
    ("no-call-content", [WEATHER], "Say hello.", "Hello! How can I help you today?", None, [],
     ["no-call"]),
    ("reasoning-content", [WEATHER], "What is 2+2?", "2 + 2 = 4.", "Simple arithmetic.", [],
     ["no-call", "reasoning"]),
]
for name, tools, user, content, reasoning, calls, tags in CASES:
    asst = {"role": "assistant", "content": content or ""}
    if reasoning:
        asst["reasoning_content"] = reasoning
    if calls:
        asst["tool_calls"] = [{"type": "function", "function": c} for c in calls]
    def ids(msgs, gen):
        out = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=gen, tokenize=True)
        return list(out["input_ids"] if hasattr(out, "keys") else out)
    user_msgs = [{"role": "user", "content": user}]
    prompt, full = ids(user_msgs, True), ids(user_msgs + [asst], False)
    assert full[: len(prompt)] == prompt, "prompt is not a prefix of the rendered conversation"
    out = full[len(prompt):]
    out = out[: next(i for i, t in enumerate(out) if t in stop_ids)]
    print(json.dumps({
        "id": f"gemma4/x-adapter-test-{name}", "family": "gemma4", "models": [REPO], "spec_version": "0.1",
        "provenance": {"kind": "template_render", "revision": REV, "license": "Apache-2.0",
                       "source_url": f"https://huggingface.co/{REPO}/blob/{REV}/chat_template.jinja",
                       "generator": "tests/adapters/test_transformers.py::GEMMA4_GENERATOR",
                       "template_sha256": hashlib.sha256(tok.chat_template.encode()).hexdigest()},
        "tools": tools, "raw_output": tok.decode(out, skip_special_tokens=False), "output_token_ids": out,
        "tokenizer": {"repo": REPO, "revision": REV, "mode": "hf"},
        "expected": {"content": content, "reasoning_content": reasoning, "tool_calls": calls},
        "tags": tags,
    }, ensure_ascii=False))
"""

GEMMA4_FAMILY: dict[str, Any] = {
    "slug": "gemma4",
    "name": "Gemma 4 (adapter test)",
    "spec_version": "0.1",
    "has_reasoning": True,
    "markers": ["<|tool_call>", "<tool_call|>", "<|channel>", "<channel|>", "<|turn>", "<turn|>", '<|"|>'],
    "reference_models": [
        {
            "repo": GEMMA,
            "revision": GEMMA_REV,
            "default_generation_prompt": "<|turn>model\n",
            "stop_tokens": ["<eos>", "<turn|>", "<|tool_response>"],
        }
    ],
    "format_notes": "docs/formats/gemma4.md",
}


def _hf_cache_warm(repo: str, revision: str) -> bool:
    hub = Path(
        os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    )
    snap = hub / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    return all((snap / f).exists() for f in ("tokenizer_config.json", "tokenizer.json", "generation_config.json"))


def _engine_env() -> dict[str, str]:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    if _hf_cache_warm(GEMMA, GEMMA_REV):
        env["HF_HUB_OFFLINE"] = "1"
    return env


class Worker:
    """The real JSON-lines worker, running in the transformers venv."""

    def __init__(self, python: Path) -> None:
        self.proc = subprocess.Popen(
            [str(python), "-m", "canitoolcall.adapters.worker", "transformers"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=_engine_env(),
        )

    def ask(self, msg: Mapping[str, Any]) -> dict[str, Any]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        assert line, "worker exited"
        reply: dict[str, Any] = json.loads(line)
        return reply

    def close(self) -> None:
        try:
            self.ask({"op": "shutdown"})
        finally:
            self.proc.wait(timeout=30)


def _render_gemma4(python: Path) -> list[dict[str, Any]]:
    proc = subprocess.run(
        [str(python), "-c", GEMMA4_GENERATOR, GEMMA, GEMMA_REV],
        capture_output=True,
        text=True,
        env=_engine_env(),
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]


def _decoded(result: Mapping[str, Any], *, strip: bool = False) -> dict[str, Any]:
    """A ParseResult dict with arguments decoded, for comparison with ``expected``.

    ``strip`` drops surrounding whitespace from content/reasoning: the checks
    grade such differences ``soft_pass`` (spec/README.md, soft-v1).
    """

    def text(v: str | None) -> str | None:
        return (v.strip() if strip and v else v) or None

    return {
        "content": text(result["content"]),
        "reasoning_content": text(result["reasoning_content"]),
        "tool_calls": [{"name": t["name"], "arguments": json.loads(t["arguments_raw"])} for t in result["tool_calls"]],
        "exception": result["exception"],
    }


@pytest.mark.engine("transformers")
def test_worker_hello_and_stdlib_core(engine_python_for: Path) -> None:
    w = Worker(engine_python_for)
    try:
        hello = w.ask({"op": "hello"})
        assert hello["ok"], hello
        assert hello["engine"] == "transformers" and hello["version"] == TransformersAdapter.pinned_version
        assert hello["details"]["torch"] is None
        assert hello["details"]["pinned_repos"] == PINNED_REPOS
    finally:
        w.close()


@pytest.mark.engine("transformers")
def test_core_sample_is_unsupported(engine_python_for: Path, sample_fixtures_dir: Path) -> None:
    ((_, rec),) = list(read_jsonl(sample_fixtures_dir / "qwen3-hermes" / "sample.jsonl"))
    fam = load_families(sample_fixtures_dir)["qwen3-hermes"].to_dict()
    w = Worker(engine_python_for)
    try:
        reply = w.ask({"op": "replay", "fixture": rec, "family": fam, "strategies": STRATEGIES})
    finally:
        w.close()
    assert reply["ok"], reply
    assert reply["supported"] is False
    assert "response_template" in reply["reason"]


@pytest.mark.engine("transformers")
def test_gemma4_rendered_outputs_replay_correctly(engine_python_for: Path) -> None:
    records = _render_gemma4(engine_python_for)
    assert len(records) == 4
    w = Worker(engine_python_for)
    try:
        for rec in records:
            reply = w.ask({"op": "replay", "fixture": rec, "family": GEMMA4_FAMILY, "strategies": STRATEGIES})
            assert reply["ok"], reply.get("error")
            assert reply["supported"], reply
            cfg = reply["parser_config"]
            assert cfg["response_template_source"] == f"{GEMMA}@{GEMMA_REV}:tokenizer_config.json"
            assert len(cfg["response_template_sha256"]) == 64
            want = {**rec["expected"], "exception": None}
            assert _decoded(reply["nonstream"]) == want, (rec["id"], rec["raw_output"])
            assert set(reply["streams"]) == set(STRATEGIES), reply.get("skipped")
            for sid, got in reply["streams"].items():
                assert _decoded(got, strip=True) == want, (rec["id"], sid, got)
                if rec["expected"]["reasoning_content"]:
                    # Observed engine behaviour at 5.17.0 and 5.23.0 (whitespace-only, soft_pass):
                    # ResponseParser's thinking ``region_chunk`` keeps the "\n" before
                    # ``<channel|>`` while its ``region_close`` value (and parse_response)
                    # strips it, so ``transformers serve`` streams a trailing newline.
                    assert got["reasoning_content"] == rec["expected"]["reasoning_content"] + "\n", (sid, got)
    finally:
        w.close()


@pytest.mark.engine("transformers")
def test_gemma4_synthetic_char_stream(engine_python_for: Path) -> None:
    """``char:<seed>`` feeds raw text deltas (splitting special tokens) to the same parser."""
    rec = next(r for r in _render_gemma4(engine_python_for) if "parallel" in r["id"])
    w = Worker(engine_python_for)
    try:
        reply = w.ask({"op": "replay", "fixture": rec, "family": GEMMA4_FAMILY, "strategies": ["char:0", "char:7"]})
    finally:
        w.close()
    assert reply["ok"], reply.get("error")
    assert set(reply["streams"]) == {"char:0", "char:7"}, reply.get("skipped")
    want = {**rec["expected"], "exception": None}
    for sid, got in reply["streams"].items():
        assert _decoded(got, strip=True) == want, (sid, got)


@pytest.mark.engine("transformers")
def test_gemma4_truncated_call_is_an_outcome_not_a_harness_error(engine_python_for: Path) -> None:
    """Cut a rendered tool call mid-arguments: the worker must answer ``ok``."""
    rec = next(r for r in _render_gemma4(engine_python_for) if "parallel" in r["id"])
    ids = rec["output_token_ids"][: len(rec["output_token_ids"]) // 2]
    trunc = {**rec, "id": "gemma4/x-adapter-test-truncated", "output_token_ids": ids, "tags": ["truncated"]}
    trunc.pop("expected")
    trunc["expected_error"] = {
        "reason": "cut mid-call",
        "accept": ["no_tool_calls", "content_passthrough", "exception"],
    }
    w = Worker(engine_python_for)
    try:
        reply = w.ask({"op": "replay", "fixture": trunc, "family": GEMMA4_FAMILY, "strategies": STRATEGIES})
    finally:
        w.close()
    assert reply["ok"], reply.get("error")
    assert reply["supported"]
    assert set(reply["streams"]) == set(STRATEGIES)


def _corpus_gemma4() -> list[tuple[str, dict[str, Any]]]:
    d = ROOT / "fixtures" / "gemma4"
    return [(f"{p.name}:{n}", rec) for p in sorted(d.glob("*.jsonl")) for n, rec in read_jsonl(p)] if d.is_dir() else []


@pytest.mark.engine("transformers")
def test_gemma4_corpus_replays_without_harness_errors(engine_python_for: Path) -> None:
    corpus = _corpus_gemma4()
    if not corpus:
        pytest.skip("fixtures/gemma4 has no records yet")
    fam_path = ROOT / "fixtures" / "gemma4" / "family.json"
    fam = json.loads(fam_path.read_text(encoding="utf-8")) if fam_path.exists() else None
    w = Worker(engine_python_for)
    try:
        for where, rec in corpus:
            reply = w.ask({"op": "replay", "fixture": rec, "family": fam, "strategies": STRATEGIES})
            assert reply["ok"], (where, reply.get("error"))
    finally:
        w.close()


def _corpus_qwen38() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    d = ROOT / "fixtures" / "qwen3-xml"
    fam = json.loads((d / "family.json").read_text(encoding="utf-8"))
    recs = [rec for p in sorted(d.glob("*.jsonl")) for _, rec in read_jsonl(p) if rec["models"][0] == QWEN38]
    return fam, recs


@pytest.mark.engine("transformers")
def test_qwen38_replays_with_the_template_from_its_pinned_revision(engine_python_for: Path) -> None:
    """Fixtures pin the tokenizer at a revision without a template; the template comes from PINNED_REPOS."""
    fam, recs = _corpus_qwen38()
    assert recs
    w = Worker(engine_python_for)
    try:
        for rec in recs:
            reply = w.ask({"op": "replay", "fixture": rec, "family": fam, "strategies": STRATEGIES})
            assert reply["ok"], (rec["id"], reply.get("error"))
            assert reply["supported"], (rec["id"], reply)
            cfg = reply["parser_config"]
            assert cfg["response_template_source"] == f"{QWEN38}@{PINNED_REPOS[QWEN38]}:tokenizer_config.json"
            assert cfg["tokenizer"] == f"{QWEN38}@{rec['tokenizer']['revision']}"
            if rec["id"] == "qwen3-xml/q38-single-call":
                # The template names the field reasoning_content (serve reads it via _REASONING_FIELDS).
                want = {**rec["expected"], "exception": None}
                assert _decoded(reply["nonstream"], strip=True) == want, reply["nonstream"]
    finally:
        w.close()

"""Tests for the Ollama adapter.

* Fast unit tests run in the main dev env. They need no engine and no harness:
  parser selection, the llama-server UTF-8 hold-back port, the token-repeat
  guard, and OpenAI-style accumulation.
* ``@pytest.mark.engine("ollama")`` tests drive the real worker in
  ``.venvs/ollama`` (or ``$CANITOOLCALL_OLLAMA_PYTHON``), which talks to the
  harnesses built by ``scripts/engines/ollama.sh``. They are skipped when that
  env or the harness binaries are missing. Each replay also needs the model's
  vocab-only GGUF from ``scripts/engines/gguf_vocab.sh`` (in ``.engines/gguf``
  or ``$CANITOOLCALL_GGUF_DIR``), and a case is skipped when its GGUF is absent.
  They replay:

  - the core Qwen3 sample (template render with token ids) under every default
    chunking strategy, plus a truncated slice of it;
  - raw outputs copied verbatim from Ollama's own parser tests at the pinned
    commit (MIT). Where the Ollama test states the expected parse, the test
    asserts it;
  - every record in ``fixtures/<family>/*.jsonl`` that the adapter supports.
    Each must replay without a harness error.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from canitoolcall.adapters import adapter_class
from canitoolcall.adapters.base import AdapterUnavailable
from canitoolcall.adapters.ollama import (
    BIN_DIR_ENV,
    GGUF_DIR_ENV,
    PARSER_RULES,
    TOKEN_REPEAT_LIMIT,
    OllamaAdapter,
    accumulate_nonstream,
    accumulate_stream,
    eog_stop_index,
    parser_name,
    repeat_abort_index,
    runner_events,
    select_rule,
    utf8_complete_len,
)
from canitoolcall.chunking import DEFAULT_STRATEGIES
from canitoolcall.fixtures import read_jsonl, repo_root
from canitoolcall.results import ParsedToolCall

ROOT = repo_root() or Path(__file__).resolve().parents[2]
PIN = OllamaAdapter.pinned_version
STRATEGIES = [s.id for s in DEFAULT_STRATEGIES]
SAMPLE_DIR = ROOT / "tests" / "core" / "data" / "fixtures" / "qwen3-hermes"

# --------------------------------------------------------------------------
# fast unit tests (no engine)
# --------------------------------------------------------------------------


def test_registered_and_pinned() -> None:
    assert adapter_class("ollama") is OllamaAdapter
    script = (ROOT / "scripts" / "engines" / "ollama.sh").read_text()
    m = re.search(r"^OLLAMA_REF=([0-9a-f]{40})", script, re.MULTILINE)
    assert m is not None and m.group(1) == PIN


@pytest.mark.parametrize(
    ("family", "model", "thinking", "parser"),
    [
        ("qwen3-hermes", "Qwen/Qwen3-0.6B", None, "qwen3-thinking"),
        ("qwen3-hermes", "Qwen/Qwen3-0.6B", True, "qwen3-thinking"),
        ("qwen3-hermes", "Qwen/Qwen3-0.6B", False, "qwen3"),
        ("qwen3-hermes", "Qwen/Qwen3-4B-Thinking-2507", None, "qwen3-thinking"),
        ("qwen3-hermes", "Qwen/Qwen3-4B-Instruct-2507", None, "qwen3"),
        ("qwen3-xml", "Qwen/Qwen3-Coder-30B-A3B-Instruct", None, "qwen3-coder"),
        ("qwen3-xml", "Qwen/Qwen3.5-9B", None, "qwen3.5"),
        ("gpt-oss", "openai/gpt-oss-20b", None, "harmony"),
        ("deepseek", "deepseek-ai/DeepSeek-V3.1", False, "deepseek3"),
        ("deepseek", "deepseek-ai/DeepSeek-V3.1-Terminus", None, "deepseek3"),
        ("glm", "zai-org/GLM-4.7", None, "glm-4.7"),
        ("gemma4", "google/gemma-4-31B-it", None, "gemma4"),
        ("mistral", "mistralai/Ministral-3-8B-Instruct-2512", None, "ministral"),
        ("mistral", "mistralai/Devstral-2-123B-Instruct-2512", None, "ministral"),
        ("nemotron", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", None, "nemotron-3-nano"),
        ("nemotron", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", False, "nemotron-3-nano"),
        ("nemotron", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", None, "nemotron-3-nano"),
        ("nemotron", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", None, "nemotron-3-nano"),
        ("nemotron", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", None, "nemotron-3-nano"),
        ("nemotron", "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", None, "nemotron-3.5-nano"),
    ],
)
def test_parser_selection(family: str, model: str, thinking: bool | None, parser: str) -> None:
    assert OllamaAdapter().supports(family, model)
    rule = select_rule(family, model)
    assert rule is not None
    assert parser_name(rule, thinking) == parser


@pytest.mark.parametrize(
    ("family", "model"),
    [
        ("kimi", "moonshotai/Kimi-K2-Instruct-0905"),
        ("llama", "meta-llama/Llama-3.1-8B-Instruct"),
        ("glm", "zai-org/GLM-4.5"),
        ("glm", "zai-org/GLM-4.6"),
        ("deepseek", "deepseek-ai/DeepSeek-V3-0324"),
        ("mistral", "mistralai/Mistral-Small-3.2-24B-Instruct-2506"),
        ("mistral", "mistralai/Devstral-Small-2-24B-Instruct-2512"),
        ("qwen3-xml", "Qwen/Qwen3-0.6B"),
        ("nemotron", "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16"),
        # Unverified Nemotron 3/3.5 repos: the rules name only the five checked templates.
        ("nemotron", "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"),
        ("nemotron", "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"),
        ("nemotron", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16"),
        ("nemotron", "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM"),
        ("seed-oss", "ByteDance-Seed/Seed-OSS-36B-Instruct"),
    ],
)
def test_unsupported_is_honest(family: str, model: str) -> None:
    sup = OllamaAdapter().supports(family, model)
    assert not sup
    assert sup.reason is not None and "legacy template parser not covered" in sup.reason


def test_every_rule_cites_a_source() -> None:
    for rule in PARSER_RULES:
        assert rule.source
        re.compile(rule.pattern)


def test_utf8_hold_back_matches_llama_server() -> None:
    zue = "ü".encode()  # two bytes
    emoji = "🌤".encode()  # four bytes
    assert utf8_complete_len(b"ab") == 2
    assert utf8_complete_len(b"a" + zue[:1]) == 1
    assert utf8_complete_len(b"a" + emoji[:3]) == 1
    assert utf8_complete_len(b"a" + emoji) == 5
    pieces = [b"a", zue[:1], zue[1:], emoji[:2], emoji[2:3], emoji[3:], b"b"]
    assert runner_events(pieces) == ["a", None, "ü", None, None, "🌤", "b"]


def test_token_repeat_guard() -> None:
    assert repeat_abort_index(["a", "b", "c"]) is None
    # The 101st repeat of the same trimmed text trips the guard.
    events: list[str | None] = ["x"] * (TOKEN_REPEAT_LIMIT + 2)
    assert repeat_abort_index(events) == TOKEN_REPEAT_LIMIT + 1
    # Held-back tokens send no event and do not count.
    events = ["x", None] * (TOKEN_REPEAT_LIMIT + 2)
    assert repeat_abort_index(events) == 2 * (TOKEN_REPEAT_LIMIT + 1)
    # Whitespace-only events trim to "", which already equals Go's initial lastToken,
    # so every one counts; the trailing EOG + stop events count too.
    assert repeat_abort_index([" "] * (TOKEN_REPEAT_LIMIT - 1)) == TOKEN_REPEAT_LIMIT
    assert repeat_abort_index([" "] * (TOKEN_REPEAT_LIMIT - 2)) is None


def test_eog_stop_index() -> None:
    assert eog_stop_index([]) is None
    assert eog_stop_index([False, False]) is None
    # llama-server processes the EOG token itself, then stops.
    assert eog_stop_index([False, True, False, True]) == 2


def _event(content: str = "", thinking: str = "", calls: list[tuple[int, str, str]] | None = None) -> dict[str, Any]:
    return {
        "content": content,
        "thinking": thinking,
        "tool_calls": [
            {"id": "call_x", "index": i, "type": "function", "function": {"name": n, "arguments": a}}
            for i, n, a in calls or []
        ],
    }


def test_accumulation() -> None:
    events = [
        _event(thinking="t1"),
        _event(thinking="t2", content="c"),
        _event(calls=[(0, "f", '{"a":1}')]),
        _event(calls=[(0, "f", "")]),
        _event(calls=[(1, "g", "{}")]),
    ]
    stream = accumulate_stream(events)
    assert stream.reasoning_content == "t1t2" and stream.content == "c"
    # A repeated name at the same index is concatenated, as openai-python's accumulate_delta does.
    assert stream.tool_calls == (ParsedToolCall("ff", '{"a":1}'), ParsedToolCall("g", "{}"))
    nonstream = accumulate_nonstream(events, has_tools=True)
    assert nonstream.tool_calls == (ParsedToolCall("f", '{"a":1}'), ParsedToolCall("f", ""), ParsedToolCall("g", "{}"))
    # writeChatResponse keeps tool calls only when the request had tools.
    assert accumulate_nonstream(events, has_tools=False).tool_calls == ()
    assert accumulate_nonstream([_event()], has_tools=True).content is None


def test_missing_harness_is_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BIN_DIR_ENV, str(tmp_path))
    with pytest.raises(AdapterUnavailable, match=r"scripts/engines/ollama\.sh"):
        OllamaAdapter().version()


def test_module_is_stdlib_only() -> None:
    src = (ROOT / "src" / "canitoolcall" / "adapters" / "ollama.py").read_text()
    imports = re.findall(r"^(?:from|import) ([\w.]+)", src, re.MULTILINE)
    allowed = {"__future__", "hashlib", "json", "os", "re", "subprocess", "sys", "threading"}
    for mod in imports:
        top = mod.split(".")[0]
        assert top in allowed or top in {"collections", "dataclasses", "pathlib", "typing", "canitoolcall"}, mod


# --------------------------------------------------------------------------
# engine tests (real worker, real Ollama parsers)
# --------------------------------------------------------------------------


def _bin_dir() -> Path:
    env = os.environ.get(BIN_DIR_ENV)
    return Path(env) if env else ROOT / ".engines" / "ollama" / "bin"


def _gguf_dir() -> Path:
    env = os.environ.get(GGUF_DIR_ENV)
    return Path(env) if env else ROOT / ".engines" / "gguf"


def _need_gguf(repo: str) -> None:
    path = _gguf_dir() / f"{repo.replace('/', '--')}.vocab.gguf"
    if not path.is_file():
        pytest.skip(f"no vocab GGUF for {repo} (run scripts/engines/gguf_vocab.sh {repo}@<revision>)")


class Worker:
    """The real protocol-v1 worker for the ``ollama`` engine, in its own interpreter."""

    def __init__(self, python: Path) -> None:
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        self.proc = subprocess.Popen(
            [str(python), "-m", "canitoolcall.adapters.worker", "ollama"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=env,
        )

    def call(self, msg: Mapping[str, Any]) -> dict[str, Any]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        assert line, "worker exited"
        reply: dict[str, Any] = json.loads(line)
        return reply

    def replay(self, record: Mapping[str, Any], family: Mapping[str, Any] | None = None) -> dict[str, Any]:
        reply = self.call({"op": "replay", "fixture": record, "family": family, "strategies": STRATEGIES})
        assert reply["ok"], reply.get("error")
        return reply

    def close(self) -> None:
        self.call({"op": "shutdown"})
        self.proc.wait(timeout=30)


@pytest.fixture(scope="module")
def worker(request: pytest.FixtureRequest) -> Iterator[Worker]:
    # Resolved like tests/conftest.py's engine_python_for, at module scope.
    from canitoolcall.adapters import engine_python

    for exe in ("ctcreplay", "ctc-detok"):
        if not (_bin_dir() / exe).is_file():
            pytest.skip(f"{_bin_dir() / exe} not built (run scripts/engines/ollama.sh)")
    w = Worker(engine_python("ollama", ROOT))
    yield w
    w.close()


def _args(result: Mapping[str, Any]) -> list[tuple[str, Any]]:
    return [(tc["name"], json.loads(tc["arguments_raw"])) for tc in result["tool_calls"]]


def _all_results(reply: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {"nonstream": reply["nonstream"], **reply["streams"]}


@pytest.mark.engine("ollama")
def test_hello_reports_pins(worker: Worker) -> None:
    hello = worker.call({"op": "hello"})
    assert hello["ok"], hello.get("error")
    assert hello["engine"] == "ollama"
    assert hello["commit"] == PIN and hello["version"] == PIN[:8]
    assert hello["details"]["llama_cpp_build"].startswith("b")


def _sample() -> tuple[dict[str, Any], dict[str, Any]]:
    ((_, rec),) = list(read_jsonl(SAMPLE_DIR / "sample.jsonl"))
    fam = json.loads((SAMPLE_DIR / "family.json").read_text(encoding="utf-8"))
    return rec, fam


@pytest.mark.engine("ollama")
def test_core_sample_every_strategy(worker: Worker) -> None:
    rec, fam = _sample()
    _need_gguf(rec["tokenizer"]["repo"])
    reply = worker.replay(rec, fam)
    assert reply["supported"]
    cfg = reply["parser_config"]
    assert cfg["parser"] == "qwen3-thinking" and cfg["think_effective"] is True
    assert "<tool_call>" in cfg["preserved_tokens"]
    assert cfg["vocab_gguf_revision_matches_tokenizer"] is True
    assert set(reply["streams"]) | set(reply.get("skipped", {})) == set(STRATEGIES)
    assert "token" in reply["streams"] and "one" in reply["streams"]
    want = [(tc["name"], tc["arguments"]) for tc in rec["expected"]["tool_calls"]]
    for sid, result in _all_results(reply).items():
        assert result["exception"] is None, sid
        assert _args(result) == want, sid
        assert result["content"] is None, sid
        # Observed: whether thinking keeps its leading "\n" depends on the token grouping
        # (a whitespace-level split variance; see docs/DESIGN.md section 6).
        assert (result["reasoning_content"] or "").strip() == rec["expected"]["reasoning_content"], sid


@pytest.mark.engine("ollama")
def test_truncated_sample_degrades_without_harness_error(worker: Worker) -> None:
    rec, fam = _sample()
    _need_gguf(rec["tokenizer"]["repo"])
    raw = rec["raw_output"]
    cut = {k: v for k, v in rec.items() if k not in ("output_token_ids", "tokenizer", "expected")}
    cut |= {"id": "qwen3-hermes/test-truncated", "raw_output": raw[: raw.index('"unit"')]}
    cut["expected_error"] = {"reason": "truncated inside the first call", "accept": ["no_tool_calls"]}
    reply = worker.replay(cut, fam)
    for sid, result in _all_results(reply).items():
        assert result["tool_calls"] == [], sid


def _tool(name: str, props: Mapping[str, str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": {"type": "object", "properties": {k: {"type": t} for k, t in props.items()}},
        },
    }


def _engine_test_record(
    family: str, model: str, path_line: str, raw: str, tools: list[dict[str, Any]], thinking: bool | None
) -> dict[str, Any]:
    """A test-only record around a raw output copied verbatim from Ollama's tests (MIT)."""
    return {
        "id": f"{family}/ollama-test-{path_line.rsplit('/', 1)[-1]}",
        "family": family,
        "models": [model],
        "spec_version": "0.1",
        "provenance": {
            "kind": "engine_test",
            "source_url": f"https://github.com/ollama/ollama/blob/{PIN}/{path_line}",
            "revision": PIN,
            "license": "MIT",
        },
        "tools": tools,
        "raw_output": raw,
        "tags": [],
        "thinking": thinking,
        "expected": None,
    }


# (family, model, thinking, source path#line, tools, raw output, expected parse or None)
# Raw outputs are copied verbatim. A trailing stop token is removed, because raw_output
# ends before the stop token. Expected parses are those the cited Ollama test asserts.
OLLAMA_TEST_CASES: list[tuple[str, str, bool | None, str, list[dict[str, Any]], str, dict[str, Any] | None]] = [
    (
        "gpt-oss",
        "openai/gpt-oss-20b",
        None,
        "harmony/harmonyparser_test.go#L118",
        [_tool("get_current_weather", {"location": "string"})],
        '<|channel|>analysis<|message|>User asks weather in SF. We need location. Use get_current_weather with location "San Francisco, CA".<|end|><|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"location":"San Francisco, CA"}',  # noqa: E501 - verbatim; "<|call|>" stop token removed
        None,
    ),
    (
        "glm",
        "zai-org/GLM-4.7",
        None,
        "model/parsers/glm47_test.go#L21",
        [_tool("calculate", {"count": "integer", "enabled": "boolean"})],
        "plan</think>Answer<tool_call>calculate<arg_key>count</arg_key><arg_value>3</arg_value><arg_key>enabled</arg_key><arg_value>true</arg_value></tool_call>",
        {"reasoning": "plan", "content": "Answer", "calls": [("calculate", {"count": 3, "enabled": True})]},
    ),
    (
        "qwen3-hermes",
        "Qwen/Qwen3-4B-Instruct-2507",
        False,
        "model/parsers/qwen3_test.go#L120",
        [_tool("get_weather", {"location": "string", "unit": "string"})],
        '<tool_call>{"name":"get_weather","arguments":{"location":"San Francisco","unit":"celsius"}}</tool_call>',
        {
            "reasoning": None,
            "content": None,
            "calls": [("get_weather", {"location": "San Francisco", "unit": "celsius"})],
        },
    ),
    (
        "deepseek",
        "deepseek-ai/DeepSeek-V3.1",
        False,
        "model/parsers/deepseek3_test.go#L48",
        [_tool("get_weather", {"location": "string"})],
        'I\'ll check the weather.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',  # noqa: E501, RUF001 - verbatim
        {"reasoning": None, "content": "I'll check the weather.", "calls": [("get_weather", {"location": "Paris"})]},
    ),
    (
        "qwen3-xml",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        None,
        "model/parsers/qwen3coder_test.go#L1042",
        [_tool("first", {"a": "string"}), _tool("second", {"b": "string"}), _tool("third", {"c": "string"})],
        "<tool_call><function=first><parameter=a>1</parameter></function></tool_call>\n<tool_call><function=second><parameter=b>2</parameter></function></tool_call>\n<tool_call><function=third><parameter=c>3</parameter></function></tool_call>",
        {
            "reasoning": None,
            "content": None,
            "calls": [("first", {"a": "1"}), ("second", {"b": "2"}), ("third", {"c": "3"})],
        },
    ),
    (
        "gemma4",
        "google/gemma-4-E2B-it",
        None,
        "model/parsers/gemma4_test.go#L64",
        [_tool("get_weather", {"location": "string", "units": "string"})],
        '<|tool_call>call:get_weather{location:<|"|>Paris<|"|>,units:<|"|>metric<|"|>}<tool_call|>',
        {"reasoning": None, "content": None, "calls": [("get_weather", {"location": "Paris", "units": "metric"})]},
    ),
]


@pytest.mark.engine("ollama")
@pytest.mark.parametrize(
    ("family", "model", "thinking", "path_line", "tools", "raw", "expected"),
    OLLAMA_TEST_CASES,
    ids=[f"{c[0]}:{c[3].rsplit('/', 1)[-1]}" for c in OLLAMA_TEST_CASES],
)
def test_ollama_test_outputs(
    worker: Worker,
    family: str,
    model: str,
    thinking: bool | None,
    path_line: str,
    tools: list[dict[str, Any]],
    raw: str,
    expected: dict[str, Any] | None,
) -> None:
    _need_gguf(model)
    reply = worker.replay(_engine_test_record(family, model, path_line, raw, tools, thinking))
    assert reply["supported"], reply["reason"]
    results = _all_results(reply)
    if expected is None:
        # harmony: the analysis channel is thinking and the commentary message is the call.
        analysis = raw.split("<|message|>", 1)[1].split("<|end|>", 1)[0]
        body = raw.rsplit("<|message|>", 1)[1]
        expected = {"reasoning": analysis, "content": None, "calls": [("get_current_weather", json.loads(body))]}
    for sid, result in results.items():
        assert result["exception"] is None, sid
        assert result["reasoning_content"] == expected["reasoning"], sid
        assert result["content"] == expected["content"], sid
        assert _args(result) == expected["calls"], sid


def _corpus() -> list[Any]:
    params: list[Any] = []
    adapter = OllamaAdapter()
    for fam_file in sorted((ROOT / "fixtures").glob("*/family.json")):
        family = json.loads(fam_file.read_text(encoding="utf-8"))
        for jl in sorted(fam_file.parent.glob("*.jsonl")):
            for _, rec in read_jsonl(jl):
                if adapter.supports(rec["family"], rec["models"][0]):
                    params.append(pytest.param(rec, family, id=rec["id"]))
    return params or [pytest.param(None, None, id="no-supported-fixtures-yet")]


@pytest.mark.engine("ollama")
@pytest.mark.parametrize(("record", "family"), _corpus())
def test_fixture_corpus_replays(worker: Worker, record: dict[str, Any] | None, family: dict[str, Any] | None) -> None:
    if record is None:
        pytest.skip("fixtures/ has no records the Ollama adapter supports yet")
    tok = record.get("tokenizer") or {}
    _need_gguf(tok.get("repo") or record["models"][0])
    reply = worker.replay(record, family)
    assert reply["supported"]
    assert reply["parser_config"]["parser"]
    assert set(reply["streams"]) | set(reply.get("skipped", {})) == set(STRATEGIES)


@pytest.mark.engine("ollama")
def test_interior_stop_token_ends_generation(worker: Worker) -> None:
    """gpt-oss/vllm-sequential-calls has an interior <|call|> (an EOG id). llama-server,
    which Ollama runs, stops there, so only the first call can reach Ollama's parser."""
    fam_dir = ROOT / "fixtures" / "gpt-oss"
    rec = next(r for _, r in read_jsonl(fam_dir / "imported.jsonl") if r["id"] == "gpt-oss/vllm-sequential-calls")
    family = json.loads((fam_dir / "family.json").read_text(encoding="utf-8"))
    _need_gguf(rec["tokenizer"]["repo"])
    reply = worker.replay(rec, family)
    for sid, result in _all_results(reply).items():
        assert result["exception"] is None, sid
        assert [name for name, _ in _args(result)] == ["get_weather"], sid

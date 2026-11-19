"""Tests for the SGLang adapter.

* Fast unit tests run in the main dev env and never import SGLang.
* ``@pytest.mark.engine("sglang")`` tests drive the real worker inside
  ``.venvs/sglang`` (built by ``scripts/engines/sglang.sh``) and are skipped
  when that env is missing. They replay:

  - the core Qwen3 sample (tests/core/data, rendered through Qwen3's official
    template);
  - tool-call turns rendered in the engine venv through the OFFICIAL chat
    template of one cached reference model per family
    (``apply_chat_template(tokenize=True)``, prompt ids sliced off, cut at the
    first ``generation_config.json`` stop id, as spec/README.md prescribes for
    ``template_render`` fixtures);
  - the gpt-oss tool call from the Harmony spec (openai/harmony
    ``docs/format.md`` line 402), which SGLang only extracts when ``<|call|>``
    is kept in the text;
  - the first records of every family in ``fixtures/`` once the fixture groups
    have added them. Each must replay without harness errors.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from canitoolcall.adapters import adapter_class, engine_python
from canitoolcall.adapters.sglang import (
    HARMONY_CALL,
    HARMONY_RETURN,
    PARSERS,
    SglangAdapter,
    StreamAccumulator,
    exception_text,
    harmony_stop_token,
    resolve_parsers,
)
from canitoolcall.chunking import DEFAULT_STRATEGIES
from canitoolcall.fixtures import load_families, read_jsonl, repo_root
from canitoolcall.results import ParsedToolCall, ParseResult
from canitoolcall.runner import ENGINE_ENV_DEFAULTS

ROOT = repo_root() or Path(__file__).resolve().parents[2]
# Engine tests read tokenizers/configs from the HF cache only (DESIGN.md rule 10);
# export HF_HUB_OFFLINE=0 once to warm the cache. The engine's CPU-only defaults
# (CUDA_VISIBLE_DEVICES=-1) apply here as they do under `canitoolcall run`.
ENGINE_ENV = {
    **os.environ,
    **ENGINE_ENV_DEFAULTS["sglang"],
    "PYTHONPATH": str(ROOT / "src"),
    "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
}
CACHE_MISS = ("LocalEntryNotFoundError", "OfflineModeIsEnabled", "outgoing traffic has been disabled")
SAMPLE_DIR = ROOT / "tests" / "core" / "data" / "fixtures"
STRATEGIES = [s.id for s in DEFAULT_STRATEGIES]


# --------------------------------------------------------------------------
# fast unit tests (no engine)
# --------------------------------------------------------------------------


def test_registered_and_pinned() -> None:
    cls = adapter_class("sglang")
    assert cls is SglangAdapter
    assert cls.pinned_version == "0.5.23"
    assert cls.supports_text_deltas


def test_setup_smoke_test_hides_gpus() -> None:
    # sglang.sh's import smoke test failed with "No module named 'flashinfer'" on a host with a visible GPU.
    script = (ROOT / "scripts" / "engines" / "sglang.sh").read_text(encoding="utf-8")
    assert re.search(r'^CUDA_VISIBLE_DEVICES=-1 "\$PY" ', script, re.MULTILINE)


def test_setup_reunpacks_when_the_pin_changes() -> None:
    # An env built at the old pin kept its unpacked source: the guard only checked that sglang/ existed.
    script = (ROOT / "scripts" / "engines" / "sglang.sh").read_text(encoding="utf-8")
    assert 'if [ ! -d "$SRC_DIR/sglang-$SGLANG_VERSION.dist-info" ]; then' in script


def test_module_does_not_import_sglang() -> None:
    code = (
        "import sys, canitoolcall.adapters.sglang as m; m.SglangAdapter().supports('glm', 'x'); "
        "print('sglang' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False"


@pytest.mark.parametrize(
    ("family", "model", "tool_parser", "reasoning_parser"),
    [
        ("qwen3-hermes", "Qwen/Qwen3-8B", "qwen25", "qwen3"),
        ("qwen3-hermes", "Qwen/Qwen3-4B-Thinking-2507", "qwen25", "qwen3"),
        ("qwen3-hermes", "Qwen/Qwen3-4B-Instruct-2507", "qwen25", None),
        ("qwen3-hermes", "Qwen/Qwen2.5-7B-Instruct", "qwen25", None),
        ("qwen3-xml", "Qwen/Qwen3-Coder-30B-A3B-Instruct", "qwen3_coder", None),
        ("qwen3-xml", "Qwen/Qwen3.6-35B-A3B", "qwen3_coder", "qwen3"),
        ("gpt-oss", "openai/gpt-oss-20b", "gpt-oss", "gpt-oss"),
        ("deepseek", "deepseek-ai/DeepSeek-V3.1", "deepseekv31", "deepseek-v3"),
        ("deepseek", "deepseek-ai/DeepSeek-V3.2", "deepseekv32", "deepseek-v3"),
        ("deepseek", "deepseek-ai/DeepSeek-V4-Flash", "deepseekv4", "deepseek-v4"),
        ("deepseek", "deepseek-ai/DeepSeek-V4.1-Flash", "deepseekv41", "deepseek-v41"),
        ("deepseek", "deepseek-ai/DeepSeek-R1-0528", "deepseekv3", "deepseek-r1"),
        ("kimi", "moonshotai/Kimi-K2-Instruct-0905", "kimi_k2", None),
        ("kimi", "moonshotai/Kimi-K2.6", "kimi_k2", "kimi_k2"),
        ("kimi", "moonshotai/Kimi-K3", "kimi_k3", "kimi_k3"),
        ("glm", "zai-org/GLM-4.5-Air", "glm45", "glm45"),
        ("glm", "zai-org/GLM-4.7", "glm47", "glm45"),
        ("glm", "zai-org/GLM-5.3-Flash", "glm47", "glm45"),
        ("llama", "meta-llama/Llama-3.1-8B-Instruct", "llama3", None),
        ("mistral", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "mistral", None),
        ("mistral", "mistralai/Magistral-Small-2509", "mistral", "mistral"),
        ("gemma4", "google/gemma-4-31B-it", "gemma4", "gemma4"),
        ("minimax", "MiniMaxAI/MiniMax-M2.7", "minimax-m2", "minimax"),
        ("minimax", "MiniMaxAI/MiniMax-M2", "minimax-m2", "minimax"),
    ],
)
def test_parser_mapping(family: str, model: str, tool_parser: str, reasoning_parser: str | None) -> None:
    rule = resolve_parsers(family, model)
    assert rule is not None
    assert (rule.tool_call_parser, rule.reasoning_parser) == (tool_parser, reasoning_parser)
    assert SglangAdapter().supports(family, model)


@pytest.mark.parametrize(
    ("family", "model"),
    [
        ("nope", "org/model"),
        ("llama", "meta-llama/Llama-2-7b-chat-hf"),
        ("minimax", "MiniMaxAI/MiniMax-M3"),
        ("seed-oss", "ByteDance-Seed/Seed-OSS-36B-Instruct"),
    ],
)
def test_unknown_pairs_are_unsupported(family: str, model: str) -> None:
    sup = SglangAdapter().supports(family, model)
    assert not sup
    assert sup.reason and "no SGLang parser mapping" in sup.reason


def test_every_family_slug_is_mapped() -> None:
    slugs = {"qwen3-hermes", "qwen3-xml", "gpt-oss", "deepseek", "kimi", "glm", "minimax", "llama", "mistral", "gemma4"}
    assert set(PARSERS) == slugs


@pytest.mark.parametrize(
    ("text", "stop"),
    [
        # openai/harmony docs/format.md: recipient in the channel section
        (
            "<|channel|>analysis<|message|>x<|end|><|start|>assistant<|channel|>commentary "
            'to=functions.get_weather <|constrain|>json<|message|>{"a":1}',
            HARMONY_CALL,
        ),
        # recipient in the role section (first message after the prompt has no <|start|>)
        ('to=functions.f<|channel|>commentary json<|message|>{"a":1}', HARMONY_CALL),
        ("<|channel|>analysis<|message|>x<|end|><|start|>assistant<|channel|>final<|message|>Hi", HARMONY_RETURN),
        # a recipient in an EARLIER message does not count
        (
            "<|channel|>commentary to=functions.f<|message|>{}<|end|><|start|>assistant<|channel|>final<|message|>ok",
            HARMONY_RETURN,
        ),
    ],
)
def test_harmony_stop_token(text: str, stop: str) -> None:
    assert harmony_stop_token(text) == stop


def _sse(delta: Mapping[str, Any]) -> str:
    return (
        "data: "
        + json.dumps({"id": "x", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta}]})
        + "\n\n"
    )


def test_stream_accumulator_decodes_sse_openai_client_style() -> None:
    acc = StreamAccumulator()
    acc.add_reasoning("think ")
    acc.add_reasoning(None)
    acc.add_reasoning("more")
    acc.add_sse(_sse({"content": "Hi"}))
    acc.add_sse(_sse({"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "f", "arguments": ""}}]}))
    acc.add_sse(_sse({"tool_calls": [{"index": 0, "id": None, "function": {"name": None, "arguments": '{"a": '}}]}))
    acc.add_sse(_sse({"tool_calls": [{"index": 1, "id": "call_2", "function": {"name": "g", "arguments": "{}"}}]}))
    acc.add_sse(_sse({"tool_calls": [{"index": 0, "function": {"name": None, "arguments": "1}"}}]}))
    acc.add_sse("data: [DONE]\n\n")
    assert acc.result() == ParseResult(
        content="Hi",
        reasoning_content="think more",
        tool_calls=(ParsedToolCall("f", '{"a": 1}'), ParsedToolCall("g", "{}")),
    )
    # A name re-sent in a later delta is concatenated, as openai-python's accumulate_delta does.
    acc.add_sse(_sse({"tool_calls": [{"index": 1, "function": {"name": "g"}}]}))
    assert acc.result().tool_calls[1].name == "gg"
    assert StreamAccumulator().result(exception="E: x") == ParseResult(exception="E: x")


def test_exception_text() -> None:
    assert exception_text(ValueError("bad")) == "ValueError: bad"


# --------------------------------------------------------------------------
# engine tests (real SGLang 0.5.23 in .venvs/sglang)
# --------------------------------------------------------------------------


class Worker:
    """The real JSON-lines worker running in the SGLang venv."""

    def __init__(self, python: Path) -> None:
        self.proc = subprocess.Popen(
            [str(python), "-W", "ignore", "-m", "canitoolcall.adapters.worker", "sglang"],
            env=ENGINE_ENV,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )

    def ask(self, msg: Mapping[str, Any]) -> dict[str, Any]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        assert line, "worker exited"
        reply: dict[str, Any] = json.loads(line)
        return reply

    def replay(
        self, fixture: Mapping[str, Any], family: Mapping[str, Any] | None, strategies: list[str] = STRATEGIES
    ) -> dict[str, Any]:
        reply = self.ask({"op": "replay", "fixture": fixture, "family": family, "strategies": strategies})
        if not reply["ok"] and any(m in reply["error"] for m in CACHE_MISS):
            pytest.skip(f"{fixture['id']}: tokenizer/config not in the HF cache (run once with HF_HUB_OFFLINE=0)")
        assert reply["ok"], reply.get("error")
        return reply

    def close(self) -> None:
        try:
            self.ask({"op": "shutdown"})
        finally:
            self.proc.wait(timeout=60)


@pytest.fixture(scope="module")
def worker() -> Iterator[Worker]:
    w = Worker(engine_python("sglang", ROOT))
    yield w
    w.close()


def _calls(result: Mapping[str, Any]) -> list[tuple[str, Any]]:
    return [(c["name"], json.loads(c["arguments_raw"])) for c in result["tool_calls"]]


def _expected_calls(fixture: Mapping[str, Any]) -> list[tuple[str, Any]]:
    return [(c["name"], c["arguments"]) for c in fixture["expected"]["tool_calls"]]


@pytest.mark.engine("sglang")
def test_engine_hello(worker: Worker) -> None:
    hello = worker.ask({"op": "hello"})
    assert hello["ok"] and hello["engine"] == "sglang"
    assert hello["version"] == SglangAdapter.pinned_version
    assert hello["details"]["deps"]["transformers"] == "5.19.0"  # sglang 0.5.23 pins it


@pytest.mark.engine("sglang")
def test_engine_parser_names_exist(engine_python_for: Path) -> None:
    """Every PARSERS name is registered in SGLang 0.5.23."""
    code = r"""
import json
from canitoolcall.adapters.sglang import PARSERS
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.managers.schedule_batch import INIT_INCREMENTAL_DETOKENIZATION_OFFSET
bad = []
for fam, rules in PARSERS.items():
    for r in rules:
        if r.tool_call_parser and r.tool_call_parser not in FunctionCallParser.ToolCallParserEnum:
            bad.append((fam, r.tool_call_parser))
        if r.reasoning_parser and r.reasoning_parser not in ReasoningParser.DetectorMap:
            bad.append((fam, r.reasoning_parser))
print(json.dumps({"bad": bad, "offset": INIT_INCREMENTAL_DETOKENIZATION_OFFSET}))
"""
    out = subprocess.run(
        [str(engine_python_for), "-W", "ignore", "-c", code],
        env=ENGINE_ENV,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res == {"bad": [], "offset": 5}


@pytest.mark.engine("sglang")
def test_engine_core_sample(worker: Worker) -> None:
    ((_, rec),) = list(read_jsonl(SAMPLE_DIR / "qwen3-hermes" / "sample.jsonl"))
    fam = load_families(SAMPLE_DIR)["qwen3-hermes"].to_dict()
    r = worker.replay(rec, fam)
    cfg = r["parser_config"]
    assert (cfg["tool_call_parser"], cfg["reasoning_parser"]) == ("qwen25", "qwen3")
    assert cfg["tool_call_detector"] == "Qwen25Detector"
    assert cfg["tokenizer"]["revision"] == rec["tokenizer"]["revision"]
    assert cfg["skip_special_tokens"] is False  # tools present
    assert cfg["tool_choice"] == "auto"
    assert cfg["stop"] == {
        "token": "<|im_end|>",
        "id": 151645,
        "rule": "first stop token of the reference model",
        "appended": True,
        "finish_reason": "stop",
        "kept_by_engine": False,
    }
    expected = _expected_calls(rec)
    assert _calls(r["nonstream"]) == expected
    assert r["nonstream"]["reasoning_content"].strip() == rec["expected"]["reasoning_content"]
    assert r["nonstream"]["content"] is None
    for sid in ("token", "rand:1:8", "rand:2:8", "rand:3:8"):
        assert _calls(r["streams"][sid]) == expected, sid
    # Observed with SGLang 0.5.20 and 0.5.23 (candidate discrepancy, untriaged): when the
    # whole output arrives in one delta, qwen25 streams only the first call's
    # name and no arguments.
    assert [c["name"] for c in r["streams"]["one"]["tool_calls"]] == ["get_weather"]
    assert r["streams"]["one"]["tool_calls"][0]["arguments_raw"] == ""
    assert r["skipped"] == {}


# (family, repo, revision, reasoning text or "", tool call id)
RENDER_CASES = [
    (
        "qwen3-hermes",
        "Qwen/Qwen3-4B-Thinking-2507",
        "768f209d9ea81521153ed38c47d515654e938aea",
        "User wants weather.",
        "call_0",
    ),
    ("qwen3-xml", "Qwen/Qwen3-Coder-30B-A3B-Instruct", "b2cff646eb4bb1d68355c01b18ae02e7cf42d120", "", "call_0"),
    ("glm", "zai-org/GLM-4.5-Air", "a24ceef6ce4f3536971efe9b778bdaa1bab18daa", "User wants weather.", "call_0"),
    ("deepseek", "deepseek-ai/DeepSeek-V3.1", "c0781d039fb7a1ba2abc4add0bdc293e92d2b8db", "", "call_0"),
    ("kimi", "moonshotai/Kimi-K2-Instruct", "fd1984e2b7a3350dbf7305fe73a4ede25c14de50", "", "functions.get_weather:0"),
    ("gemma4", "google/gemma-4-E2B-it", "3e22461f65e89153144f8adb70e3b8c2cc9845a7", "User wants weather.", "call_0"),
]

# Renders one assistant tool-call turn through the model's official chat
# template and prints {"fixture": ..., "family": ...}. Runs in the engine venv.
RENDER = r"""
import json, sys
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from transformers.utils import logging

logging.set_verbosity_error()
FAMILY, REPO, REV, REASONING, CALL_ID = sys.argv[1:6]
tok = AutoTokenizer.from_pretrained(REPO, revision=REV, trust_remote_code=True)
with open(hf_hub_download(REPO, "generation_config.json", revision=REV)) as f:
    eos = json.load(f).get("eos_token_id")
stop_ids = set(eos if isinstance(eos, list) else [eos])
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
                   "required": ["city"]}}}]
ARGS = {"city": "Zürich", "unit": "°C"}

def ids(x):
    return list(x["input_ids"] if hasattr(x, "keys") else x)

user = [{"role": "user", "content": "Weather in Zürich?"}]
last = None
for args in (ARGS, json.dumps(ARGS, ensure_ascii=False)):  # templates take dict or JSON-string arguments
    asst = {"role": "assistant", "content": "", "tool_calls": [
        {"id": CALL_ID, "type": "function", "function": {"name": "get_weather", "arguments": args}}]}
    if REASONING:
        asst["reasoning_content"] = REASONING
    try:
        p0 = ids(tok.apply_chat_template(user, tools=TOOLS, tokenize=True, add_generation_prompt=False))
        p1 = ids(tok.apply_chat_template(user, tools=TOOLS, tokenize=True, add_generation_prompt=True))
        full = ids(tok.apply_chat_template([*user, asst], tools=TOOLS, tokenize=True))
        break
    except Exception as e:
        last = e
else:
    raise SystemExit(f"cannot render: {last}")
assert full[: len(p1)] == p1, "prompt is not a prefix of the rendered conversation"
out = full[len(p1):]
out = out[: next((i for i, t in enumerate(out) if t in stop_ids), len(out))]
gen = tok.decode(p1[len(p0):], skip_special_tokens=False) if p1[: len(p0)] == p0 else None
rec = {
    "id": f"{FAMILY}/sglang-test-render", "family": FAMILY, "models": [REPO], "spec_version": "0.1",
    "provenance": {"kind": "template_render", "source_url": f"https://huggingface.co/{REPO}/tree/{REV}",
                   "revision": REV, "license": "see model card", "generator": "tests/adapters/test_sglang.py"},
    "tools": TOOLS, "raw_output": tok.decode(out, skip_special_tokens=False), "output_token_ids": out,
    "tokenizer": {"repo": REPO, "revision": REV, "mode": "hf"},
    "expected": {"content": None, "reasoning_content": REASONING or None,
                 "tool_calls": [{"name": "get_weather", "arguments": ARGS}]},
    "tags": ["unicode"],
}
fam = {"slug": FAMILY, "name": FAMILY, "spec_version": "0.1", "has_reasoning": bool(REASONING), "markers": [],
       "reference_models": [{"repo": REPO, "revision": REV, "default_generation_prompt": gen,
                             "stop_tokens": [tok.convert_ids_to_tokens(i) for i in sorted(stop_ids) if i is not None]}],
       "format_notes": "n/a"}
print(json.dumps({"fixture": rec, "family": fam}, ensure_ascii=False))
"""


@pytest.mark.engine("sglang")
@pytest.mark.parametrize(
    ("family", "repo", "rev", "reasoning", "call_id"), RENDER_CASES, ids=[c[1] for c in RENDER_CASES]
)
def test_engine_template_renders(
    worker: Worker, engine_python_for: Path, family: str, repo: str, rev: str, reasoning: str, call_id: str
) -> None:
    gen = subprocess.run(
        [str(engine_python_for), "-W", "ignore", "-c", RENDER, family, repo, rev, reasoning, call_id],
        env=ENGINE_ENV,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if gen.returncode != 0:
        pytest.skip(f"cannot render {repo}@{rev[:8]} (offline or gated?): {gen.stderr.strip().splitlines()[-1:]}")
    data = json.loads(gen.stdout.strip().splitlines()[-1])
    fx = data["fixture"]
    r = worker.replay(fx, data["family"])
    assert r["supported"]
    cfg = r["parser_config"]
    assert cfg["tokenizer"]["revision"] == rev
    assert cfg["skip_special_tokens"] is False
    expected = _expected_calls(fx)
    # Canonical renders parse without streaming, and when streamed token by token.
    assert _calls(r["nonstream"]) == expected
    assert _calls(r["streams"]["token"]) == expected
    for sid, res in r["streams"].items():
        assert res["exception"] is None, (sid, res)
    if family == "qwen3-hermes":  # Thinking-2507 pre-fills <think>: SGLang's template detection forces reasoning
        assert cfg["template_force_reasoning"] is True
        assert r["nonstream"]["reasoning_content"].strip() == reasoning


HARMONY_SPEC_OUTPUT = (
    "<|channel|>analysis<|message|>Need to use function get_weather.<|end|><|start|>assistant<|channel|>commentary "
    'to=functions.get_weather <|constrain|>json<|message|>{"location":"San Francisco"}'
)
"""openai/harmony@abd677f docs/format.md line 402, cut before the ``<|call|>`` stop token."""
GPT_OSS_REV = "6cee5e81ee83917806bbde320786a8fb61efebee"


def _harmony_case(stop_tokens: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    fx = {
        "id": "gpt-oss/sglang-test-harmony-spec",
        "family": "gpt-oss",
        "models": ["openai/gpt-oss-20b"],
        "spec_version": "0.1",
        "provenance": {
            "kind": "template_render",
            "source_url": "https://github.com/openai/harmony/blob/abd677f7ac962629c808197caa1feb9e3e95d2b0/docs/format.md#L402",
            "revision": "abd677f7ac962629c808197caa1feb9e3e95d2b0",
            "license": "Apache-2.0",
            "attribution": "Harmony response format spec, 'Receiving tool calls' example",
        },
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Gets the current weather in the provided location.",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ],
        "raw_output": HARMONY_SPEC_OUTPUT,
        "tokenizer": {"repo": "openai/gpt-oss-20b", "revision": GPT_OSS_REV, "mode": "hf"},
        "expected": {
            "content": None,
            "reasoning_content": "Need to use function get_weather.",
            "tool_calls": [{"name": "get_weather", "arguments": {"location": "San Francisco"}}],
        },
        "tags": [],
    }
    fam = {
        "slug": "gpt-oss",
        "name": "gpt-oss",
        "spec_version": "0.1",
        "has_reasoning": True,
        "markers": [],
        "reference_models": [
            {
                "repo": "openai/gpt-oss-20b",
                "revision": GPT_OSS_REV,
                "default_generation_prompt": "<|start|>assistant",
                "stop_tokens": stop_tokens,
            }
        ],
        "format_notes": "n/a",
    }
    return fx, fam


@pytest.mark.engine("sglang")
def test_engine_gpt_oss_keeps_call_stop_token(worker: Worker) -> None:
    fx, fam = _harmony_case(["<|return|>", "<|call|>"])
    r = worker.replay(fx, fam)
    stop = r["parser_config"]["stop"]
    assert stop["token"] == "<|call|>" and stop["appended"] and stop["kept_by_engine"]
    expected = _expected_calls(fx)
    assert _calls(r["nonstream"]) == expected
    assert r["nonstream"]["reasoning_content"] == "Need to use function get_weather."
    for sid in STRATEGIES:
        assert _calls(r["streams"][sid]) == expected, sid


@pytest.mark.engine("sglang")
def test_engine_gpt_oss_without_call_token_finds_nothing(worker: Worker) -> None:
    """Control: if the output had ended with <|return|> (trimmed by SGLang), the detector extracts no call."""
    fx, fam = _harmony_case(["<|return|>"])
    r = worker.replay(fx, fam)
    assert r["parser_config"]["stop"]["kept_by_engine"] is False
    assert r["nonstream"]["tool_calls"] == []


def _corpus_fixture(fixture_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    fam_dir = ROOT / "fixtures" / fixture_id.split("/")[0]
    for path in sorted(fam_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line)["id"] == fixture_id:
                return json.loads(line), json.loads((fam_dir / "family.json").read_text(encoding="utf-8"))
    raise LookupError(fixture_id)


@pytest.mark.engine("sglang")
def test_engine_truncated_fixture_gets_no_stop_token(worker: Worker) -> None:
    """Output cut by max_tokens never gets a stop token: SGLang sees FINISH_LENGTH instead."""
    fx, fam = _corpus_fixture("gpt-oss/harmony-truncated-in-arguments")
    assert "truncated" in fx["tags"]
    r = worker.replay(fx, fam)
    stop = r["parser_config"]["stop"]
    assert stop["appended"] is False and stop["id"] is None and stop["finish_reason"] == "length"
    for parse in (r["nonstream"], *r["streams"].values()):
        assert "<|call|>" not in (parse["content"] or "")


@pytest.mark.engine("sglang")
@pytest.mark.parametrize(
    ("fixture_id", "spec"), [("deepseek/v4-single-call", "dsv4"), ("deepseek/v41-parallel-two-calls", "dsv41")]
)
def test_engine_deepseek_v4_output_goes_through_recipe(worker: Worker, fixture_id: str, spec: str) -> None:
    """With the deepseekv4/deepseekv41 parsers, 0.5.23 parses output token ids with deepseek-recipe
    (serving_deepseek_recipe.py), not the text detectors."""
    fx, fam = _corpus_fixture(fixture_id)
    r = worker.replay(fx, fam, [*STRATEGIES, "char:0"])
    assert r["supported"]
    cfg = r["parser_config"]
    assert cfg["chat_encoding_spec"] == spec
    assert cfg["output_adapter"] == "DeepSeekRecipeChatEncodingAdapter"
    expected = _expected_calls(fx)
    assert _calls(r["nonstream"]) == expected
    for sid in ("one", "token", "rand:1:8"):
        assert _calls(r["streams"][sid]) == expected, sid
    # The server never sees text deltas for these models: no char:<seed> result is made up.
    assert "token ids" in (r["streams"]["char:0"]["exception"] or "")


@pytest.mark.engine("sglang")
def test_engine_text_parsers_have_no_output_adapter(worker: Worker) -> None:
    fx, fam = _corpus_fixture("deepseek/v32-single-call")
    r = worker.replay(fx, fam)
    assert r["parser_config"]["chat_encoding_spec"] == "dsv32"
    assert r["parser_config"]["output_adapter"] is None


def _corpus() -> list[Any]:
    files = sorted((ROOT / "fixtures").glob("*/*.jsonl"))
    if not files:
        return [pytest.param(Path(), marks=pytest.mark.skip(reason="no fixture corpus yet"))]
    return [pytest.param(p, id=f"{p.parent.name}/{p.name}") for p in files]


@pytest.mark.engine("sglang")
@pytest.mark.parametrize("path", _corpus())
def test_engine_corpus_replays(worker: Worker, path: Path) -> None:
    """The first records of each corpus file replay without harness errors."""
    fam_file = path.parent / "family.json"
    fam = json.loads(fam_file.read_text(encoding="utf-8")) if fam_file.is_file() else None
    for _, rec in list(read_jsonl(path))[:3]:
        r = worker.replay(rec, fam)
        if not r["supported"]:
            assert r["reason"]
            continue
        assert r["parser_config"]["engine"] == "sglang"
        assert set(r["streams"]) | set(r["skipped"]) == set(STRATEGIES)

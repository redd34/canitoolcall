"""Tests for the mlx-lm adapter.

* Fast unit tests run in the main dev env and never import mlx or mlx_lm.
* ``@pytest.mark.engine("mlx_lm")`` tests drive the real worker inside
  ``.venvs/mlx_lm`` (built by ``scripts/engines/mlx_lm.sh``) and are skipped when
  that env is missing (always on hosts where mlx cannot be imported). They replay
  the core Qwen3 sample and the corpus records of every family, reading
  tokenizers from the HF cache only (export ``HF_HUB_OFFLINE=0`` once to warm it).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from canitoolcall.adapters import AdapterUnavailable, adapter_class
from canitoolcall.adapters.mlx_lm import (
    FAMILY_PARSERS,
    MlxLmAdapter,
    exception_text,
    message_result,
    sse_result,
)
from canitoolcall.chunking import DEFAULT_STRATEGIES
from canitoolcall.fixtures import load_families, read_jsonl, repo_root
from canitoolcall.results import ParsedToolCall, ParseResult

ROOT = repo_root() or Path(__file__).resolve().parents[2]
ENGINE_ENV = {
    **os.environ,
    "PYTHONPATH": str(ROOT / "src"),
    "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
}
STRATEGIES = [s.id for s in DEFAULT_STRATEGIES]
SLUGS = sorted(p.parent.name for p in (ROOT / "fixtures").glob("*/family.json"))


# --------------------------------------------------------------------------
# fast unit tests (no engine)
# --------------------------------------------------------------------------


def test_registered_and_pinned() -> None:
    cls = adapter_class("mlx_lm")
    assert cls is MlxLmAdapter
    assert cls.pinned_version == "0.32.2"
    # mlx_lm.server answers one Response per generated token.
    assert cls.tokens_per_step == "one"


def test_parser_table_only_names_mlx_lm_parsers() -> None:
    known = {"json_tools", "qwen3_coder", "glm47", "gemma4", "kimi_k2", "kimi_k3", "mistral"}
    assert set(FAMILY_PARSERS) <= set(SLUGS)
    assert {p for models in FAMILY_PARSERS.values() for p in models.values()} <= known


@pytest.mark.parametrize(
    ("family", "model", "parser"),
    [
        ("qwen3-hermes", "Qwen/Qwen3-0.6B", "json_tools"),
        ("qwen3-xml", "Qwen/Qwen3-Coder-30B-A3B-Instruct", "qwen3_coder"),
        ("glm", "zai-org/GLM-4.5", "glm47"),
        ("gemma4", "google/gemma-4-31B-it", "gemma4"),
        ("kimi", "moonshotai/Kimi-K2-Instruct-0905", "kimi_k2"),
        ("kimi", "moonshotai/Kimi-K3", "kimi_k3"),
        ("mistral", "mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
    ],
)
def test_supports_mapped_models(family: str, model: str, parser: str) -> None:
    assert MlxLmAdapter().supports(family, model)
    assert FAMILY_PARSERS[family][model] == parser


@pytest.mark.parametrize(
    ("family", "model", "why"),
    [
        ("gpt-oss", "openai/gpt-oss-20b", "no tool parser"),
        ("deepseek", "deepseek-ai/DeepSeek-V3.1", "no tool parser"),
        ("llama", "meta-llama/Llama-3.3-70B-Instruct", "no tool parser"),
        ("mistral", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "infers no tool parser"),
        ("qwen3-hermes", "Qwen/Qwen3-235B-A22B", "not a reference model"),
    ],
)
def test_supports_declines_honestly(family: str, model: str, why: str) -> None:
    sup = MlxLmAdapter().supports(family, model)
    assert not sup
    assert sup.reason is not None and why in sup.reason


def test_message_result_reads_the_mlx_lm_message() -> None:
    # mlx_lm.server names the reasoning field "reasoning" and serializes arguments itself.
    msg = {
        "role": "assistant",
        "content": None,
        "reasoning": "think",
        "tool_calls": [{"function": {"name": "f", "arguments": '{"a": 1}'}, "type": "function", "id": "x"}],
    }
    assert message_result(msg) == ParseResult(
        content=None, reasoning_content="think", tool_calls=(ParsedToolCall("f", '{"a": 1}'),)
    )


def test_sse_result_accumulates_openai_client_style() -> None:
    def chunk(delta: dict[str, Any]) -> str:
        return "data: " + json.dumps({"choices": [{"index": 0, "delta": {"role": "assistant", **delta}}]}) + "\n\n"

    body = (
        ": keepalive 1/2\n\n"
        + chunk({"reasoning": "a"})
        + chunk({"reasoning": "b", "content": "hi"})
        + chunk({"tool_calls": [{"index": 0, "function": {"name": "f", "arguments": "{}"}}]})
        + chunk({"tool_calls": [{"index": 1, "function": {"name": "g", "arguments": '{"x": 2}'}}]})
        + "data: [DONE]\n\n"
    )
    assert sse_result(body.encode()) == ParseResult(
        content="hi",
        reasoning_content="ab",
        tool_calls=(ParsedToolCall("f", "{}"), ParsedToolCall("g", '{"x": 2}')),
    )


def test_exception_text() -> None:
    assert exception_text(ValueError("bad")) == "ValueError: bad"


@pytest.mark.skipif(importlib.util.find_spec("mlx_lm") is not None, reason="mlx_lm is importable here")
def test_without_mlx_the_engine_is_unavailable() -> None:
    # Hosts where mlx cannot be installed or imported: the worker answers "unavailable", never a crash.
    with pytest.raises(AdapterUnavailable, match=r"scripts/engines/mlx_lm\.sh"):
        MlxLmAdapter().version()


def test_setup_script_pins_and_reports_missing_mlx() -> None:
    script = (ROOT / "scripts" / "engines" / "mlx_lm.sh").read_text(encoding="utf-8")
    assert 'MLX_LM_VERSION="0.32.2"' in script
    assert "mlx-lm[cpu]" in script  # Linux: mlx-lm requires mlx only on Darwin
    assert "exit 3" in script and "raise SystemExit(3)" in script  # nightly: 3 = cannot run on this platform


# --------------------------------------------------------------------------
# engine tests (real mlx-lm 0.32.2 in .venvs/mlx_lm)
# --------------------------------------------------------------------------


class Worker:
    """The real JSON-lines worker, running in the mlx_lm venv."""

    def __init__(self, python: Path) -> None:
        self.proc = subprocess.Popen(
            [str(python), "-m", "canitoolcall.adapters.worker", "mlx_lm"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=ENGINE_ENV,
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
            self.proc.wait(timeout=60)


def _decoded(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "content": (result["content"] or "").strip() or None,
        "reasoning_content": (result["reasoning_content"] or "").strip() or None,
        "tool_calls": [{"name": t["name"], "arguments": json.loads(t["arguments_raw"])} for t in result["tool_calls"]],
        "exception": result["exception"],
    }


@pytest.mark.engine("mlx_lm")
def test_worker_hello(engine_python_for: Path) -> None:
    w = Worker(engine_python_for)
    try:
        hello = w.ask({"op": "hello"})
    finally:
        w.close()
    assert hello["ok"], hello
    assert hello["engine"] == "mlx_lm" and hello["version"] == "0.32.2"
    assert hello["tokens_per_step"] == "one"
    assert hello["details"]["mlx"]


@pytest.mark.engine("mlx_lm")
def test_core_sample_replays_correctly(engine_python_for: Path, sample_fixtures_dir: Path) -> None:
    ((_, rec),) = list(read_jsonl(sample_fixtures_dir / "qwen3-hermes" / "sample.jsonl"))
    fam = load_families(sample_fixtures_dir)["qwen3-hermes"].to_dict()
    w = Worker(engine_python_for)
    try:
        reply = w.ask({"op": "replay", "fixture": rec, "family": fam, "strategies": STRATEGIES})
    finally:
        w.close()
    assert reply["ok"], reply.get("error")
    assert reply["supported"], reply
    cfg = reply["parser_config"]
    assert cfg["tool_parser"] == "json_tools"
    assert cfg["think"] == ["<think>", "</think>"]
    assert len(cfg["chat_template_sha256"]) == 64
    want = {**rec["expected"], "exception": None}
    assert _decoded(reply["nonstream"]) == want
    assert set(reply["streams"]) == set(STRATEGIES), reply.get("skipped")
    for sid, got in reply["streams"].items():
        assert _decoded(got) == want, (sid, got)


@pytest.mark.engine("mlx_lm")
def test_unsupported_family_is_reported(engine_python_for: Path) -> None:
    path = next(iter(sorted((ROOT / "fixtures" / "gpt-oss").glob("*.jsonl"))))
    (_, rec), *_ = list(read_jsonl(path))
    fam = json.loads((ROOT / "fixtures" / "gpt-oss" / "family.json").read_text(encoding="utf-8"))
    w = Worker(engine_python_for)
    try:
        reply = w.ask({"op": "replay", "fixture": rec, "family": fam, "strategies": ["one"]})
    finally:
        w.close()
    assert reply["ok"] and reply["supported"] is False
    assert "no tool parser" in reply["reason"]


@pytest.mark.engine("mlx_lm")
@pytest.mark.parametrize("family", sorted(FAMILY_PARSERS))
def test_family_corpus_replays_without_harness_errors(engine_python_for: Path, family: str) -> None:
    d = ROOT / "fixtures" / family
    fam = json.loads((d / "family.json").read_text(encoding="utf-8"))
    records = [rec for p in sorted(d.glob("*.jsonl")) for _, rec in read_jsonl(p)]
    w = Worker(engine_python_for)
    supported = 0
    try:
        for rec in records:
            reply = w.ask({"op": "replay", "fixture": rec, "family": fam, "strategies": STRATEGIES})
            assert reply["ok"], (rec["id"], reply.get("error"))
            if reply["supported"]:
                supported += 1
                assert reply["parser_config"]["tool_parser"] == FAMILY_PARSERS[family][rec["models"][0]]
    finally:
        w.close()
    assert supported, f"no {family} fixture was supported"

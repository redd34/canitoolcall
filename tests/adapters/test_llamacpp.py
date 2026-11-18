"""Tests for the llama.cpp adapter.

* The fast tests need no build: family support, converter-error handling,
  template-source selection, OpenAI-style stream accumulation, the harness
  process wrapper, and the adapter's request/response mapping, all against a
  fake harness script that speaks the same JSON-lines protocol.
* The ``@pytest.mark.engine("llamacpp")`` tests drive the real compiled harness
  (``scripts/engines/llamacpp.sh``) and the real worker in ``.venvs/llamacpp``.
  They are skipped when the venv is missing, and also when the harness binary
  or a needed vocab-only GGUF (``scripts/engines/gguf_vocab.sh --all``) is.
"""

from __future__ import annotations

import dataclasses
import json
import os
import stat
import subprocess
import sys
import textwrap
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from canitoolcall.adapters.base import AdapterUnavailable, ReplayInput
from canitoolcall.adapters.llamacpp import (
    FAMILIES,
    PROMPT_MESSAGES,
    Harness,
    HarnessError,
    LlamaCppAdapter,
    accumulate,
    gguf_stem,
    llamacpp_template_file,
    message_result,
)
from canitoolcall.chunking import ChunkStrategy, split
from canitoolcall.fixtures import Fixture, TokenizerPin, load_families, load_fixtures, read_jsonl, repo_root
from canitoolcall.results import ParsedToolCall, ParseResult

FAMILY_SLUGS = (
    "qwen3-hermes",
    "qwen3-xml",
    "gpt-oss",
    "deepseek",
    "kimi",
    "glm",
    "llama",
    "mistral",
    "gemma4",
    "seed-oss",
)
PIN = "a25c9865fe03c954c93fd755b5d79ae86ba99750"

# --------------------------------------------------------------------------- fast (no build)


def test_families_cover_every_slug() -> None:
    assert set(FAMILIES) == set(FAMILY_SLUGS)
    assert LlamaCppAdapter.pinned_version == PIN
    assert LlamaCppAdapter.supports_text_deltas


def test_supports(tmp_path: Path) -> None:
    a = LlamaCppAdapter(harness=tmp_path / "missing", gguf_dir=tmp_path)
    assert a.supports("qwen3-hermes", "Qwen/Qwen3-0.6B")
    sup = a.supports("falcon", "tiiuae/falcon-7b")
    assert not sup and "falcon" in (sup.reason or "")

    # llama.cpp's converter rejecting the model -> unsupported, with its own words
    (tmp_path / "org--new.vocab.error.json").write_text(
        json.dumps(
            {"stage": "convert", "error": "Traceback...\nERROR:hf-to-gguf:Model NewForCausalLM is not supported"}
        )
    )
    sup = a.supports("deepseek", "org/new")
    assert not sup and "NewForCausalLM is not supported" in (sup.reason or "")

    # a download problem is a setup problem, never "unsupported"
    (tmp_path / "org--gated.vocab.error.json").write_text(json.dumps({"stage": "download", "error": "401"}))
    assert a.supports("llama", "org/gated")
    l4 = a.supports("llama", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
    assert not l4 and l4.reason and "Llama 4 chat template" in l4.reason


def test_gguf_stem_and_template_copies(tmp_path: Path) -> None:
    assert gguf_stem("Qwen/Qwen3-0.6B") == "Qwen--Qwen3-0.6B"
    tdir = tmp_path / "models" / "templates"
    tdir.mkdir(parents=True)
    (tdir / "deepseek-ai-DeepSeek-V3.1.jinja").write_text("a")
    (tdir / "Kimi-K3.jinja").write_text("b")
    (tdir / "openai-gpt-oss-120b.jinja").write_text("c")
    assert llamacpp_template_file(tmp_path, "deepseek-ai/DeepSeek-V3.1") == tdir / "deepseek-ai-DeepSeek-V3.1.jinja"
    assert llamacpp_template_file(tmp_path, "moonshotai/Kimi-K3") == tdir / "Kimi-K3.jinja"
    # never a sibling model's template
    assert llamacpp_template_file(tmp_path, "openai/gpt-oss-20b") is None


def test_accumulate_openai_client_style() -> None:
    deltas: list[dict[str, Any]] = [
        {"reasoning_content": "thin"},
        {"reasoning_content": "king"},
        {"content": "Hi"},
        {"index": 0, "name": "get_", "id": "a", "arguments": ""},
        {"index": 0, "name": "weather", "id": "a", "arguments": '{"city": '},  # name fragments concatenate
        {"index": 0, "name": "", "id": "", "arguments": '"Paris"}'},
        {"index": 1, "name": "search", "id": "b", "arguments": "{}"},
    ]
    r = accumulate(deltas)
    assert r == ParseResult(
        content="Hi",
        reasoning_content="thinking",
        tool_calls=(ParsedToolCall("get_weather", '{"city": "Paris"}'), ParsedToolCall("search", "{}")),
    )
    assert accumulate([]) == ParseResult()
    assert accumulate([{"content": "x"}], "LlamaCppError: boom").exception == "LlamaCppError: boom"


def test_message_result_is_the_openai_response() -> None:
    r = message_result(
        {"content": "", "reasoning_content": "", "tool_calls": [{"name": "f", "arguments": "{}", "id": "x"}]}
    )
    # to_json_oaicompat: content is always sent (""), empty reasoning is omitted
    assert r == ParseResult(content="", reasoning_content=None, tool_calls=(ParsedToolCall("f", "{}"),))


def test_thinking_kwargs() -> None:
    raw = ReplayInput("f", "qwen3-hermes", "Qwen/Qwen3-0.6B", "x")
    assert LlamaCppAdapter._kwargs(raw) == {}
    off = ReplayInput("f", "kimi", "moonshotai/Kimi-K3", "x", thinking=False)
    assert LlamaCppAdapter._kwargs(off) == {"enable_thinking": False, "thinking": False}


# A stand-in for the compiled harness: same JSON-lines protocol, canned replies,
# and every request echoed back so the tests can check what the adapter sent.
FAKE_HARNESS = textwrap.dedent(
    """\
    #!{python}
    import json, sys
    replies = json.load(open({replies!r}))
    for line in sys.stdin:
        req = json.loads(line)
        op = req["op"]
        if op == "crash":
            sys.exit(3)
        if op == "shutdown":
            print(json.dumps({{"ok": True}}), flush=True)
            break
        reply = dict(replies.get(op, {{"ok": False, "error": "unknown op " + op}}))
        if op == "replay" and req.get("return_template"):
            reply = dict(replies["template_probe"])
        reply["echo"] = req
        with open({log!r}, "a") as f:
            f.write(json.dumps(req) + "\\n")
        print(json.dumps(reply), flush=True)
    """
)


# The fake harness is a `#!python` script that the adapter launches directly,
# which Windows cannot do (no shebangs, no exec bit); the real harness is a POSIX build.
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="runs a #!-script directly; POSIX only")


def _fake_harness(tmp_path: Path, replies: dict[str, Any]) -> tuple[Path, Path]:
    replies_path = tmp_path / "replies.json"
    replies_path.write_text(json.dumps(replies))
    log = tmp_path / "requests.jsonl"
    script = tmp_path / "fake-harness"
    script.write_text(FAKE_HARNESS.format(python=sys.executable, replies=str(replies_path), log=str(log)))
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script, log


def _requests(log: Path, op: str = "replay") -> list[dict[str, Any]]:
    reqs = [json.loads(line) for line in log.read_text().splitlines()]
    return [r for r in reqs if r["op"] == op and not r.get("return_template")]


HELLO = {"ok": True, "protocol": 1, "commit": "a25c986", "build_info": "b1-a25c986"}


@posix_only
def test_harness_process(tmp_path: Path) -> None:
    script, _ = _fake_harness(tmp_path, {"hello": HELLO})
    h = Harness(script)
    assert h.request({"op": "hello"})["commit"] == "a25c986"
    with pytest.raises(HarnessError, match="unknown op nope"):
        h.request({"op": "nope"})
    with pytest.raises(HarnessError, match="exited with code 3"):
        h.request({"op": "crash"})
    assert h.request({"op": "hello"})["ok"]  # restarted after the crash
    h.close()
    with pytest.raises(AdapterUnavailable, match="not built"):
        Harness(tmp_path / "missing")


@posix_only
def test_version_is_checked_against_the_pin(tmp_path: Path) -> None:
    script, _ = _fake_harness(tmp_path, {"hello": HELLO})
    a = LlamaCppAdapter(harness=script, gguf_dir=tmp_path)
    assert a.version() == PIN[:8]
    assert a.commit() == PIN
    assert a.engine_details()["build_info"] == "b1-a25c986"
    a.close()

    other = tmp_path / "other"
    other.mkdir()
    script2, _ = _fake_harness(other, {"hello": {**HELLO, "commit": "deadbee"}})
    with pytest.raises(AdapterUnavailable, match="expected a25c9865"):
        LlamaCppAdapter(harness=script2, gguf_dir=tmp_path).version()


def _setup_engine_dirs(tmp_path: Path) -> tuple[Path, Path]:
    gguf_dir = tmp_path / "gguf"
    gguf_dir.mkdir()
    (gguf_dir / "org--m.vocab.gguf").write_bytes(b"GGUF")
    (gguf_dir / "org--m.vocab.json").write_text(json.dumps({"repo": "org/m", "revision": "r1", "sha256": "s"}))
    src = tmp_path / "llama.cpp"
    (src / "models" / "templates").mkdir(parents=True)
    (src / "models" / "templates" / "org-m.jinja").write_text("{{ curated }}")
    return gguf_dir, src


REPLAY_OK = {
    "ok": True,
    "config": {"format": "peg-native", "generation_prompt": "<|a|>", "preserved_tokens": ["<t>"], "end_token": "</s>"},
    "text": "raw",
    "nonstream": {"msg": {"content": "", "reasoning_content": "r", "tool_calls": [{"name": "f", "arguments": "{}"}]}},
    "streams": [{"deltas": [{"reasoning_content": "r"}, {"index": 0, "name": "f", "arguments": "{}"}], "msg": {}}],
}


@posix_only
def test_parse_mapping_and_template_choice(tmp_path: Path) -> None:
    gguf_dir, src = _setup_engine_dirs(tmp_path)
    script, log = _fake_harness(
        tmp_path,
        {
            "hello": HELLO,
            "replay": REPLAY_OK,
            "template_probe": {"ok": True, "config": {"gguf_has_template": False, "template_source": "chatml"}},
        },
    )
    a = LlamaCppAdapter(harness=script, gguf_dir=gguf_dir, source_dir=src)
    raw = ReplayInput(
        "fam/x",
        "qwen3-hermes",
        "org/m",
        "raw",
        token_ids=(1, 2, 3),
        tokenizer=TokenizerPin("org/m", "r1"),
        thinking=True,
        stop_tokens=("</s>",),
    )
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]

    assert a.units(raw) == [1, 2, 3]
    stream = a.parse_stream(raw, [[1], [2, 3]], tools)
    assert stream == ParseResult(content=None, reasoning_content="r", tool_calls=(ParsedToolCall("f", "{}"),))
    ns = a.parse(raw, tools)
    assert ns == ParseResult(content="", reasoning_content="r", tool_calls=(ParsedToolCall("f", "{}"),))

    reqs = _requests(log)
    first = reqs[0]
    # the GGUF has no chat template -> llama.cpp's curated copy is passed as the override
    assert first["template"] == "{{ curated }}"
    assert first["ids"] == [1, 2, 3]
    assert first["streams"] == [[1, 2]]
    assert first["end_tokens"] == ["</s>"]
    assert first["reasoning_format"] == "deepseek"
    assert first["chat_template_kwargs"] == {"enable_thinking": True, "thinking": True}
    assert first["messages"] == list(PROMPT_MESSAGES)
    assert first["tools"] == tools

    cfg = a.parser_config(raw)
    assert cfg["template_source"] == "llamacpp"
    assert cfg["template_source_requested"] == "gguf"
    assert "has no tokenizer.chat_template" in cfg["template_source_reason"]
    assert cfg["template_path"] == "models/templates/org-m.jinja"
    assert cfg["format"] == "peg-native"
    assert cfg["detokenized_matches_raw_output"] is True
    assert cfg["tokenizer"] == "org/m@r1"
    assert cfg["template_alternatives"]["gguf"]["available"] is False

    # A generation cut by max_tokens never ended with a stop token: none is fed.
    truncated = dataclasses.replace(raw, fixture_id="fam/cut", truncated=True)
    a.parse(truncated, tools)
    a.parse_stream(truncated, [[1, 2, 3]], tools)
    cut_reqs = _requests(log)[len(reqs) :]
    assert cut_reqs and all(r["end_tokens"] == [] for r in cut_reqs)
    a.close()


@posix_only
def test_engine_errors_are_outcomes(tmp_path: Path) -> None:
    gguf_dir, src = _setup_engine_dirs(tmp_path)
    script, log = _fake_harness(
        tmp_path,
        {
            "hello": HELLO,
            "replay": {"ok": True, "config": {"format": "peg-native"}, "engine_error": "chat template: boom"},
            "template_probe": {"ok": True, "config": {"gguf_has_template": True, "template_source": "{{ hf }}"}},
        },
    )
    a = LlamaCppAdapter(harness=script, gguf_dir=gguf_dir, source_dir=src)
    raw = ReplayInput("fam/y", "glm", "org/m", "raw", token_ids=(5,))
    assert a.parse(raw, []).exception == "LlamaCppError: chat template: boom"
    assert a.parse_stream(raw, [[5]], []).exception == "LlamaCppError: chat template: boom"
    assert a.parse_stream_text(raw, ["r", "aw"], []).exception == "LlamaCppError: chat template: boom"
    text_req = _requests(log)[-1]
    assert text_req["streams"] == [{"text": ["r", "aw"]}]
    assert text_req["template"] is None  # the GGUF's own template (primary source)
    cfg = a.parser_config(raw)
    assert cfg["template_source"] == "gguf" and cfg["template_source_reason"] is None
    assert cfg["engine_error"] == "chat template: boom"
    assert cfg["template_alternatives"]["llamacpp"]["available"] is True
    a.close()


def test_missing_vocab_gguf_is_a_harness_problem(tmp_path: Path) -> None:
    script, _ = _fake_harness(tmp_path, {"hello": HELLO})
    a = LlamaCppAdapter(harness=script, gguf_dir=tmp_path)
    with pytest.raises(AdapterUnavailable, match=r"gguf_vocab\.sh"):
        a.parse(ReplayInput("f", "glm", "org/none", "x", token_ids=(1,)), [])


# --------------------------------------------------------------------------- engine (real harness)

ROOT = repo_root() or Path(__file__).resolve().parents[2]
DEFAULT = LlamaCppAdapter()
needs_build = pytest.mark.skipif(
    not DEFAULT.harness_path.is_file(), reason="llama.cpp harness not built (run scripts/engines/llamacpp.sh)"
)


def _vocab_or_skip(repo: str) -> None:
    if not (DEFAULT.gguf_dir / f"{gguf_stem(repo)}.vocab.gguf").is_file():
        pytest.skip(f"no vocab-only GGUF for {repo} (run scripts/engines/gguf_vocab.sh --all)")


def _run_worker(python: Path, msgs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    proc = subprocess.run(
        [str(python), "-m", "canitoolcall.adapters.worker", "llamacpp"],
        input="".join(json.dumps(m) + "\n" for m in msgs),
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return [json.loads(line) for line in proc.stdout.splitlines()]


def _calls(result: dict[str, Any]) -> list[tuple[str, Any]]:
    return [(t["name"], json.loads(t["arguments_raw"])) for t in result["tool_calls"]]


@needs_build
@pytest.mark.engine("llamacpp")
def test_worker_replays_core_sample(engine_python_for: Path, sample_fixtures_dir: Path) -> None:
    _vocab_or_skip("Qwen/Qwen3-0.6B")
    ((_, rec),) = list(read_jsonl(sample_fixtures_dir / "qwen3-hermes" / "sample.jsonl"))
    fam = load_families(sample_fixtures_dir)["qwen3-hermes"].to_dict()
    strategies = ["one", "special", "token", "rand:1:8", "rand:2:3", "char:0"]
    hello, rep, bye = _run_worker(
        engine_python_for,
        [
            {"op": "hello"},
            {"op": "replay", "fixture": rec, "family": fam, "strategies": strategies},
            {"op": "shutdown"},
        ],
    )
    assert hello["ok"] and hello["version"] == PIN[:8] and hello["commit"] == PIN
    assert rep["ok"], rep.get("error")
    assert rep["supported"]
    expected = [(c["name"], c["arguments"]) for c in rec["expected"]["tool_calls"]]
    assert _calls(rep["nonstream"]) == expected
    # llama.cpp keeps the newline before </think> (a soft difference, not a bug in the replay)
    assert rep["nonstream"]["reasoning_content"].strip() == rec["expected"]["reasoning_content"]
    assert set(rep["streams"]) == set(strategies)
    for sid, stream in rep["streams"].items():
        assert stream["exception"] is None, sid
        assert _calls(stream) == expected, sid
        assert stream["reasoning_content"] == rep["nonstream"]["reasoning_content"], sid
    cfg = rep["parser_config"]
    assert cfg["format"] == "peg-native"
    assert cfg["template_source"] == "gguf"
    # the GGUF's embedded template is exactly the fixture's HF template at the same revision
    assert cfg["template_sha256"] == rec["provenance"]["template_sha256"]
    assert cfg["generation_prompt"] == fam["reference_models"][0]["default_generation_prompt"]
    assert cfg["detokenized_matches_raw_output"] is True
    assert cfg["preserved_tokens"] == ["<think>", "</think>", "<tool_call>", "</tool_call>"]
    assert bye == {"ok": True}


def _one_fixture(family: str) -> Fixture:
    try:
        fixtures = [f for f in load_fixtures(families=[family]) if f.output_token_ids and f.expected is not None]
    except FileNotFoundError:
        fixtures = []
    for fx in fixtures:
        repo = fx.tokenizer.repo if fx.tokenizer else fx.reference_model
        if (DEFAULT.gguf_dir / f"{gguf_stem(repo)}.vocab.gguf").is_file() and DEFAULT.supports(
            family, fx.reference_model
        ):
            return fx
    pytest.skip(f"no replayable {family} fixture with a built vocab GGUF yet")


@needs_build
@pytest.mark.engine("llamacpp")
@pytest.mark.parametrize("family", FAMILY_SLUGS)
def test_worker_replays_one_fixture_per_family(engine_python_for: Path, family: str) -> None:
    fx = _one_fixture(family)
    fam = load_families().get(family)
    (rep,) = _run_worker(
        engine_python_for,
        [
            {
                "op": "replay",
                "fixture": fx.to_dict(),
                "family": fam.to_dict() if fam else None,
                "strategies": ["one", "token", "rand:3:8"],
            }
        ],
    )
    assert rep["ok"], rep.get("error")
    assert rep["supported"]
    cfg = rep["parser_config"]
    assert cfg["format"], cfg
    assert cfg["template_sha256"] and cfg["vocab_gguf_sha256"]
    # the engine detokenizes the fixture's ids back to its raw output
    assert cfg["detokenized_matches_raw_output"] is True
    assert set(rep["streams"]) == {"one", "token", "rand:3:8"}
    for result in (rep["nonstream"], *rep["streams"].values()):
        ParseResult.from_dict(result)  # well-formed parse results (engine exceptions are outcomes)


def _find(fixture_id: str) -> tuple[Fixture, ReplayInput]:
    family = fixture_id.split("/")[0]
    fams = load_families()
    for fx in load_fixtures(families=[family]):
        if fx.id == fixture_id:
            return fx, ReplayInput.from_fixture(fx, fams.get(family))
    pytest.skip(f"fixture {fixture_id} not in the corpus")


@needs_build
@pytest.mark.engine("llamacpp")
def test_gpt_oss_call_through_real_parser() -> None:
    fx, raw = _find("gpt-oss/harmony-single-call")
    _vocab_or_skip("openai/gpt-oss-20b")
    a = LlamaCppAdapter()
    try:
        tools = list(fx.tools)
        assert fx.expected is not None
        expected = [(c.name, dict(c.arguments)) for c in fx.expected.tool_calls]
        ns = a.parse(raw, tools)
        assert [(t.name, t.arguments()) for t in ns.tool_calls] == expected
        assert ns.reasoning_content == fx.expected.reasoning_content
        units = a.units(raw)
        for strat in ("one", "token", "rand:4:8"):
            st = a.parse_stream(raw, split(units, ChunkStrategy.parse(strat)), tools)
            assert [(t.name, t.arguments()) for t in st.tool_calls] == expected, strat
    finally:
        a.close()


@needs_build
@pytest.mark.engine("llamacpp")
def test_deepseek_v31_template_sources_are_both_recorded() -> None:
    """The HF DeepSeek-V3.1 template (embedded in the GGUF) yields malformed
    preserved_tokens and a parser that rejects the model's own call; llama.cpp's
    models/templates copy parses it. Both outcomes are recorded, neither hidden."""
    fx, raw = _find("deepseek/v31-single-call")
    _vocab_or_skip("deepseek-ai/DeepSeek-V3.1")
    a = LlamaCppAdapter()
    try:
        ns = a.parse(raw, list(fx.tools))
        cfg = a.parser_config(raw)
        assert cfg["template_source"] == "gguf"
        assert ns.exception and "does not match" in ns.exception
        alt = cfg["template_alternatives"]["llamacpp"]
        assert alt["available"] and alt["path"] == "models/templates/deepseek-ai-DeepSeek-V3.1.jinja"
        alt_result = ParseResult.from_dict(alt["nonstream"])
        assert alt_result.exception is None
        assert fx.expected is not None
        assert [t.name for t in alt_result.tool_calls] == [c.name for c in fx.expected.tool_calls]
        # the same fixture with the llama.cpp copy as the primary source
        b = LlamaCppAdapter(template_source="llamacpp")
        try:
            assert b.parse(raw, list(fx.tools)) == alt_result
            assert b.parser_config(raw)["template_alternatives"]["gguf"]["nonstream"]["exception"] == ns.exception
        finally:
            b.close()
    finally:
        a.close()


@needs_build
@pytest.mark.engine("llamacpp")
def test_utf8_holdback_and_text_deltas(sample_fixtures_dir: Path) -> None:
    """Per-token streaming splits multi-byte characters (Zürich, café); the server's
    validate_utf8 hold-back must keep the accumulated stream identical. The
    synthetic per-character text path must agree too."""
    _vocab_or_skip("Qwen/Qwen3-0.6B")
    fx = load_fixtures([sample_fixtures_dir / "qwen3-hermes" / "sample.jsonl"])[0]
    raw = ReplayInput.from_fixture(fx, load_families(sample_fixtures_dir)["qwen3-hermes"])
    a = LlamaCppAdapter()
    try:
        tools = list(fx.tools)
        ns = a.parse(raw, tools)
        per_token = a.parse_stream(raw, [[i] for i in a.units(raw)], tools)
        per_char = a.parse_stream_text(raw, list(raw.text), tools)
        for st in (per_token, per_char):
            assert st.tool_calls == ns.tool_calls
            assert st.reasoning_content == ns.reasoning_content
        special = a.special_token_ids(raw)
        assert {151657, 151658, 151667, 151668} <= special  # <tool_call> </tool_call> <think> </think>
        assert 13 not in special  # "."
    finally:
        a.close()

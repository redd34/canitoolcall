"""Replay-metadata contracts. Synthetic controls here are not new corpus fixtures."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from canitoolcall.adapters import load_adapter
from canitoolcall.adapters.base import Adapter, ReplayInput, Support, ToolSpec
from canitoolcall.adapters.llamacpp import TemplateChoice
from canitoolcall.adapters.worker import handle, replay
from canitoolcall.checks import case_status
from canitoolcall.chunking import ChunkStrategy
from canitoolcall.fixtures import (
    Finish,
    Fixture,
    default_fixtures_dir,
    load_families,
    load_fixtures,
    read_jsonl,
    validate,
)
from canitoolcall.results import CheckResult, ParseResult, Status

ENGINES = ("vllm", "sglang", "llamacpp", "ollama", "transformers")
MARKED = {"gpt-oss/vllm-sequential-calls", "gpt-oss/vllm-call-then-final"}


@pytest.fixture
def sample(sample_fixtures_dir: Path) -> dict[str, Any]:
    ((_, record),) = list(read_jsonl(sample_fixtures_dir / "qwen3-hermes" / "sample.jsonl"))
    return record


class Spy(Adapter):
    """Cheap unit-test adapter; any expensive entry records the received raw input."""

    name = "spy"
    pinned_version = "0"

    def __init__(self, supported: Support | None = None) -> None:
        self.support = supported if supported is not None else Support(True)
        self.seen: list[tuple[str, ReplayInput]] = []

    def version(self) -> str:
        return "0"

    def supports(self, family: str, model: str) -> Support:
        return self.support

    def units(self, raw: ReplayInput) -> list[int]:
        self.seen.append(("units", raw))
        return [1, 2]

    def parse(self, raw: ReplayInput, tools: Sequence[ToolSpec]) -> ParseResult:
        self.seen.append(("parse", raw))
        return ParseResult(content=raw.text)

    def parse_stream(self, raw: ReplayInput, chunks: Sequence[Sequence[int]], tools: Sequence[ToolSpec]) -> ParseResult:
        self.seen.append(("stream", raw))
        return ParseResult(content=raw.text)

    def parser_config(self, raw: ReplayInput) -> dict[str, Any]:
        self.seen.append(("config", raw))
        return {"parser": "spy"}


@pytest.mark.parametrize("supported", [Support(True, "retained success reason"), Support(False, "old refusal")])
def test_default_hook_preserves_legacy_support_and_reason(sample: dict[str, Any], supported: Support) -> None:
    raw = ReplayInput.from_fixture(Fixture.from_dict(sample))
    adapter = Spy(supported)
    assert adapter.supports_replay(raw) is supported
    assert adapter.supports_replay(replace(raw, assumes=())) is supported
    assert adapter.seen == []


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"assumes": []},
        {"assumes": ["stop_token_ignored"]},
        {"finish": {"reason": "stop", "token": "<second>"}},
        {"finish": {"reason": "length", "token": None}},
        {"assumes": ["stop_token_ignored"], "finish": {"reason": "stop", "token": "<second>"}},
    ],
)
def test_metadata_roundtrip_is_lossless(
    sample: dict[str, Any], sample_fixtures_dir: Path, metadata: dict[str, Any]
) -> None:
    record = {**sample, **metadata}
    fixture = Fixture.from_dict(record)
    assert fixture.to_dict() == record
    wire = json.loads(json.dumps(fixture.to_dict()))
    family = load_families(sample_fixtures_dir)[fixture.family]
    raw = ReplayInput.from_fixture(Fixture.from_dict(wire), family)
    assert raw.assumes == (tuple(metadata["assumes"]) if "assumes" in metadata else None)
    assert raw.finish == (Finish.from_dict(metadata["finish"]) if "finish" in metadata else None)
    assert raw.stop_tokens == family.reference_for(fixture.reference_model).stop_tokens
    assert not hasattr(raw, "expected") and not hasattr(raw, "expected_error")
    if raw.finish:
        assert raw.finished_by_length == (raw.finish.reason == "length")


@pytest.mark.parametrize(
    "patch",
    [
        {"assumes": ["unknown"]},
        {"assumes": ["stop_token_ignored", "stop_token_ignored"]},
        {"assumes": "stop_token_ignored"},
        {"assumes": None},
        {"assumes": ""},
        {"assumes": {}},
        {"assumes": {"stop_token_ignored": True}},
        {"finish": None},
        {"finish": {}},
        {"finish": {"reason": "unknown", "token": None}},
        {"finish": {"reason": "length", "token": "<first>"}},
        {"finish": {"reason": "stop", "token": None}},
        {"finish": {"reason": "stop", "token": ""}},
        {"finish": {"reason": "stop", "token": True}},
        {"finish": {"reason": "length", "token": None, "extra": 1}},
        {"finish": {"reason": "stop", "token": "<first>"}, "tags": ["truncated"]},
        {"finish": {"reason": "stop", "token": "<first>"}, "raw_output": "answer<first>"},
    ],
)
def test_invalid_metadata_is_validation_error_not_unsupported(
    sample: dict[str, Any], sample_fixtures_dir: Path, tmp_path: Path, patch: dict[str, Any]
) -> None:
    record = {**sample, **patch}
    dest = tmp_path / "qwen3-hermes"
    dest.mkdir()
    (dest / "family.json").write_bytes((sample_fixtures_dir / "qwen3-hermes/family.json").read_bytes())
    (dest / "case.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert validate([tmp_path])
    with pytest.raises((TypeError, ValueError)):
        Fixture.from_dict(record)
    spy = Spy()
    answer = handle(spy, {"op": "replay", "fixture": record, "strategies": ["one"]})
    assert answer["ok"] is False and "supported" not in answer
    assert spy.seen == []


def test_refusal_happens_before_all_parse_work(sample: dict[str, Any]) -> None:
    fixture = Fixture.from_dict({**sample, "assumes": ["stop_token_ignored"]})
    spy = Spy()
    answer = replay(spy, fixture, None, [ChunkStrategy.parse("one")])
    assert answer == {
        "ok": True,
        "fixture_id": fixture.id,
        "supported": False,
        "reason": "replay assumption not implemented: stop_token_ignored",
    }
    assert spy.seen == []


def test_override_cannot_widen_family_refusal(sample: dict[str, Any]) -> None:
    class Overbroad(Spy):
        def supports_replay(self, raw: ReplayInput) -> Support:
            raise AssertionError("family refusal must win before the override")

    spy = Overbroad(Support(False, "family not implemented"))
    answer = replay(spy, Fixture.from_dict(sample), None, [])
    assert answer["supported"] is False and answer["reason"] == "family not implemented"
    assert spy.seen == []


@pytest.mark.parametrize("finish", [Finish("stop", "<second>"), Finish("length", None)])
def test_worker_delivers_identical_finish_to_all_paths(sample: dict[str, Any], finish: Finish) -> None:
    class ImplementsFinish(Spy):
        def supports_replay(self, raw: ReplayInput) -> Support:
            return self.supports(raw.family, raw.model)

    spy = ImplementsFinish()
    fixture = Fixture.from_dict({**sample, "finish": finish.to_dict()})
    answer = replay(spy, fixture, None, [ChunkStrategy.parse("one"), ChunkStrategy.parse("token")])
    assert answer["supported"] is True
    assert {entry for entry, _ in spy.seen} == {"units", "parse", "stream", "config"}
    assert all(raw is spy.seen[0][1] and raw.finish == finish for _, raw in spy.seen)
    assert answer["parser_config"]["replay_conditions"]["finish"] == finish.to_dict()


@pytest.mark.parametrize("engine", ENGINES)
def test_two_annotated_cases_refuse_without_engine_imports(engine: str) -> None:
    adapter = load_adapter(engine)
    families = load_families(default_fixtures_dir())
    fixtures = load_fixtures([default_fixtures_dir()])
    before = set(sys.modules)
    for fixture in fixtures:
        raw = ReplayInput.from_fixture(fixture, families[fixture.family])
        previous = adapter.supports(raw.family, raw.model)
        current = adapter.supports_replay(raw)
        if fixture.id in MARKED:
            assert not current and current.reason
            answer = replay(adapter, fixture, families[fixture.family], [ChunkStrategy.parse("one")])
            assert answer["supported"] is False and "nonstream" not in answer and "parser_config" not in answer
        else:
            assert raw.assumes is None and raw.finish is None
            assert current == previous
    assert not {name for name in set(sys.modules) - before if name.split(".")[0] in set(ENGINES) | {"torch"}}


def test_marked_and_unmarked_in_same_actual_worker(sample: dict[str, Any], reference_adapter_spec: str) -> None:
    marked = {**sample, "id": sample["id"] + "-conditional", "assumes": ["stop_token_ignored"]}
    messages = [{"op": "replay", "fixture": f, "strategies": ["one"]} for f in [sample, marked, sample]]
    messages.append({"op": "shutdown"})
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root / "src"), str(Path(__file__).parent)]))
    process = subprocess.run(
        [sys.executable, "-m", "canitoolcall.adapters.worker", reference_adapter_spec],
        input="".join(json.dumps(m) + "\n" for m in messages),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=20,
    )
    assert process.returncode == 0, process.stderr
    replies = [json.loads(line) for line in process.stdout.splitlines()]
    assert len(replies) == 4 and replies[0] == replies[2]
    assert replies[0]["supported"] is True and "replay_conditions" not in replies[0]["parser_config"]
    assert replies[1]["supported"] is False and replies[1]["reason"]
    assert "nonstream" not in replies[1] and replies[-1] == {"ok": True}


@pytest.mark.parametrize("strategy", ["nonstream", "one", "char:1"])
def test_wrong_layer_unsupported_never_becomes_pass(strategy: str) -> None:
    with pytest.raises(ValueError, match="adapter applicability"):
        case_status([CheckResult("expected_match", strategy, Status.UNSUPPORTED)])
    assert case_status([CheckResult("expected_match", "nonstream", Status.PASS)]) is Status.PASS


def raw_finish(finish: Finish | None = None, truncated: bool = False) -> ReplayInput:
    return ReplayInput(
        "synthetic/control",
        "qwen3-hermes",
        "Qwen/Qwen3-0.6B",
        "answer",
        stop_tokens=("<first>", "<second>"),
        truncated=truncated,
        finish=finish,
    )


@pytest.mark.parametrize(
    ("finish", "truncated", "expected"),
    [
        (None, False, ("<first>", "<second>")),
        (None, True, ()),
        (Finish("stop", "<second>"), False, ("<second>",)),
        (Finish("length", None), False, ()),
    ],
)
def test_candidates_are_separate_from_termination(
    finish: Finish | None, truncated: bool, expected: tuple[str, ...]
) -> None:
    raw = raw_finish(finish, truncated)
    assert raw.stop_tokens == ("<first>", "<second>") and raw.replay_end_tokens == expected


@pytest.mark.parametrize(
    ("finish", "truncated", "token", "reason"),
    [
        (None, False, "<first>", "stop"),
        (None, True, None, "length"),
        (Finish("stop", "<second>"), False, "<second>", "stop"),
        (Finish("length", None), False, None, "length"),
    ],
)
def test_sglang_actual_stop_planner(finish: Finish | None, truncated: bool, token: str | None, reason: str) -> None:
    adapter = load_adapter("sglang")
    tokenizer = SimpleNamespace(
        unk_token_id=-1,
        eos_token="<fallback>",
        convert_tokens_to_ids=lambda x: {"<first>": 11, "<second>": 42}.get(x, -1),
    )
    raw = raw_finish(finish, truncated)
    plan = adapter._stop_plan(SimpleNamespace(tokenizer=tokenizer), raw, [1, 2])
    assert plan.token == token and plan.finish_type == reason
    if reason == "length":
        assert plan.finish_reason() == {"type": "length", "length": 2}
        assert plan.appended is False
    else:
        assert plan.finish_reason()["matched"] == tokenizer.convert_tokens_to_ids(token)


@pytest.mark.parametrize("nonstream", [False, True])
@pytest.mark.parametrize("finish", [Finish("stop", "<second>"), Finish("length", None)])
def test_llamacpp_actual_request_uses_same_explicit_finish(
    monkeypatch: pytest.MonkeyPatch, nonstream: bool, finish: Finish
) -> None:
    adapter = load_adapter("llamacpp")
    requests: list[dict[str, Any]] = []

    def request(value: dict[str, Any]) -> dict[str, Any]:
        requests.append(copy.deepcopy(value))
        return {"ok": True, "config": {"end_token": finish.token}}

    monkeypatch.setattr(adapter, "_vocab", lambda raw: SimpleNamespace(path=Path("unused-vocab.gguf")))
    monkeypatch.setattr(adapter, "_h", lambda: SimpleNamespace(request=request))
    monkeypatch.setattr(adapter, "_kwargs", lambda raw: {})
    raw = raw_finish(finish)
    template = TemplateChoice(source="gguf", text=None, sha256="0" * 64, path="unused")
    adapter._request(raw, [], template=template, nonstream=nonstream, units=[1, 2])
    assert requests[0]["end_tokens"] == ([finish.token] if finish.token is not None else [])
    assert requests[0]["ids"] == [1, 2] and raw.stop_tokens == ("<first>", "<second>")
    monkeypatch.setattr(
        adapter, "_h", lambda: SimpleNamespace(request=lambda value: {"ok": True, "config": {"end_token": "<wrong>"}})
    )
    with pytest.raises(ValueError, match=r"explicit finish\.token"):
        adapter._request(raw, [], template=template, nonstream=nonstream, units=[1, 2])


@pytest.mark.parametrize("engine", ENGINES)
def test_explicit_finish_is_never_silently_ignored(engine: str) -> None:
    raw = raw_finish(Finish("stop", "<second>"))
    adapter = load_adapter(engine)
    result = adapter.supports_replay(raw)
    if engine in ("sglang", "llamacpp"):
        assert result == adapter.supports(raw.family, raw.model)
    else:
        assert not result and result.reason


def test_all_current_corpus_records_roundtrip_and_validate() -> None:
    root = default_fixtures_dir()
    assert validate([root]) == []
    conditional = set()
    for path in root.glob("*/*.jsonl"):
        for _, record in read_jsonl(path):
            assert Fixture.from_dict(record).to_dict() == record
            if record.get("assumes"):
                assert record["assumes"] == ["stop_token_ignored"]
                conditional.add(record["id"])
    assert conditional == MARKED


def test_runner_keeps_refusal_out_of_checks(
    sample: dict[str, Any],
    sample_fixtures_dir: Path,
    tmp_path: Path,
    reference_adapter_spec: str,
    reference_env: dict[str, str],
) -> None:
    from canitoolcall.runner import RunConfig, run

    directory = tmp_path / "qwen3-hermes"
    directory.mkdir()
    (directory / "family.json").write_bytes((sample_fixtures_dir / "qwen3-hermes/family.json").read_bytes())
    conditional = {**sample, "id": sample["id"] + "-conditional", "assumes": ["stop_token_ignored"]}
    (directory / "cases.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in (sample, conditional)), encoding="utf-8"
    )
    result = run(
        RunConfig(engine=reference_adapter_spec, fixtures=(tmp_path,), python=Path(sys.executable), env=reference_env)
    )
    rows = {case.fixture_id: case for case in result.cases}
    assert rows[sample["id"]].status is Status.PASS
    declined = rows[conditional["id"]]
    assert declined.status is Status.UNSUPPORTED and declined.reason
    assert declined.checks == () and declined.observed is None and declined.parser_config is None


def test_sglang_explicit_unknown_token_never_falls_back() -> None:
    adapter = load_adapter("sglang")
    tokenizer = SimpleNamespace(unk_token_id=-1, eos_token="<first>", convert_tokens_to_ids=lambda value: -1)
    with pytest.raises(ValueError, match=r"finish\.token is not a known token"):
        adapter._stop_plan(SimpleNamespace(tokenizer=tokenizer), raw_finish(Finish("stop", "<second>")), [1])

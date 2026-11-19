"""The replay report checks shared by both upstream-PR verifiers, on synthetic files only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import replay_report


def write(out: Path, *payloads: bytes) -> Path:
    for i, payload in enumerate(payloads):
        (out / f"report-{i}.json").write_bytes(payload)
    return out


def document(value: object) -> bytes:
    return json.dumps(value).encode("utf-8")


def cases(*entries: tuple[object, object]) -> bytes:
    # Only the report fields the verifiers consume; these are not engine fixtures.
    return document({"cases": [{"fixture_id": key, "status": status} for key, status in entries]})


@pytest.mark.parametrize("code", [0, 1])
def test_accepts_a_complete_report_with_every_status(tmp_path: Path, code: int) -> None:
    statuses = ["pass", "soft_pass", "fail", "error", "unsupported"]
    write(tmp_path, cases(*((f"fam/{s}", s) for s in statuses)))
    assert replay_report.read_cases(tmp_path, code) == {f"fam/{s}": s for s in statuses}


@pytest.mark.parametrize("code", [2, 3, -9, 130])
def test_rejects_unsuccessful_replay_even_with_a_valid_report(tmp_path: Path, code: int) -> None:
    write(tmp_path, cases(("fam/a", "pass")))
    with pytest.raises(replay_report.ReportError, match=f"replay exited {code}$"):
        replay_report.read_cases(tmp_path, code)


@pytest.mark.parametrize(
    ("payloads", "message"),
    [
        pytest.param((), "exactly one results JSON file, found 0", id="missing-report"),
        pytest.param((cases(("fam/a", "pass")),) * 2, "exactly one results JSON file, found 2", id="multiple-reports"),
        pytest.param((b"{",), "cannot read results JSON", id="malformed-json"),
        pytest.param((b"\xff",), "cannot read results JSON", id="non-utf8"),
        pytest.param((document([]),), "nonempty cases list", id="report-not-object"),
        pytest.param((document({}),), "nonempty cases list", id="missing-cases"),
        pytest.param((document({"cases": {}}),), "nonempty cases list", id="cases-not-list"),
        pytest.param((document({"cases": None}),), "nonempty cases list", id="null-cases"),
        pytest.param((cases(),), "nonempty cases list", id="empty-cases"),
        pytest.param((document({"cases": ["fam/a"]}),), "case 0 must be an object", id="case-not-object"),
        pytest.param((document({"cases": [{"status": "pass"}]}),), "nonempty string fixture_id", id="missing-id"),
        pytest.param((cases(("", "pass")),), "nonempty string fixture_id", id="empty-id"),
        pytest.param((cases((" \t", "pass")),), "nonempty string fixture_id", id="blank-id"),
        pytest.param((cases((1, "pass")),), "nonempty string fixture_id", id="numeric-id"),
        pytest.param((cases(([], "pass")),), "nonempty string fixture_id", id="unhashable-id"),
        pytest.param((cases(("fam/a", "pass"), ("fam/a", "fail")),), "duplicate fixture_id", id="duplicate-id"),
        pytest.param((document({"cases": [{"fixture_id": "fam/a"}]}),), "invalid status", id="missing-status"),
        pytest.param((cases(("fam/a", "unknown")),), "invalid status", id="unknown-status"),
        pytest.param((cases(("fam/a", [])),), "invalid status", id="non-string-status"),
    ],
)
def test_rejects_incomplete_or_invalid_reports(tmp_path: Path, payloads: tuple[bytes, ...], message: str) -> None:
    write(tmp_path, *payloads)
    with pytest.raises(replay_report.ReportError, match=message):
        replay_report.read_cases(tmp_path, 0)


def test_report_errors_are_runtime_errors() -> None:
    # Both verifiers already turn RuntimeError into exit code 3.
    assert issubclass(replay_report.ReportError, RuntimeError)


def test_matching_inventories_in_any_order_pass() -> None:
    replay_report.check_inventories({"a": "pass", "b": "fail"}, {"b": "pass", "a": "error"})


@pytest.mark.parametrize(
    ("head", "lines"),
    [
        pytest.param({"a": "pass"}, ["missing at head: b"], id="missing"),
        pytest.param({"a": "pass", "b": "pass", "c": "fail"}, ["added at head: c"], id="added"),
        pytest.param({"a": "pass", "c": "pass"}, ["missing at head: b", "added at head: c"], id="same-size"),
        pytest.param({}, ["missing at head: a, b"], id="empty-head"),
    ],
)
def test_rejects_mismatched_inventories(head: dict[str, str], lines: list[str]) -> None:
    with pytest.raises(replay_report.ReportError) as raised:
        replay_report.check_inventories({"a": "pass", "b": "pass"}, head)
    assert str(raised.value).splitlines() == [
        "base and head fixture inventories differ; comparison is incomplete",
        *lines,
    ]

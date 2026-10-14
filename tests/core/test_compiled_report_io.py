"""Bounded report-I/O faults on disposable files, with no engine or network."""

from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("compiled_io_subject", SCRIPTS / "verify_compiled_pr.py")
compiled = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compiled)


@pytest.fixture
def study(tmp_path, monkeypatch):
    root, scratch = tmp_path / "repo", tmp_path / "scratch"
    pinned = root / ".engines/ollama/bin/ctcreplay"
    pinned.parent.mkdir(parents=True)
    pinned.write_bytes(b"synthetic pinned binary")
    scratch.mkdir()
    monkeypatch.setattr(compiled.tempfile, "mkdtemp", lambda **kwargs: str(scratch))
    metadata = json.dumps({"base": {"sha": "a" * 40}, "head": {"sha": "b" * 40}})
    monkeypatch.setattr(compiled.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=metadata))
    monkeypatch.setattr(compiled, "build", lambda *args: ({}, {"binary": "synthetic-hash"}))
    monkeypatch.setattr(compiled, "replay", lambda *args: {"one": "pass"})
    return root, scratch, pinned, tmp_path / "export/report.json"


def inject_write_fault(monkeypatch, scratch, output, scenario):
    """Keep real file writes; deliberately fail a selected logical report write."""
    real_open, real_fdopen = Path.open, compiled.os.fdopen
    real_mkstemp = compiled.tempfile.mkstemp
    descriptors, attempts = {}, {"local": 0, "export": 0}

    def mkstemp(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        descriptors[fd] = Path(name)
        return fd, name

    def classify(path):
        if path == scratch / "verification.json" or path.name.startswith(".verification.json."):
            return "local"
        if path == output or path.name.startswith(".report.json."):
            return "export"
        return None

    @contextmanager
    def fault_context(stream, path):
        kind = classify(path)
        if kind:
            attempts[kind] += 1
        with stream:
            fault = (kind == "local" and scenario.startswith("local-")) or (
                kind == "export" and scenario.startswith("export-")
            )
            fault |= scenario == "export-and-recovery" and kind == "local" and attempts[kind] > 1
            if fault:
                if scenario.endswith("partial"):
                    stream.write('{"partial":')
                    stream.flush()
                raise OSError("SYNTHETIC_REPORT_WRITE_FAILURE")
            yield stream

    def path_open(path, mode="r", *args, **kwargs):
        stream = real_open(path, mode, *args, **kwargs)
        return fault_context(stream, path) if any(m in mode for m in "wxa") else stream

    def fdopen(fd, *args, **kwargs):
        return fault_context(real_fdopen(fd, *args, **kwargs), descriptors[fd])

    monkeypatch.setattr(compiled.tempfile, "mkstemp", mkstemp)
    monkeypatch.setattr(compiled.os, "fdopen", fdopen)
    monkeypatch.setattr(Path, "open", path_open)
    return attempts


@pytest.mark.parametrize(
    "scenario", ["local-denied", "local-partial", "export-denied", "export-partial", "export-and-recovery"]
)
def test_report_failure_is_explicit_and_never_leaves_partial_export(study, monkeypatch, capsys, scenario):
    root, scratch, pinned, output = study
    attempts = inject_write_fault(monkeypatch, scratch, output, scenario)
    assert compiled.verify(root, "ollama", 22, [], output, 2) == 3
    captured = capsys.readouterr()
    fallback = [line for line in captured.err.splitlines() if line.startswith("VERIFICATION_REPORT_JSON ")]
    report = (
        json.loads(fallback[-1].split(" ", 1)[1])
        if fallback
        else json.loads((scratch / "verification.json").read_text())
    )
    assert report["exit_code"] == 3
    assert "SYNTHETIC_REPORT_WRITE_FAILURE" in report.get("report_error", report.get("error", ""))
    assert report["fixed"] == 0 and report["regressed"] == 0
    assert "fixtures replayed:" not in captured.out
    assert not output.exists()
    assert not list(output.parent.glob(".report.json.*"))
    assert attempts["local"] >= 1
    assert pinned.read_bytes() == b"synthetic pinned binary"


@pytest.mark.parametrize(("status", "expected"), [("pass", 0), ("fail", 1), ("unsupported", 3)])
def test_normal_reports_keep_exit_and_contents(study, monkeypatch, status, expected):
    root, scratch, pinned, output = study
    monkeypatch.setattr(
        compiled, "replay", lambda root, families, dest, env: {"one": "pass" if dest.name == "base" else status}
    )
    assert compiled.verify(root, "ollama", 22, [], output, 2) == expected
    assert output.read_bytes() == (scratch / "verification.json").read_bytes()
    assert json.loads(output.read_text())["exit_code"] == expected
    assert pinned.read_bytes() == b"synthetic pinned binary"


def test_file_created_during_replay_is_preserved(study, monkeypatch):
    root, scratch, pinned, output = study

    def replay(*args):
        output.parent.mkdir(exist_ok=True)
        output.write_text("concurrent unrelated report")
        return {"one": "pass"}

    monkeypatch.setattr(compiled, "replay", replay)
    assert compiled.verify(root, "ollama", 22, [], output, 2) == 3
    assert output.read_text() == "concurrent unrelated report"
    assert json.loads((scratch / "verification.json").read_text())["exit_code"] == 3
    assert pinned.read_bytes() == b"synthetic pinned binary"


def test_output_parent_becoming_a_file_is_explicit(study, monkeypatch):
    root, scratch, pinned, output = study

    def replay(*args):
        output.parent.write_text("not a directory")
        return {"one": "pass"}

    monkeypatch.setattr(compiled, "replay", replay)
    assert compiled.verify(root, "ollama", 22, [], output, 2) == 3
    assert output.parent.read_text() == "not a directory"
    assert json.loads((scratch / "verification.json").read_text())["exit_code"] == 3
    assert pinned.read_bytes() == b"synthetic pinned binary"

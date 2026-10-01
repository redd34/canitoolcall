"""Compiled PR isolation, pin validation, argument and report controls."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import verify_compiled_pr as compiled  # noqa: E402

SHA = "a" * 40
OTHER = "b" * 40


@pytest.mark.parametrize("bad", [None, "", "HEAD", "../x", "a" * 39, "A" * 40, "g" * 40, 123])
def test_commit_rejects_nonimmutable_inputs(bad):
    with pytest.raises(RuntimeError, match="commit SHA"):
        compiled.commit_sha(bad)


def test_commit_accepts_full_sha():
    assert compiled.commit_sha(SHA) == SHA


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"cases": []},
        {"cases": [None]},
        {"cases": [{"fixture_id": "", "status": "pass"}]},
        {"cases": [{"fixture_id": "a", "status": "unknown"}]},
        {"cases": [{"fixture_id": "a", "status": "unsupported"}]},
        {"cases": [{"fixture_id": "a", "status": "pass"}] * 2},
    ],
)
def test_invalid_reports_are_not_success(tmp_path, value):
    (tmp_path / "result.json").write_text(json.dumps(value))
    with pytest.raises(RuntimeError):
        compiled.read_cases(tmp_path, 0)


def test_replay_failure_does_not_hide_behind_valid_json(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"cases": [{"fixture_id": "a", "status": "pass"}]}))
    with pytest.raises(RuntimeError, match="exited 3"):
        compiled.read_cases(tmp_path, 3)
    assert compiled.read_cases(tmp_path, 0) == {"a": "pass"}


@pytest.mark.parametrize("head", [{}, {"other": "pass"}, {"a": "unsupported"}])
def test_dropped_or_unsupported_cases_do_not_count_as_improvements(head):
    with pytest.raises(RuntimeError):
        compiled.compare({"a": "pass"}, head)


def test_comparison_counts_real_regressions_and_fixes():
    value = compiled.compare({"a": "pass", "b": "fail"}, {"a": "error", "b": "soft_pass"})
    assert value == {
        "fixtures": 2,
        "changed": [("a", "pass", "error"), ("b", "fail", "soft_pass")],
        "fixed": 1,
        "regressed": 1,
    }


@pytest.mark.parametrize("engine", ["ollama", "llamacpp"])
def test_build_uses_fresh_workspace_and_separate_adapter_pin(tmp_path, monkeypatch, engine):
    root, destination = tmp_path / "repo", tmp_path / "work"
    (root / "scripts/engines").mkdir(parents=True)
    (root / "scripts/engines" / f"{engine}.sh").write_text("echo stub")
    (root / "harnesses" / engine).mkdir(parents=True)
    pinned = root / ".engines" / engine / "bin/ctcreplay"
    pinned.parent.mkdir(parents=True)
    pinned.write_bytes(b"original pinned bytes")
    monkeypatch.setenv("JOBS", "99")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == "git":
            return SimpleNamespace(stdout=SHA, returncode=0)
        project = destination / "project"
        eng = project / ".engines" / engine
        names = ["bin/ctcreplay", "bin/ctc-detok"] if engine == "ollama" else ["build/canitoolcall-llamacpp"]
        for name in names:
            path = eng / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"new isolated build")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compiled.subprocess, "run", run)
    monkeypatch.setattr(compiled, "run_logged", run)
    env, hashes = compiled.build(root, engine, SHA, destination, 2)
    assert pinned.read_bytes() == b"original pinned bytes"
    assert hashes and all(len(value) == 64 for value in hashes.values())
    assert calls[0][1]["cwd"] == destination / "project"
    assert calls[0][1]["env"][f"{engine.upper()}_REF"] == SHA
    assert calls[0][1]["env"]["JOBS"] == "2"
    assert "--no-venv" in calls[0][0]
    assert ("--no-vocab" in calls[0][0]) == (engine == "llamacpp")
    assert env["PYTHONPATH"] == str(destination)
    assert SHA in (destination / "ctc_revision_adapter.py").read_text()
    assert not (root / ".venvs").exists()
    assert not (root / "ctc_revision_adapter.py").exists()


@pytest.mark.parametrize("engine", ["ollama", "llamacpp"])
def test_build_failure_keeps_pinned_binaries(tmp_path, monkeypatch, engine):
    root = tmp_path / "repo"
    (root / "scripts/engines").mkdir(parents=True)
    (root / "harnesses" / engine).mkdir(parents=True)
    name = "bin/ctcreplay" if engine == "ollama" else "build/canitoolcall-llamacpp"
    path = root / ".engines" / engine / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b"do not alter")
    before = compiled.pinned_binaries(root, engine)
    monkeypatch.setattr(compiled.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1))
    monkeypatch.setattr(compiled, "run_logged", lambda *a, **kw: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match="build exited"):
        compiled.build(root, engine, SHA, tmp_path / "work", 2)
    assert compiled.pinned_binaries(root, engine) == before

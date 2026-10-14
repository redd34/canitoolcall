"""Check interruption reports separately from real process lifecycle tests."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import verify_compiled_pr as compiled
from compiled_process import VerificationInterrupted


@pytest.mark.parametrize("stage", ["build", "replay"])
@pytest.mark.parametrize(
    ("error", "code"),
    [
        (subprocess.TimeoutExpired("owned-test", 1), 3),
        (KeyboardInterrupt(), 130),
        (VerificationInterrupted(signal.SIGTERM), 143),
    ],
)
def test_failure_reports_preserve_stage_and_never_claim_success(tmp_path, monkeypatch, capsys, stage, error, code):
    root, scratch = tmp_path / "repo", tmp_path / "scratch"
    scratch.mkdir()
    pinned = root / ".engines/ollama/bin/ctcreplay"
    pinned.parent.mkdir(parents=True)
    pinned.write_bytes(b"synthetic pinned binary")
    monkeypatch.setattr(compiled.tempfile, "mkdtemp", lambda **kw: str(scratch))
    metadata = json.dumps({"base": {"sha": "a" * 40}, "head": {"sha": "b" * 40}})
    monkeypatch.setattr(compiled.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=metadata))
    monkeypatch.setattr(compiled, "build", lambda *a: ({}, {}))
    monkeypatch.setattr(compiled, "replay", lambda *a: {"one": "pass"})

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(compiled, stage, fail)
    output = tmp_path / "report.json"
    assert compiled.verify(root, "ollama", 22, [], output, 2) == code
    saved = json.loads(output.read_text())
    assert saved["exit_code"] == code and saved["pinned_unchanged"]
    assert saved["status"] == ("unreplayable" if code == 3 else "interrupted")
    assert json.loads((scratch / "verification.json").read_text()) == saved
    assert pinned.read_bytes() == b"synthetic pinned binary"
    assert "fixtures replayed:" not in capsys.readouterr().out

"""Output-safety and command exit tests; no real network or compiler is used."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "compiled_contract_subject", os.environ.get("CTC_CONTRACT_MODULE", str(SCRIPTS / "verify_compiled_pr.py"))
)
compiled = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compiled)


@pytest.fixture
def setup_case(tmp_path, monkeypatch):
    root, scratch = tmp_path / "repo", tmp_path / "scratch"
    pinned = root / ".engines/ollama/bin/ctcreplay"
    pinned.parent.mkdir(parents=True)
    pinned.write_bytes(b"original synthetic pinned binary")
    calls = []

    def temporary(**kwargs):
        scratch.mkdir()
        return str(scratch)

    def metadata(*args, **kwargs):
        calls.append("metadata")
        return SimpleNamespace(stdout=json.dumps({"base": {"sha": "a" * 40}, "head": {"sha": "b" * 40}}))

    monkeypatch.setattr(compiled.tempfile, "mkdtemp", temporary)
    monkeypatch.setattr(compiled.subprocess, "run", metadata)
    monkeypatch.setattr(compiled, "build", lambda *args: ({}, {"binary": "synthetic-build-hash"}))
    monkeypatch.setattr(compiled, "replay", lambda *args: {"one": "pass"})
    return root, scratch, pinned, calls


@pytest.mark.parametrize(
    "relative",
    [
        ".engines/ollama/bin/ctcreplay",
        ".engines/llamacpp/build/report.json",
        ".engines/gguf/vocabulary.gguf",
        ".venvs/ollama/bin/python",
        ".git/config",
    ],
)
def test_output_cannot_write_engine_or_environment(setup_case, relative):
    root, scratch, pinned, calls = setup_case
    output = root / relative
    old = output.read_bytes() if output.is_file() else None
    assert compiled.verify(root, "ollama", 22, ["glm"], output, 2) == 3
    assert not calls and not scratch.exists()
    assert (output.read_bytes() if output.is_file() else None) == old
    assert pinned.read_bytes() == b"original synthetic pinned binary"


def test_existing_report_is_not_overwritten(setup_case, tmp_path):
    root, scratch, _pinned, calls = setup_case
    output = tmp_path / "report.json"
    output.write_text("existing user report")
    assert compiled.verify(root, "ollama", 22, [], output, 2) == 3
    assert output.read_text() == "existing user report"
    assert not calls and not scratch.exists()


@pytest.mark.parametrize("link_type", ["symlink", "hardlink"])
def test_alias_cannot_overwrite_pinned_binary(setup_case, tmp_path, link_type):
    root, scratch, pinned, calls = setup_case
    alias = tmp_path / "report.json"
    try:
        if link_type == "hardlink":
            os.link(pinned, alias)
        else:
            alias.symlink_to(pinned)
    except OSError as exc:
        pytest.skip(f"host cannot create {link_type}: {exc}")
    assert compiled.verify(root, "ollama", 22, [], alias, 2) == 3
    assert pinned.read_bytes() == b"original synthetic pinned binary"
    assert not calls and not scratch.exists()


@pytest.mark.parametrize(("status", "expected"), [("pass", 0), ("fail", 1), ("unsupported", 3)])
def test_verify_reports_outcome_without_touching_pin(setup_case, monkeypatch, tmp_path, status, expected):
    root, scratch, pinned, _calls = setup_case
    monkeypatch.setattr(
        compiled, "replay", lambda root, families, dest, env: {"one": "pass" if dest.name == "base" else status}
    )
    output = tmp_path / "report.json"
    assert compiled.verify(root, "ollama", 22, ["glm"], output, 2) == expected
    result = json.loads(output.read_text())
    assert result["exit_code"] == expected and result["pinned_unchanged"]
    assert result == json.loads((scratch / "verification.json").read_text())
    assert pinned.read_bytes() == b"original synthetic pinned binary"


def load_cli():
    spec = importlib.util.spec_from_file_location("cli_contract_subject", SCRIPTS / "verify_upstream_pr.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("engine", ["ollama", "llamacpp"])
@pytest.mark.parametrize("exit_code", [0, 1, 3])
def test_cli_dispatches_and_preserves_exit(monkeypatch, engine, exit_code):
    cli, calls = load_cli(), []
    monkeypatch.setattr(cli.verify_compiled_pr, "verify", lambda *args: calls.append(args) or exit_code)
    monkeypatch.setattr(
        sys, "argv", ["verify", "--engine", engine, "--pr", "22", "--family", "glm", "--build-jobs", "3"]
    )
    assert cli.main() == exit_code
    assert calls[0][1:] == (engine, 22, ["glm"], None, 3)


@pytest.mark.parametrize(
    "suffix",
    [
        ["--pr", "0"],
        ["--pr", "-1"],
        ["--build-jobs", "0"],
        ["--build-jobs", "-1"],
        ["--engine", "invalid"],
    ],
)
def test_invalid_cli_arguments_never_build(monkeypatch, suffix):
    cli, calls = load_cli(), []
    monkeypatch.setattr(cli.verify_compiled_pr, "verify", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["verify", "--engine", "ollama", "--pr", "22", *suffix])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2 and not calls

"""Review regressions on actual orchestration, without external engines or accounts."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
SHA = "a" * 40
SECRETS = ("GH_TOKEN", "GITHUB_TOKEN", "HF_TOKEN", "CANITOOLCALL_API_KEY", "UNRELATED_PRIVATE_VALUE")


@pytest.fixture
def compiled(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("verify_compiled_pr")


def test_python_cli_import_and_help_without_compiled_modules(tmp_path):
    code = """
import importlib.util, sys
path = sys.argv[1]
spec = importlib.util.spec_from_file_location('isolated_verifier', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert 'verify_compiled_pr' not in sys.modules
assert 'compiled_process' not in sys.modules
sys.argv = ['verify', '--help']
try:
    module.main()
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError('help did not exit')
assert 'verify_compiled_pr' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(SCRIPTS / "verify_upstream_pr.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "ollama" in result.stdout and "sglang" in result.stdout


@pytest.mark.parametrize("engine", ["ollama", "llamacpp"])
def test_build_filters_environment_and_applies_default_timeout(compiled, tmp_path, monkeypatch, engine):
    root, dest = tmp_path / "repo", tmp_path / "work"
    (root / "scripts/engines").mkdir(parents=True)
    (root / "scripts/engines" / (engine + ".sh")).write_text("echo synthetic")
    (root / "harnesses" / engine).mkdir(parents=True)
    for name in SECRETS:
        monkeypatch.setenv(name, "synthetic-private-marker")
    monkeypatch.setenv("BASH_ENV", "/synthetic/not-to-execute")
    monkeypatch.setenv("GOCACHE", str(tmp_path / "go-cache"))
    monkeypatch.setenv("CANITOOLCALL_GGUF_DIR", str(tmp_path / "vocab"))
    seen = []

    def build_recipe(command, **kwargs):
        seen.append(kwargs)
        base = dest / "project/.engines" / engine
        names = ["bin/ctcreplay", "bin/ctc-detok"] if engine == "ollama" else ["build/canitoolcall-llamacpp"]
        for name in names:
            p = base / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"synthetic binary bytes, never executed")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compiled, "run_logged", build_recipe)
    monkeypatch.setattr(compiled.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=SHA, returncode=0))
    overrides, _ = compiled.build(root, engine, SHA, dest, 2)
    passed = seen[0]["env"]
    assert not set(SECRETS).intersection(passed)
    assert "BASH_ENV" not in passed
    assert passed["GOCACHE"] == str(tmp_path / "go-cache")
    assert passed["CANITOOLCALL_GGUF_DIR"] == str(tmp_path / "vocab")
    assert passed[f"{engine.upper()}_REF"] == SHA and passed["JOBS"] == "2"
    assert seen[0].get("timeout") == 3600
    assert overrides["CANITOOLCALL_GGUF_DIR"] == str(tmp_path / "vocab")


def test_replay_filters_environment_and_preserves_owned_overrides(compiled, tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    for name in SECRETS:
        monkeypatch.setenv(name, "synthetic-private-marker")
    monkeypatch.setenv("PYTHONPATH", "/unrelated/import/path")
    overrides = {"PYTHONPATH": str(tmp_path), "CANITOOLCALL_OLLAMA_BIN_DIR": str(tmp_path / "owned-bin")}
    seen = []

    def replay(command, **kwargs):
        seen.append((command, kwargs))
        (tmp_path / "results/one.json").write_text(json.dumps({"cases": [{"fixture_id": "case", "status": "pass"}]}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compiled, "run_logged", replay)
    assert compiled.replay(root, ["glm"], tmp_path, overrides) == {"case": "pass"}
    command, kw = seen[0]
    assert not set(SECRETS).intersection(kw["env"])
    assert kw["env"]["CANITOOLCALL_OLLAMA_BIN_DIR"] == overrides["CANITOOLCALL_OLLAMA_BIN_DIR"]
    assert kw["env"]["PYTHONPATH"] == os.pathsep.join([str(root / "src"), str(tmp_path)])
    assert "HF_HUB_OFFLINE=1" in command


@pytest.mark.skipif(os.name != "posix", reason="private process-group execution requires POSIX")
def test_filtered_environment_reaches_real_child(compiled, tmp_path, monkeypatch):
    for name in SECRETS:
        monkeypatch.setenv(name, "synthetic-private-marker")
    monkeypatch.setenv("GOCACHE", str(tmp_path / "go-cache"))
    log = tmp_path / "child.log"
    code = "import os,json; print(json.dumps({k:os.environ.get(k) for k in " + repr([*SECRETS, "GOCACHE"]) + "}))"
    with log.open("w") as stream:
        result = compiled.run_logged(
            [sys.executable, "-I", "-S", "-c", code],
            cwd=tmp_path,
            env=compiled.child_environment(),
            log=stream,
            timeout=10,
        )
    assert result.returncode == 0
    observed = json.loads(log.read_text())
    assert all(observed[name] is None for name in SECRETS)
    assert observed["GOCACHE"] == str(tmp_path / "go-cache")


def test_custom_build_timeout_has_explicit_diagnostic(compiled, tmp_path, monkeypatch):
    root, dest = tmp_path / "repo", tmp_path / "work"
    (root / "scripts/engines").mkdir(parents=True)
    (root / "harnesses/ollama").mkdir(parents=True)

    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 42
        raise subprocess.TimeoutExpired(command, 42)

    monkeypatch.setattr(compiled, "run_logged", timeout)
    with pytest.raises(RuntimeError, match=r"ollama build timed out after 42 seconds.*build.log"):
        compiled.build(root, "ollama", SHA, dest, 2, 42)


@pytest.mark.parametrize("harness_error", [None, "synthetic previously recorded harness error"])
def test_error_is_a_fixture_outcome_not_an_unusable_report(compiled, tmp_path, harness_error):
    cases = [
        {"fixture_id": "old-error", "status": "error", "harness_error": harness_error},
        {"fixture_id": "good", "status": "pass"},
    ]
    (tmp_path / "result.json").write_text(json.dumps({"cases": cases}))
    assert compiled.read_cases(tmp_path, 1) == {"old-error": "error", "good": "pass"}


@pytest.mark.parametrize("seconds", [3600, 7200])
def test_verify_forwards_timeout_to_both_builds_and_reports_other_changes(
    compiled, tmp_path, monkeypatch, capsys, seconds
):
    root, scratch = tmp_path / "repo", tmp_path / "scratch"
    scratch.mkdir()
    pinned = root / ".engines/ollama/bin/ctcreplay"
    pinned.parent.mkdir(parents=True)
    pinned.write_bytes(b"preserve this fixture")
    metadata = {"base": {"sha": SHA}, "head": {"sha": "b" * 40}}
    monkeypatch.setattr(compiled.tempfile, "mkdtemp", lambda **kw: str(scratch))
    monkeypatch.setattr(compiled.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=json.dumps(metadata)))
    builds = []

    def build(*args):
        builds.append(args)
        return {}, {"binary": "synthetic"}

    def replay(_root, _families, dest, _env):
        return (
            {"fixed": "error", "other": "fail", "unchanged": "error"}
            if dest.name == "base"
            else {
                "fixed": "pass",
                "other": "error",
                "unchanged": "error",
            }
        )

    monkeypatch.setattr(compiled, "build", build)
    monkeypatch.setattr(compiled, "replay", replay)
    args = [root, "ollama", 22, ["glm"], tmp_path / "out.json", 2]
    if seconds != 3600:
        args.append(seconds)
    assert compiled.verify(*args) == 0
    assert len(builds) == 2 and all(args[-1] == seconds for args in builds)
    assert "fixtures replayed: 3   fixed: 1   regressed: 0   other changes: 1" in capsys.readouterr().out
    report = json.loads((tmp_path / "out.json").read_text())
    assert report["status"] == "compared" and report["pinned_unchanged"]
    assert pinned.read_bytes() == b"preserve this fixture"


@pytest.mark.parametrize("option", ["--build-jobs", "--build-timeout"])
def test_python_engine_rejects_compiled_only_options(compiled, monkeypatch, capsys, option):
    spec = importlib.util.spec_from_file_location("cli_options_review", SCRIPTS / "verify_upstream_pr.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "gh_json", lambda *a: pytest.fail("must not request metadata"))
    monkeypatch.setattr(sys, "argv", ["verify", "--engine", "sglang", "--pr", "1", option, "2"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert "only supported for compiled engines" in capsys.readouterr().err


@pytest.mark.parametrize("seconds", [1, 7200])
def test_cli_forwards_explicit_timeout(compiled, monkeypatch, seconds):
    spec = importlib.util.spec_from_file_location("timeout_cli_review", SCRIPTS / "verify_upstream_pr.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(compiled, "verify", lambda *args: calls.append(args) or 0)
    monkeypatch.setattr(sys, "argv", ["verify", "--engine", "ollama", "--pr", "22", "--build-timeout", str(seconds)])
    assert module.main() == 0
    assert calls[0][-2:] == (2, seconds)


def test_compiled_cli_rejects_arbitrary_env_before_build(compiled, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("environment_cli_review", SCRIPTS / "verify_upstream_pr.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(compiled, "verify", lambda *args: pytest.fail("must not build"))
    monkeypatch.setattr(sys, "argv", ["verify", "--engine", "ollama", "--pr", "22", "--env", "HF_TOKEN=synthetic"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert "--env is only supported for Python engines" in capsys.readouterr().err

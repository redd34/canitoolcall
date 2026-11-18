"""The GitHub Action (action.yml): its summary renderer, the standalone mock server and the metadata.

Every endpoint here is the local mock server from conftest.py; no real endpoint or key is used.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from canitoolcall import __version__, cli

ROOT = Path(__file__).resolve().parents[2]
ACTION = ROOT / "action.yml"
SUMMARY_SCRIPT = ROOT / "scripts" / "probe_summary.py"
MOCK_SCRIPT = ROOT / "scripts" / "probe_mock_server.py"


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summary = _load(SUMMARY_SCRIPT)


def _outcome(sid: str, mode: str, status: str, detail: str | None = None, warnings: tuple[str, ...] = ()) -> Any:
    return {
        "scenario_id": sid,
        "check": "stream_equals_nonstream" if mode == "equiv" else "scenario",
        "stream": mode != "nonstream",
        "status": status,
        "detail": detail,
        "warnings": list(warnings),
        "latency_s": 0.0,
        "observed": None,
    }


# Hand-written report data in the shape `canitoolcall probe --json` writes; not from a real endpoint.
SMALL_REPORT = {
    "canitoolcall_version": "0.2.0",
    "base_url": "http://127.0.0.1:8011/v1",
    "model": "mock-model",
    "started_at": "2026-11-19T00:00:00+00:00",
    "finished_at": "2026-11-19T00:00:01+00:00",
    "summary": {"pass": 4, "fail": 1, "error": 0, "skip": 0},
    "outcomes": [
        _outcome("single-call", "nonstream", "pass"),
        _outcome("single-call", "stream", "pass", warnings=("finish_reason was 'stop' after a tool call",)),
        _outcome("single-call", "equiv", "pass"),
        _outcome("parallel-calls", "nonstream", "pass"),
        _outcome("parallel-calls", "stream", "fail", "tool-call delta without an integer 'index'"),
    ],
}

SMALL_MARKDOWN = """\
### CanIToolCall probe: 4 pass, 1 fail, 0 error, 0 skip

Model mock-model at http\u200b://127.0.0.1:8011/v1, canitoolcall 0.2.0.

| scenario | non-stream | stream | stream=non-stream |
|---|---|---|---|
| single-call | pass | pass\\* | pass |
| parallel-calls | pass | **fail** | - |

**Problems**

- parallel-calls \\[stream\\] fail: tool-call delta without an integer 'index'

**Warnings** (marked \\*, they do not affect the exit code)

- single-call \\[stream\\]: finish\\_reason was 'stop' after a tool call
"""


def test_render_small_report_exactly() -> None:
    assert summary.render(SMALL_REPORT) == SMALL_MARKDOWN


def test_render_all_pass_has_no_problem_sections() -> None:
    report = {
        **SMALL_REPORT,
        "summary": {"pass": 1, "fail": 0, "error": 0, "skip": 0},
        "outcomes": [_outcome("no-call", "nonstream", "pass")],
    }
    md = summary.render(report)
    assert "| no-call | pass | - | - |" in md
    assert "Problems" not in md and "Warnings" not in md


def test_render_escapes_server_controlled_text() -> None:
    # Details quote what the server returned, so they must not inject markup or break the table.
    evil = "a|b\n<script>alert(1)</script> [x](http://e) `code` *em*"
    report = {
        **SMALL_REPORT,
        "model": "m|`<b>`",
        "outcomes": [_outcome("single-call", "nonstream", "error", evil)],
    }
    md = summary.render(report)
    problems = md.split("**Problems**")[1]
    assert "<script>" not in md and "&lt;script&gt;" in problems
    assert "\n<" not in problems
    assert "a\\|b &lt;script&gt;" in problems  # newline folded to a space, pipe escaped
    assert "\\[x\\]" in problems and "\\`code\\`" in problems and "\\*em\\*" in problems
    assert "<b>" not in md


def test_render_defangs_bare_urls() -> None:
    # GFM autolinks bare URLs, www. hosts and e-mail addresses even in escaped text; a zero-width
    # space keeps a server-supplied link from becoming clickable in the summary.
    text = "see https://evil.example/x, www.evil.example or me@evil.example"
    assert summary.escape(text) == "see https\u200b://evil.example/x, www\u200b.evil.example or me@\u200bevil.example"


def test_render_header_shows_the_exit_code() -> None:
    head = summary.render(SMALL_REPORT, exit_code="1").splitlines()[0]
    assert head == "### CanIToolCall probe: 4 pass, 1 fail, 0 error, 0 skip (exit code 1)"


def test_render_truncates_long_details() -> None:
    report = {**SMALL_REPORT, "outcomes": [_outcome("single-call", "nonstream", "fail", "x" * 5000)]}
    line = next(ln for ln in summary.render(report).splitlines() if ln.startswith("- single-call"))
    assert len(line) < 600 and line.endswith("…")


def test_main_writes_markdown_for_a_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "r.json"
    path.write_text(json.dumps(SMALL_REPORT), encoding="utf-8")
    assert summary.main([str(path), "--exit-code", "1"]) == 0
    assert capsys.readouterr().out == SMALL_MARKDOWN.replace(" 0 skip\n", " 0 skip (exit code 1)\n", 1)


def test_main_without_a_report_says_why(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert summary.main([str(tmp_path / "missing.json"), "--exit-code", "2"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("### CanIToolCall probe: no report\n")
    assert "exited with code 2" in out


def test_render_real_probe_report(mock_openai: Any, tmp_path: Path) -> None:
    from canitoolcall.probe import BUILTIN_SCENARIOS

    mock_openai.override("parallel-calls", stream=True, omit_index=True)
    out = tmp_path / "report.json"
    code = cli.main(["probe", "--base-url", mock_openai.base_url, "--model", "mock-model", "--json", str(out)])
    assert code == 1
    md = summary.render(json.loads(out.read_text(encoding="utf-8")))
    rows = [ln for ln in md.splitlines() if ln.startswith("| ") and not ln.startswith("| scenario")]
    assert [r.split(" | ")[0][2:] for r in rows] == [sc.id for sc in BUILTIN_SCENARIOS]
    assert "| parallel-calls | pass | **fail** | pass |" in rows
    assert md.startswith(f"### CanIToolCall probe: {3 * len(BUILTIN_SCENARIOS) - 1} pass, 1 fail")


# --------------------------------------------------------------------------- standalone mock server


@contextmanager
def mock_server(*args: str, env: dict[str, str] | None = None) -> Iterator[str]:
    proc = subprocess.Popen(
        [sys.executable, str(MOCK_SCRIPT), "--port", "0", *args],
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, **(env or {})},
    )
    try:
        assert proc.stdout is not None
        line = proc.stdout.readline().strip()
        match = re.fullmatch(r"mock OpenAI-compatible server listening on (http://127\.0\.0\.1:\d+/v1)", line)
        assert match, line
        yield match.group(1)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_mock_server_entrypoint_passes_a_probe(tmp_path: Path) -> None:
    with mock_server() as url:
        assert cli.main(["probe", "--base-url", url, "--model", "mock-model", "--json", str(tmp_path / "r.json")]) == 0
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["summary"]["fail"] == report["summary"]["error"] == 0


def test_mock_server_entrypoint_can_break_one_scenario() -> None:
    with mock_server("--omit-index", "parallel-calls") as url:
        assert cli.main(["probe", "--base-url", url, "--model", "mock-model"]) == 1


def test_mock_server_entrypoint_checks_the_bearer_key(monkeypatch: pytest.MonkeyPatch) -> None:
    key = "selftest-not-a-secret"
    with mock_server("--require-key-env", "MOCK_EXPECTED_KEY", env={"MOCK_EXPECTED_KEY": key}) as url:
        monkeypatch.delenv("SELFTEST_KEY", raising=False)
        assert cli.main(["probe", "--base-url", url, "--model", "m", "--api-key-env", "SELFTEST_KEY"]) == 1
        monkeypatch.setenv("SELFTEST_KEY", key)
        assert cli.main(["probe", "--base-url", url, "--model", "m", "--api-key-env", "SELFTEST_KEY"]) == 0


# --------------------------------------------------------------------------- action.yml


def _action() -> str:
    return ACTION.read_text(encoding="utf-8")


def test_action_default_version_is_the_package_version() -> None:
    # Bump action.yml together with __version__ so `@v<x>` installs the matching release.
    m = re.search(r"\n  version:\n(?:    .*\n)*?    default: \"([^\"]+)\"\n", _action())
    assert m, "inputs.version with a quoted default"
    assert m.group(1) == __version__


def test_action_pins_every_action_to_a_commit_sha() -> None:
    uses = re.findall(r"^ *(?:- )?uses: (\S+)", _action(), re.MULTILINE)
    assert uses
    for ref in uses:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", ref), ref


def test_action_run_blocks_never_interpolate_expressions() -> None:
    # Inputs reach the shell only through env:, so a crafted input cannot inject commands
    # and no value is pasted into the script text the runner logs.
    text = _action()
    blocks = re.findall(r"\n( +)run: \|\n((?:\1  .*\n|\n)+)", text)
    assert blocks
    for _, body in blocks:
        assert "${{" not in body, body


# --------------------------------------------------------------------------- action.yml run blocks
#
# The probe and summary steps run here as the runner runs them (`bash --noprofile --norc -eo pipefail`),
# with stand-ins for `uvx` and `uv` that run this checkout's canitoolcall and summary script.


def _run_blocks() -> dict[str, str]:
    blocks = [textwrap.dedent(b) for _, b in re.findall(r"\n( +)run: \|\n((?:\1  .*\n|\n)+)", _action())]
    probe = [b for b in blocks if "uvx " in b]
    summ = [b for b in blocks if "probe_summary.py" in b]
    assert len(probe) == len(summ) == 1
    return {"probe": probe[0], "summary": summ[0]}


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _run_action(tmp_path: Path, base_url: str, *, api_key_env: str = "", stale: bool = True) -> dict[str, Any]:
    """Run the probe and summary steps in ``tmp_path``; return their outputs and the summary."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # uvx --from SPEC canitoolcall ARGS...  ->  python -m canitoolcall ARGS...
    (bin_dir / "uvx").write_text(f'#!/bin/sh\nshift 3\nexec "{sys.executable}" -m canitoolcall "$@"\n')
    # uv run --no-project --python SPEC SCRIPT ARGS...  ->  python SCRIPT ARGS...
    (bin_dir / "uv").write_text(f'#!/bin/sh\nshift 4\nexec "{sys.executable}" "$@"\n')
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    work = tmp_path / "work"
    work.mkdir()
    report = work / "canitoolcall-probe.json"
    if stale:  # left by an earlier step or run that passed
        report.write_text(json.dumps({**SMALL_REPORT, "summary": {"pass": 27}, "outcomes": []}), encoding="utf-8")
    output, step_summary = tmp_path / "output", tmp_path / "summary.md"
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_OUTPUT": str(output),
        "GITHUB_STEP_SUMMARY": str(step_summary),
        "GITHUB_ACTION_PATH": str(ROOT),
        "CTC_BASE_URL": base_url,
        "CTC_MODEL": "mock-model",
        "CTC_KEY_ENV": api_key_env,
        "CTC_ARGS": "",
        "CTC_VERSION": __version__,
        "CTC_PACKAGE": "",
        "CTC_REPORT": report.name,
    }
    blocks = _run_blocks()

    def step(body: str, extra: dict[str, str]) -> subprocess.CompletedProcess[str]:
        cmd = ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", body]
        return subprocess.run(cmd, cwd=work, env={**env, **extra}, capture_output=True, text=True, timeout=120)

    probe = step(blocks["probe"], {})
    assert probe.returncode == 0, probe.stderr  # the step itself passes; a later step fails the job
    outputs = dict(ln.split("=", 1) for ln in output.read_text().splitlines())
    summ = step(blocks["summary"], {"CTC_REPORT": outputs["report"], "CTC_RC": outputs["exit-code"]})
    assert summ.returncode == 0, summ.stderr
    return {
        "outputs": outputs,
        "log": probe.stdout + probe.stderr,
        "summary": step_summary.read_text(encoding="utf-8"),
        "report": report,
    }


needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@needs_bash
def test_action_steps_pass_on_a_conforming_server(mock_openai: Any, tmp_path: Path) -> None:
    r = _run_action(tmp_path, mock_openai.base_url)
    assert r["outputs"]["exit-code"] == "0"
    assert json.loads(r["report"].read_text(encoding="utf-8"))["base_url"] == mock_openai.base_url
    head = r["summary"].splitlines()[0]
    assert re.fullmatch(r"### CanIToolCall probe: \d+ pass, 0 fail, 0 error, 0 skip \(exit code 0\)", head), head


@needs_bash
def test_action_steps_do_not_show_a_stale_report_when_unreachable(tmp_path: Path) -> None:
    r = _run_action(tmp_path, f"http://127.0.0.1:{_closed_port()}/v1")
    assert r["outputs"]["exit-code"] == "2"
    assert "cannot reach" in r["log"]
    assert not r["report"].exists()
    assert r["summary"].startswith("### CanIToolCall probe: no report\n")
    assert "exited with code 2" in r["summary"]


@needs_bash
def test_action_steps_do_not_show_a_stale_report_for_a_bad_key_name(tmp_path: Path) -> None:
    pasted = "sk-pasted-not-a-real-key"
    r = _run_action(tmp_path, "http://127.0.0.1:1/v1", api_key_env=pasted)
    assert r["outputs"]["exit-code"] == "2"
    assert "api-key-env must be the NAME" in r["log"]
    assert pasted not in r["log"] and pasted not in r["summary"]
    assert not r["report"].exists()
    assert r["summary"].startswith("### CanIToolCall probe: no report\n")

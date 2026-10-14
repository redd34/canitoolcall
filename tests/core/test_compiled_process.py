"""Small owned processes, expiring automatically even if the tested code fails."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from compiled_process import run_logged  # noqa: E402

pytestmark = pytest.mark.skipif(os.name != "posix", reason="compiled process management requires POSIX")

TREE = """import os, pathlib, signal, subprocess, sys, time
root=pathlib.Path(sys.argv[1]); role=sys.argv[2]; mode=sys.argv[3]
if mode == 'stubborn': signal.signal(signal.SIGTERM, signal.SIG_IGN)
(root/(role+'.pid')).write_text(str(os.getpid()))
if role!='grandchild':
    nextrole='child' if role=='parent' else 'grandchild'
    subprocess.Popen([sys.executable,"-S",__file__,str(root),nextrole,mode])
if role=='parent' and mode=='early_exit':
    deadline=time.monotonic()+2
    while not (root/'grandchild.pid').exists() and time.monotonic()<deadline: time.sleep(.01)
    sys.exit(0)
deadline=time.monotonic()+6
while time.monotonic()<deadline:
    (root/(role+'.beat')).write_text(str(time.monotonic_ns()))
    time.sleep(.02)
"""

DRIVER = """import json, os, pathlib, signal, subprocess, sys, time
sys.path.insert(0,sys.argv[1])
from compiled_process import run_logged, VerificationInterrupted
root=pathlib.Path(sys.argv[2]); mode=sys.argv[3]; code=99
command=[sys.executable,"-S",str(root/'tree.py'),str(root),'parent',mode]
try:
    with (root/'command.log').open('w') as log:
        code=run_logged(command,cwd=root,env=os.environ.copy(),log=log,timeout=2,grace=.15).returncode
except subprocess.TimeoutExpired: code=3
except VerificationInterrupted as exc: code=128+exc.signum
finally:
    (root/'outcome.json').write_text(json.dumps({'code':code}))
# Remain a known group leader for the test fixture's own final cleanup.
time.sleep(8)
"""


def alive(pid):
    if sys.platform.startswith("linux"):
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            return False
        return state not in ("Z", "X")
    p = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False)
    return p.returncode == 0 and bool(p.stdout.strip()) and not p.stdout.lstrip().startswith("Z")


def until(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    assert predicate(), "bounded wait did not complete"


@pytest.mark.parametrize(
    ("mode", "signum", "code"),
    [
        ("timeout", None, 3),
        ("stubborn", None, 3),
        ("early_exit", None, 0),
        ("timeout", signal.SIGINT, 130),
        ("stubborn", signal.SIGTERM, 143),
    ],
)
def test_real_descendants_stop_and_unrelated_process_survives(tmp_path, mode, signum, code):
    (tmp_path / "tree.py").write_text(TREE)
    (tmp_path / "driver.py").write_text(DRIVER)
    driver = subprocess.Popen(
        [sys.executable, "-S", str(tmp_path / "driver.py"), str(SCRIPTS), str(tmp_path), mode],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    unrelated = subprocess.Popen([sys.executable, "-S", "-c", "import time;time.sleep(8)"], start_new_session=True)
    try:
        until(lambda: (tmp_path / "grandchild.pid").exists())
        if signum is not None:
            driver.send_signal(signum)
        until(lambda: (tmp_path / "outcome.json").exists())
        assert json.loads((tmp_path / "outcome.json").read_text())["code"] == code
        pids = [int((tmp_path / f"{role}.pid").read_text()) for role in ["parent", "child", "grandchild"]]
        until(lambda: all(not alive(pid) for pid in pids))
        assert unrelated.poll() is None
    finally:
        # These are only the retained, direct child sessions created above.
        for owned in [driver, unrelated]:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(owned.pid, signal.SIGKILL)
            owned.wait(timeout=5)


@pytest.mark.parametrize("code", [0, 7])
def test_exit_and_large_output_are_preserved(tmp_path, code):
    before = {s: signal.getsignal(s) for s in [signal.SIGINT, signal.SIGTERM]}
    with (tmp_path / "output.log").open("w") as log:
        result = run_logged(
            [sys.executable, "-S", "-c", f'import sys;print("x"*100000);sys.exit({code})'],
            cwd=tmp_path,
            env=os.environ.copy(),
            log=log,
            timeout=3,
            grace=0.01,
        )
    assert result.returncode == code
    assert len((tmp_path / "output.log").read_text()) == 100001
    assert before == {s: signal.getsignal(s) for s in before}


def test_missing_command_is_an_error_and_restores_handlers(tmp_path):
    before = {s: signal.getsignal(s) for s in [signal.SIGINT, signal.SIGTERM]}
    with (tmp_path / "output.log").open("w") as log, pytest.raises(OSError):
        run_logged([str(tmp_path / "missing")], cwd=tmp_path, env=os.environ.copy(), log=log, timeout=3, grace=0.01)
    assert before == {s: signal.getsignal(s) for s in before}


@pytest.mark.parametrize(("timeout", "grace"), [(0, 0.1), (-1, 0.1), (float("inf"), 0.1), (1, -1)])
def test_bad_time_limits_never_start_a_command(tmp_path, monkeypatch, timeout, grace):
    def forbidden(*args, **kwargs):
        raise AssertionError("should not start")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with (tmp_path / "log").open("w") as log, pytest.raises(ValueError):
        run_logged(["ignored"], cwd=tmp_path, env={}, log=log, timeout=timeout, grace=grace)

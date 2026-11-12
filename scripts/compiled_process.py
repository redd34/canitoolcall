"""Run one POSIX build with a private, retained process-group leader.

No global process enumeration or unrelated PID is used. Descendants that
leave this group, an uncatchable supervisor kill and OS failure are outside
this contract. The small leader also works on Python 3.10/macOS, where
os.waitid(WNOWAIT) is unavailable.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TextIO


class VerificationInterrupted(BaseException):
    """A handled cancellation, reported after private-group cleanup."""

    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"verification interrupted by signal {signum}")


def _leader(output_fd: int, command: list[str]) -> None:
    # Python callbacks (rather than SIG_IGN) reset to default on child exec.
    # Keep the leader alive until the caller has signalled the whole group.
    signal.signal(signal.SIGTERM, lambda *_: None)
    signal.signal(signal.SIGINT, lambda *_: None)
    try:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output_fd, stderr=subprocess.STDOUT)
        result = {"returncode": child.wait()}
    except OSError as exc:
        result = {"error": str(exc), "errno": exc.errno}
    print(json.dumps(result), flush=True)
    while True:
        signal.pause()


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log: TextIO,
    timeout: float = 900,
    grace: float = 0.25,
) -> subprocess.CompletedProcess[str]:
    """Clean the private group after success, failure, timeout or SIGINT/TERM."""
    if os.name != "posix" or threading.current_thread() is not threading.main_thread():
        raise RuntimeError("compiled verification requires a POSIX main thread; use Linux, macOS or WSL")
    if not command or not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(grace) or grace < 0:
        raise ValueError("command and finite positive timeout/nonnegative grace are required")
    output_fd = log.fileno()
    log.flush()
    interrupted: list[int] = []

    def remember(signum: int, _frame: object) -> None:
        if not interrupted:
            interrupted.append(signum)

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    proc = None
    result = None
    try:
        for sig in previous:
            signal.signal(sig, remember)
        proc = subprocess.Popen(
            [sys.executable, "-I", "-S", "-u", str(Path(__file__).resolve()), str(output_fd), *command],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            pass_fds=(output_fd,),
        )
        assert proc.stdout is not None
        deadline = time.monotonic() + timeout
        received = b""
        while b"\n" not in received:
            if interrupted:
                raise VerificationInterrupted(interrupted[0])
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            ready, _, _ = select.select([proc.stdout], [], [], min(remaining, 0.05))
            if ready:
                chunk = os.read(proc.stdout.fileno(), 4096)
                if not chunk:
                    raise RuntimeError("build supervisor ended without reporting command completion")
                received += chunk
                if len(received) > 65536:
                    raise RuntimeError("oversized build-supervisor completion record")
        result = json.loads(received.partition(b"\n")[0])
        if "error" in result:
            raise OSError(result.get("errno"), result["error"])
        if not isinstance(result.get("returncode"), int):
            raise RuntimeError("invalid build-supervisor completion record")
    finally:
        try:
            if proc is not None:
                # Never poll/reap this leader before signalling the group. Its
                # retained PID cannot be recycled while cleanup is in progress.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(grace)
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                try:
                    proc.wait(timeout=5)
                finally:
                    if proc.stdout is not None:
                        proc.stdout.close()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    if interrupted:
        raise VerificationInterrupted(interrupted[0])
    assert result is not None
    return subprocess.CompletedProcess(command, result["returncode"])


if __name__ == "__main__":
    _leader(int(sys.argv[1]), sys.argv[2:])

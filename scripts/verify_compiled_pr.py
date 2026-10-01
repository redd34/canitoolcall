"""Isolated complete-source builds for compiled parser PR verification.

The ordinary pinned engine, worker environment and adapter pin stay unchanged.
Implementation by Zero with Youngseok Oh.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from compiled_process import VerificationInterrupted, run_logged

REPOSITORIES = {"ollama": "ollama/ollama", "llamacpp": "ggml-org/llama.cpp"}
ADAPTERS = {"ollama": "ollama:OllamaAdapter", "llamacpp": "llamacpp:LlamaCppAdapter"}
STATUSES = {"pass", "soft_pass", "fail", "error", "unsupported"}
GOOD = {"pass", "soft_pass"}
BAD = {"fail", "error"}


def commit_sha(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise RuntimeError(f"expected a complete lowercase commit SHA, got {value!r}")
    return value


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
        return digest.hexdigest()


def pinned_binaries(root: Path, engine: str) -> dict[str, str]:
    directory = root / ".engines" / engine
    names = ["bin/ctcreplay", "bin/ctc-detok"] if engine == "ollama" else ["build/canitoolcall-llamacpp"]
    return {name: file_digest(directory / name) for name in names if (directory / name).is_file()}


def build(root: Path, engine: str, sha: str, destination: Path, jobs: int) -> tuple[dict[str, str], dict[str, str]]:
    """Copy trusted build recipes, not the pinned engine, into a fresh workspace."""
    commit_sha(sha)
    project = destination / "project"
    project.mkdir(parents=True, exist_ok=False)
    shutil.copytree(root / "scripts" / "engines", project / "scripts" / "engines")
    shutil.copytree(root / "harnesses" / engine, project / "harnesses" / engine)
    env = os.environ.copy()
    env.update({f"{engine.upper()}_REF": sha, "JOBS": str(jobs)})
    if engine == "llamacpp":
        env["LLAMACPP_REPO"] = "https://github.com/ggml-org/llama.cpp"
    command = ["bash", str(project / "scripts" / "engines" / f"{engine}.sh"), "--no-venv"]
    if engine == "llamacpp":
        command.append("--no-vocab")
    with (destination / "build.log").open("w", encoding="utf-8") as log:
        result = run_logged(command, cwd=project, env=env, log=log)
    if result.returncode:
        raise RuntimeError(f"{engine} build exited {result.returncode}; see {destination / 'build.log'}")
    engine_root = project / ".engines" / engine
    checkout = engine_root / ("ollama" if engine == "ollama" else "llama.cpp")
    actual = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if actual != sha:
        raise RuntimeError(f"built checkout is {actual!r}, expected {sha}")
    if engine == "ollama":
        binaries = {name: engine_root / "bin" / name for name in ("ctcreplay", "ctc-detok")}
        overrides = {"CANITOOLCALL_OLLAMA_BIN_DIR": str(engine_root / "bin")}
    else:
        binaries = {"canitoolcall-llamacpp": engine_root / "build" / "canitoolcall-llamacpp"}
        overrides = {
            "CANITOOLCALL_LLAMACPP_HARNESS": str(binaries["canitoolcall-llamacpp"]),
            "CANITOOLCALL_LLAMACPP_SRC": str(checkout),
        }
    hashes = {name: file_digest(path) for name, path in binaries.items()}
    # An explicitly selected adapter checks the built commit. Never weaken the
    # normal adapter's pin or overwrite its source to accommodate a PR build.
    module, cls = ADAPTERS[engine].split(":")
    (destination / "ctc_revision_adapter.py").write_text(
        f"from canitoolcall.adapters.{module} import {cls}\n\n"
        f"class RevisionAdapter({cls}):\n    pinned_version = {sha!r}\n",
        encoding="utf-8",
    )
    overrides["PYTHONPATH"] = str(destination)
    overrides["CANITOOLCALL_GGUF_DIR"] = os.environ.get("CANITOOLCALL_GGUF_DIR", str(root / ".engines" / "gguf"))
    return overrides, hashes


def read_cases(out: Path, exit_code: int) -> dict[str, str]:
    if exit_code not in (0, 1):
        raise RuntimeError(f"replay exited {exit_code}; see replay.log")
    found = sorted(out.glob("*.json"))
    if len(found) != 1:
        raise RuntimeError(f"expected one results file, found {len(found)}")
    value = json.loads(found[0].read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list) or not value["cases"]:
        raise RuntimeError("replay needs a nonempty cases list")
    cases: dict[str, str] = {}
    for case in value["cases"]:
        if not isinstance(case, dict):
            raise RuntimeError("invalid case object")
        key, status = case.get("fixture_id"), case.get("status")
        if not isinstance(key, str) or not key.strip() or key in cases:
            raise RuntimeError("missing or duplicate fixture ID")
        if not isinstance(status, str) or status not in STATUSES:
            raise RuntimeError(f"invalid status for {key!r}")
        if status == "error" or case.get("harness_error"):
            raise RuntimeError(f"harness failed for {key!r}; see the per-fixture result")
        cases[key] = status
    if all(status == "unsupported" for status in cases.values()):
        raise RuntimeError("no supported fixtures were replayed")
    return cases


def replay(root: Path, families: list[str], destination: Path, env: dict[str, str]) -> dict[str, str]:
    out = destination / "results"
    out.mkdir()
    command = [sys.executable, "-m", "canitoolcall", "run", "--engine", "ctc_revision_adapter:RevisionAdapter"]
    command += ["--out", str(out), "--env", "HF_HUB_OFFLINE=1"]
    for key, value in env.items():
        command += ["--env", f"{key}={value}"]
    for family in families:
        command += ["--family", family]
    process_env = os.environ.copy()
    process_env.update(env)
    process_env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), env["PYTHONPATH"]])
    with (destination / "replay.log").open("w", encoding="utf-8") as log:
        result = run_logged(command, cwd=root, env=process_env, log=log)
    return read_cases(out, result.returncode)


def compare(base: dict[str, str], head: dict[str, str]) -> dict[str, object]:
    if base.keys() != head.keys():
        raise RuntimeError("base and head fixture inventories differ; comparison is incomplete")
    if any((base[key] == "unsupported") != (head[key] == "unsupported") for key in base):
        raise RuntimeError("support coverage changed; comparison is incomplete")
    changed = sorted((key, base[key], head[key]) for key in base if base[key] != head[key])
    fixed = sum(old in BAD and new in GOOD for _, old, new in changed)
    regressed = sum(old in GOOD and new in BAD for _, old, new in changed)
    return {"fixtures": len(base), "changed": changed, "fixed": fixed, "regressed": regressed}


def report_destination(root: Path, output: Path | None) -> Path | None:
    """Do not let a report path alter a pinned engine, venv or existing file."""
    if output is None:
        return None
    destination = output.expanduser().resolve()
    for name in (".engines", ".venvs", ".git"):
        if destination.is_relative_to((root / name).resolve()):
            raise ValueError(f"report output is inside protected {name}; choose a separate new file")
    if destination.exists():
        raise FileExistsError(f"report output already exists: {destination}; choose a new file")
    return destination


def write_report(path: Path, encoded: str, *, replace: bool) -> None:
    """Publish a fully written same-directory file, never a partial export."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
        if replace:
            os.replace(temporary, path)
        else:
            # Atomic no-clobber publication: an existing target is never replaced.
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def finish_reports(report: dict[str, object], code: int, work: Path, output: Path | None) -> int:
    """Report I/O failures explicitly, with stderr as the last-resort receipt."""
    report["exit_code"] = code
    local = work / "verification.json"
    try:
        write_report(local, json.dumps(report, indent=2) + "\n", replace=True)
        if output is not None:
            write_report(output, json.dumps(report, indent=2) + "\n", replace=False)
        return code
    except OSError as exc:
        report.update(status="report_error", operation_exit_code=code, report_error=str(exc), exit_code=3)
        try:
            write_report(local, json.dumps(report, indent=2) + "\n", replace=True)
        except OSError as persistence_error:
            report["report_persistence_error"] = str(persistence_error)
            print("VERIFICATION_REPORT_JSON " + json.dumps(report), file=sys.stderr)
        return 3


def verify(root: Path, engine: str, pr_number: int, families: list[str], output: Path | None, jobs: int) -> int:
    if engine not in REPOSITORIES or pr_number <= 0 or jobs <= 0:
        raise ValueError("expected a compiled engine, positive PR number and positive build job count")
    try:
        output = report_destination(root, output)
    except (OSError, ValueError) as exc:
        print(f"invalid report output: {exc}", file=sys.stderr)
        return 3
    repo = REPOSITORIES[engine]
    work = Path(tempfile.mkdtemp(prefix=f"verify-{engine}-{pr_number}-"))
    report: dict[str, object] = {"engine": engine, "repo": repo, "pr": pr_number, "work": str(work)}
    before = pinned_binaries(root, engine)
    code = 3
    try:
        proc = subprocess.run(
            ["gh", "api", f"repos/{repo}/pulls/{pr_number}"], check=True, capture_output=True, text=True, timeout=60
        )
        pr = json.loads(proc.stdout)
        base, head = commit_sha(pr["base"]["sha"]), commit_sha(pr["head"]["sha"])
        report.update(base=base, head=head)
        results: dict[str, dict[str, str]] = {}
        builds: dict[str, dict[str, str]] = {}
        for side, sha in (("base", base), ("head", head)):
            destination = work / side
            overrides, builds[side] = build(root, engine, sha, destination, jobs)
            results[side] = replay(root, families, destination, overrides)
        report["binary_sha256"] = builds
        report.update(compare(results["base"], results["head"]))
        report["status"] = "compared"
        code = 1 if report["regressed"] else 0
    except (KeyboardInterrupt, VerificationInterrupted) as exc:
        signum = exc.signum if isinstance(exc, VerificationInterrupted) else 2
        report.update(status="interrupted", error=f"verification interrupted by signal {signum}")
        code = 128 + signum
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
        report.update(status="unreplayable", error=str(exc))
    finally:
        after = pinned_binaries(root, engine)
        report.update(pinned_binaries_before=before, pinned_binaries_after=after, pinned_unchanged=before == after)
        if before != after:
            report.update(status="unreplayable", error="pinned harness changed while verification was running")
            code = 3
    code = finish_reports(report, code, work, output)
    if code not in (0, 1):
        print(
            f"{repo}#{pr_number} cannot be replayed: {report.get('report_error', report.get('error'))}", file=sys.stderr
        )
    else:
        print(f"{repo}#{pr_number}  base {base[:9]} -> head {head[:9]}")
        print("(two isolated full-source builds; pinned harness unchanged)")
        print(f"fixtures replayed: {report['fixtures']}   fixed: {report['fixed']}   regressed: {report['regressed']}")
        for key, old, new in report["changed"]:  # type: ignore[union-attr]
            print(f"  {key}: {old} -> {new}")
    print(f"verification logs: {work}", file=sys.stderr)
    return code

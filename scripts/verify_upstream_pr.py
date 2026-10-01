#!/usr/bin/env python3
"""Verify an upstream parser fix PR against the canitoolcall fixtures.

Swaps the Python files that an upstream pull request changes into the pinned
engine environment, replays the affected fixture families twice (at the PR's
base commit and at its head commit), restores the pinned files, and prints
the per-fixture difference.

Python engines use changed-file overlays. Ollama (Go) and llama.cpp (C++)
build complete base/head sources in separate temporary directories. Their
normal pinned builds are untouched; existing vocab-only GGUFs are reused.
Compiled builds require bash, git, Go/CMake/C++ as documented in AGENTS.md.

Usage (from the repo root, after `bash scripts/engines/<engine>.sh`):

    uv run python scripts/verify_upstream_pr.py --engine sglang --pr 41319 --family gemma4
    uv run python scripts/verify_upstream_pr.py --engine vllm --pr 58829 --family llama --json report.json

Needs the GitHub CLI (`gh`), authenticated, to read the PR. Only the PR's
changed files are swapped in, so a PR that depends on other unreleased
changes can fail to import. The script reports that as an error rather than
a result.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import verify_compiled_pr

ROOT = Path(__file__).resolve().parent.parent
ENGINES = {
    # engine: (GitHub repo, top-level package, path prefix of the package inside the repo)
    "vllm": ("vllm-project/vllm", "vllm", "vllm/"),
    "sglang": ("sgl-project/sglang", "sglang", "python/sglang/"),
    "transformers": ("huggingface/transformers", "transformers", "src/transformers/"),
}


def gh_json(*args: str) -> object:
    out = subprocess.run(["gh", "api", *args], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def gh_raw(repo: str, path: str, ref: str) -> bytes | None:
    r = subprocess.run(
        ["gh", "api", f"repos/{repo}/contents/{path}?ref={ref}", "-H", "Accept: application/vnd.github.raw"],
        capture_output=True,
    )
    return r.stdout if r.returncode == 0 else None


def package_root(engine: str, package: str) -> Path:
    py = next((ROOT / ".venvs" / engine / "bin").glob("python3*"), None) or ROOT / ".venvs" / engine / "bin" / "python"
    code = f"import importlib.util; print(list(importlib.util.find_spec({package!r}).submodule_search_locations)[0])"
    out = subprocess.run([str(py), "-c", code], check=True, capture_output=True, text=True).stdout.strip()
    return Path(out)


def replay(engine: str, families: list[str], out_dir: Path) -> dict[str, str]:
    cmd = ["uv", "run", "-q", "canitoolcall", "run", "--engine", engine]
    cmd += ["--env", "HF_HUB_OFFLINE=1", "--out", str(out_dir)]
    for fam in families:
        cmd += ["--family", fam]
    # `canitoolcall run` exits 1 when any fixture fails, which is expected here;
    # only a missing results file means the replay itself broke.
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    found = list(out_dir.glob("*.json"))
    if not found:
        raise RuntimeError(proc.stderr[-2000:] or proc.stdout[-2000:])
    result = json.loads(found[0].read_text())
    return {c["fixture_id"]: c["status"] for c in result["cases"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, choices=sorted(set(ENGINES) | set(verify_compiled_pr.REPOSITORIES)))
    ap.add_argument("--pr", required=True, type=int, help="upstream pull request number")
    ap.add_argument("--family", action="append", default=[], help="fixture family to replay (repeatable; default: all)")
    ap.add_argument("--json", type=Path, help="also write the report as JSON")
    ap.add_argument("--build-jobs", type=int, default=2, help="parallel compiled build jobs (default: 2)")
    a = ap.parse_args()
    if a.pr <= 0 or a.build_jobs <= 0:
        ap.error("--pr and --build-jobs must be positive")
    if a.engine in verify_compiled_pr.REPOSITORIES:
        return verify_compiled_pr.verify(ROOT, a.engine, a.pr, a.family, a.json, a.build_jobs)

    repo, package, prefix = ENGINES[a.engine]
    pr = gh_json(f"repos/{repo}/pulls/{a.pr}")
    base, head = pr["base"]["sha"], pr["head"]["sha"]  # type: ignore[index]
    files = [
        f["filename"]
        for f in gh_json(f"repos/{repo}/pulls/{a.pr}/files", "--paginate")  # type: ignore[union-attr]
        if f["filename"].startswith(prefix) and f["filename"].endswith(".py") and f["status"] != "removed"
    ]
    if not files:
        print(f"{repo}#{a.pr} changes no Python files under {prefix}; nothing to verify.", file=sys.stderr)
        return 2

    root = package_root(a.engine, package)
    targets = {f: root / f[len(prefix) :] for f in files}
    work = Path(tempfile.mkdtemp(prefix=f"verify-{a.engine}-{a.pr}-"))
    backup = {f: (work / "pinned" / f) for f in files}
    report: dict[str, object] = {
        "engine": a.engine,
        "repo": repo,
        "pr": a.pr,
        "base": base,
        "head": head,
        "files": files,
    }
    try:
        for f, t in targets.items():
            backup[f].parent.mkdir(parents=True, exist_ok=True)
            if t.exists():
                shutil.copy2(t, backup[f])
        results = {}
        for side, sha in (("base", base), ("head", head)):
            for f, t in targets.items():
                data = gh_raw(repo, f, sha)
                if data is None:  # file added by the PR: absent at base
                    t.unlink(missing_ok=True)
                else:
                    t.parent.mkdir(parents=True, exist_ok=True)
                    t.write_bytes(data)
            out = work / side
            out.mkdir()
            try:
                results[side] = replay(a.engine, a.family, out)
            except RuntimeError as e:
                print(f"replay at {side} ({sha[:9]}) failed; the PR may depend on changes", file=sys.stderr)
                print(f"that are not in the pinned engine.\n{e}", file=sys.stderr)
                return 3
    finally:
        for f, t in targets.items():
            if backup[f].exists():
                shutil.copy2(backup[f], t)
            else:
                t.unlink(missing_ok=True)

    b, h = results["base"], results["head"]
    changed = sorted((k, b[k], h.get(k, "missing")) for k in b if b[k] != h.get(k))
    fixed = [c for c in changed if c[1] in ("fail", "error") and c[2] in ("pass", "soft_pass")]
    regressed = [c for c in changed if c[1] in ("pass", "soft_pass") and c[2] in ("fail", "error")]
    report.update(fixtures=len(b), changed=changed, fixed=len(fixed), regressed=len(regressed))

    other = len(changed) - len(fixed) - len(regressed)
    print(f"{repo}#{a.pr}  base {base[:9]} -> head {head[:9]}")
    print(f"({len(files)} file(s) swapped into the pinned {a.engine})")
    print(f"fixtures replayed: {len(b)}   fixed: {len(fixed)}   regressed: {len(regressed)}   other changes: {other}")
    for k, s0, s1 in changed:
        print(f"  {k}: {s0} -> {s1}")
    if a.json:
        a.json.write_text(json.dumps(report, indent=2))
    return 1 if regressed else 0


if __name__ == "__main__":
    sys.exit(main())

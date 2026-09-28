"""Verifier contracts with synthetic process output, never a real engine or API."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

BASE = "a" * 40
HEAD = "b" * 40
FILES = ["python/sglang/detector.py", "python/sglang/added.py"]


@dataclass(frozen=True)
class ReplayOutput:
    reports: tuple[bytes, ...]
    returncode: int = 0


def document(value: object, returncode: int = 0) -> ReplayOutput:
    return ReplayOutput((json.dumps(value).encode("utf-8"),), returncode)


def cases(*entries: tuple[str, str], returncode: int = 0) -> ReplayOutput:
    # Only the report fields consumed by the verifier; these are not engine fixtures.
    return document({"cases": [{"fixture_id": key, "status": status} for key, status in entries]}, returncode)


@dataclass(frozen=True)
class Invocation:
    code: int
    stdout: str
    stderr: str
    report: Path
    replayed: tuple[str, ...]


RunVerifier = Callable[[ReplayOutput, ReplayOutput], Invocation]


@pytest.fixture
def verifier() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "verify_upstream_pr.py"
    spec = importlib.util.spec_from_file_location("verify_upstream_pr", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def run_verifier(
    verifier: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> RunVerifier:
    root = tmp_path / "engine"
    root.mkdir()
    original = b"# pinned parser\n\xff"
    (root / "detector.py").write_bytes(original)
    (root / "unrelated.py").write_bytes(b"# do not touch\n")
    pinned = {p.name: p.read_bytes() for p in root.iterdir()}
    report = tmp_path / "comparison.json"

    def gh_json(*args: str) -> object:
        if args == ("repos/sgl-project/sglang/pulls/1",):
            return {"base": {"sha": BASE}, "head": {"sha": HEAD}}
        assert args == ("repos/sgl-project/sglang/pulls/1/files", "--paginate")
        return [{"filename": name, "status": "modified" if i == 0 else "added"} for i, name in enumerate(FILES)]

    def gh_raw(repo: str, path: str, ref: str) -> bytes | None:
        assert repo == "sgl-project/sglang" and path in FILES and ref in (BASE, HEAD)
        if path == FILES[1] and ref == BASE:
            return None
        return f"# {ref}: {path}\n".encode()

    def package_root(engine: str, package: str) -> Path:
        assert (engine, package) == ("sglang", "sglang")
        return root

    monkeypatch.setattr(verifier, "gh_json", gh_json)
    monkeypatch.setattr(verifier, "gh_raw", gh_raw)
    monkeypatch.setattr(verifier, "package_root", package_root)
    # Keep real tempfile allocation and real backup/restore I/O inside pytest's sandbox.
    monkeypatch.setattr(verifier.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify", "--engine", "sglang", "--pr", "1", "--family", "fam", "--json", str(report)],
    )

    def invoke(before: ReplayOutput, after: ReplayOutput) -> Invocation:
        replayed: list[str] = []

        def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            assert cmd[:5] == ["uv", "run", "-q", "canitoolcall", "run"]
            assert cmd[cmd.index("--engine") + 1] == "sglang"
            assert cmd[cmd.index("--family") + 1] == "fam"
            assert cmd[cmd.index("--env") + 1] == "HF_HUB_OFFLINE=1"
            assert kwargs["cwd"] == verifier.ROOT
            out = Path(cmd[cmd.index("--out") + 1])
            side = ("base", "head")[len(replayed)]
            assert out.name == side
            ref = BASE if side == "base" else HEAD
            assert (root / "detector.py").read_bytes() == gh_raw("sgl-project/sglang", FILES[0], ref)
            if side == "base":
                assert not (root / "added.py").exists()
            else:
                assert (root / "added.py").read_bytes() == gh_raw("sgl-project/sglang", FILES[1], ref)
            assert (root / "unrelated.py").read_bytes() == pinned["unrelated.py"]
            replayed.append(side)
            output = before if side == "base" else after
            for i, payload in enumerate(output.reports):
                (out / f"report-{i}.json").write_bytes(payload)
            return subprocess.CompletedProcess(cmd, output.returncode, "synthetic replay\n", "synthetic diagnostic\n")

        monkeypatch.setattr(verifier.subprocess, "run", run)
        try:
            code = verifier.main()
        finally:
            # Applies to success, regression, and exceptions at either replay stage.
            assert {p.name: p.read_bytes() for p in root.iterdir()} == pinned
        captured = capsys.readouterr()
        return Invocation(code, captured.out, captured.err, report, tuple(replayed))

    return invoke


@pytest.mark.parametrize(
    ("before", "after", "code", "changed", "fixed", "regressed"),
    [
        pytest.param(cases(("fam/a", "pass")), cases(("fam/a", "pass")), 0, [], 0, 0, id="unchanged"),
        pytest.param(
            cases(("fam/a", "pass"), ("fam/b", "soft_pass")),
            cases(("fam/b", "error"), ("fam/a", "fail"), returncode=1),
            1,
            [["fam/a", "pass", "fail"], ["fam/b", "soft_pass", "error"]],
            0,
            2,
            id="regressions-and-reordered-inventory",
        ),
        pytest.param(
            cases(("fam/a", "fail"), ("fam/b", "error"), returncode=1),
            cases(("fam/a", "pass"), ("fam/b", "soft_pass")),
            0,
            [["fam/a", "fail", "pass"], ["fam/b", "error", "soft_pass"]],
            2,
            0,
            id="improvements",
        ),
        pytest.param(
            cases(("fam/a", "fail"), ("fam/b", "error"), ("fam/c", "unsupported"), returncode=1),
            cases(("fam/a", "fail"), ("fam/b", "error"), ("fam/c", "unsupported"), returncode=1),
            0,
            [],
            0,
            0,
            id="existing-failures-and-unsupported",
        ),
        pytest.param(
            cases(("fam/a", "pass"), ("fam/b", "unsupported")),
            cases(("fam/a", "soft_pass"), ("fam/b", "pass")),
            0,
            [["fam/a", "pass", "soft_pass"], ["fam/b", "unsupported", "pass"]],
            0,
            0,
            id="other-changes",
        ),
    ],
)
def test_valid_comparison_preserves_cli_contract(
    run_verifier: RunVerifier,
    before: ReplayOutput,
    after: ReplayOutput,
    code: int,
    changed: list[list[str]],
    fixed: int,
    regressed: int,
) -> None:
    result = run_verifier(before, after)
    count = len(json.loads(before.reports[0])["cases"])
    assert result.code == code
    assert result.replayed == ("base", "head")
    assert result.stderr == ""
    assert result.stdout == (
        f"sgl-project/sglang#1  base {BASE[:9]} -> head {HEAD[:9]}\n"
        "(2 file(s) swapped into the pinned sglang)\n"
        f"fixtures replayed: {count}   fixed: {fixed}   regressed: {regressed}   "
        f"other changes: {len(changed) - fixed - regressed}\n"
        + "".join(f"  {key}: {old} -> {new}\n" for key, old, new in changed)
    )
    assert json.loads(result.report.read_text()) == {
        "engine": "sglang",
        "repo": "sgl-project/sglang",
        "pr": 1,
        "base": BASE,
        "head": HEAD,
        "files": FILES,
        "fixtures": count,
        "changed": changed,
        "fixed": fixed,
        "regressed": regressed,
    }


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(cases(("fam/a", "pass"), ("fam/b", "pass")), cases(("fam/a", "pass")), id="missing"),
        pytest.param(cases(("fam/a", "pass")), cases(("fam/a", "pass"), ("fam/b", "fail")), id="added"),
        pytest.param(cases(("fam/a", "pass")), cases(("fam/b", "pass")), id="same-size-different-ids"),
    ],
)
def test_incomparable_inventories_are_not_reported_as_success(
    run_verifier: RunVerifier, before: ReplayOutput, after: ReplayOutput
) -> None:
    result = run_verifier(before, after)
    assert result.code == 3
    assert result.replayed == ("base", "head")
    assert result.stderr and "Traceback" not in result.stderr
    assert result.stdout == ""
    assert not result.report.exists()


@pytest.mark.parametrize("side", ["base", "head"])
@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(ReplayOutput(()), id="missing-report"),
        pytest.param(ReplayOutput(cases(("fam/a", "pass")).reports * 2), id="multiple-reports"),
        pytest.param(cases(("fam/a", "pass"), returncode=2), id="process-error-with-json"),
        pytest.param(cases(("fam/a", "pass"), returncode=-9), id="terminated-with-json"),
        pytest.param(ReplayOutput((b"{",)), id="malformed-json"),
        pytest.param(ReplayOutput((b"\xff",)), id="non-utf8"),
        pytest.param(document([]), id="report-not-object"),
        pytest.param(document({}), id="missing-cases"),
        pytest.param(document({"cases": {}}), id="cases-not-list"),
        pytest.param(document({"cases": None}), id="null-cases"),
        pytest.param(cases(), id="empty-cases"),
        pytest.param(document({"cases": ["fam/a"]}), id="case-not-object"),
        pytest.param(document({"cases": [{"status": "pass"}]}), id="missing-id"),
        pytest.param(cases(("", "pass")), id="empty-id"),
        pytest.param(cases((" \t", "pass")), id="blank-id"),
        pytest.param(document({"cases": [{"fixture_id": 1, "status": "pass"}]}), id="numeric-id"),
        pytest.param(document({"cases": [{"fixture_id": None, "status": "pass"}]}), id="null-id"),
        pytest.param(document({"cases": [{"fixture_id": [], "status": "pass"}]}), id="unhashable-id"),
        pytest.param(cases(("fam/a", "pass"), ("fam/a", "pass")), id="duplicate-id-same-status"),
        pytest.param(cases(("fam/a", "fail"), ("fam/a", "pass")), id="duplicate-id-different-status"),
        pytest.param(document({"cases": [{"fixture_id": "fam/a"}]}), id="missing-status"),
        pytest.param(cases(("fam/a", "unknown")), id="unknown-status"),
        pytest.param(document({"cases": [{"fixture_id": "fam/a", "status": []}]}), id="non-string-status"),
    ],
)
def test_invalid_replay_returns_3_and_restores_files(run_verifier: RunVerifier, bad: ReplayOutput, side: str) -> None:
    good = cases(("fam/a", "pass"))
    result = run_verifier(bad, good) if side == "base" else run_verifier(good, bad)
    assert result.code == 3
    assert result.replayed == (("base",) if side == "base" else ("base", "head"))
    assert result.stderr and side in result.stderr and "Traceback" not in result.stderr
    assert result.stdout == ""
    assert not result.report.exists()

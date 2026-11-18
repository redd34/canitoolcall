"""Replay report checks shared by both upstream-PR verifiers.

`verify_upstream_pr.py` (Python overlays) and `verify_compiled_pr.py` (full
compiled builds) both replay fixtures at a PR's base and head with
`canitoolcall run`. Neither may report a comparison unless each side wrote one
complete results file and both sides cover the same fixtures. A ReportError
is a RuntimeError, which both verifiers turn into exit code 3. Stdlib only:
the Python verifier imports this before any engine environment.
"""

from __future__ import annotations

import json
from pathlib import Path

STATUSES = frozenset({"pass", "soft_pass", "fail", "error", "unsupported"})


class ReportError(RuntimeError):
    """A replay produced no complete, valid results report."""


def read_cases(out: Path, exit_code: int) -> dict[str, str]:
    """Return fixture_id -> status from the one results file `canitoolcall run` wrote to `out`."""
    # `canitoolcall run` exits 1 when any fixture fails, which is expected here.
    # Other exits cannot establish a usable replay, even if a report was written.
    if exit_code not in (0, 1):
        raise ReportError(f"replay exited {exit_code}")
    found = sorted(out.glob("*.json"))
    if len(found) != 1:
        raise ReportError(f"expected exactly one results JSON file, found {len(found)}")
    try:
        result = json.loads(found[0].read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ReportError(f"cannot read results JSON {found[0].name}: {e}") from e
    if not isinstance(result, dict) or not isinstance(result.get("cases"), list) or not result["cases"]:
        raise ReportError("results must contain a nonempty cases list")
    cases: dict[str, str] = {}
    for i, case in enumerate(result["cases"]):
        if not isinstance(case, dict):
            raise ReportError(f"case {i} must be an object")
        fixture_id, status = case.get("fixture_id"), case.get("status")
        if not isinstance(fixture_id, str) or not fixture_id.strip():
            raise ReportError(f"case {i} must have a nonempty string fixture_id")
        if fixture_id in cases:
            raise ReportError(f"duplicate fixture_id: {fixture_id!r}")
        if not isinstance(status, str) or status not in STATUSES:
            raise ReportError(f"invalid status for {fixture_id!r}: {status!r}")
        cases[fixture_id] = status
    return cases


def check_inventories(base: dict[str, str], head: dict[str, str]) -> None:
    """Refuse to compare base and head unless they replayed the same fixture IDs."""
    if base.keys() == head.keys():
        return
    lines = ["base and head fixture inventories differ; comparison is incomplete"]
    if missing := sorted(base.keys() - head.keys()):
        lines.append(f"missing at head: {', '.join(missing)}")
    if added := sorted(head.keys() - base.keys()):
        lines.append(f"added at head: {', '.join(added)}")
    raise ReportError("\n".join(lines))

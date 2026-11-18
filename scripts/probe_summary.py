"""Render a ``canitoolcall probe --json`` report as Markdown for ``$GITHUB_STEP_SUMMARY``.

Used by the GitHub Action (``action.yml``). Standard library only, so it runs with
any Python >= 3.10 and reads reports from any canitoolcall release:

    python scripts/probe_summary.py report.json --exit-code 1 >> "$GITHUB_STEP_SUMMARY"

Text the server controls (problem details, the model name) is escaped, so it cannot
add markup or break the table. It always exits 0; the action passes the probe's exit
code through itself.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

STATUSES = ("pass", "fail", "error", "skip")
COLUMNS = ("nonstream", "stream", "equiv")
EQUIVALENCE_CHECK = "stream_equals_nonstream"
MAX_DETAIL = 300

_MD_PUNCT = re.compile(r"([\\`*_\[\]|#~])")


def escape(text: object) -> str:
    """One line of literal text: HTML entities, Markdown punctuation and newlines neutralised."""
    flat = " ".join(str(text).split())
    return _MD_PUNCT.sub(r"\\\1", html.escape(flat, quote=False))


def _column(o: dict[str, Any]) -> str:
    if o.get("check") == EQUIVALENCE_CHECK:
        return "equiv"
    return "stream" if o.get("stream") else "nonstream"


def _label(o: dict[str, Any]) -> str:
    mode = {"equiv": "stream=non-stream", "stream": "stream", "nonstream": "non-stream"}[_column(o)]
    return f"{o.get('scenario_id')} [{mode}]"


def _cell(o: dict[str, Any]) -> str:
    status = escape(o.get("status")) + ("\\*" if o.get("warnings") else "")
    return f"**{status}**" if o.get("status") in ("fail", "error") else status


def _detail(text: object) -> str:
    s = " ".join(str(text).split())
    return escape(s if len(s) <= MAX_DETAIL else s[:MAX_DETAIL] + "…")


def render(report: dict[str, Any]) -> str:
    outcomes: list[dict[str, Any]] = list(report.get("outcomes") or [])
    counts = report.get("summary") or {}
    head = ", ".join(f"{int(counts.get(k, 0))} {k}" for k in STATUSES)
    out = [
        f"### CanIToolCall probe: {head}",
        "",
        f"Model {escape(report.get('model'))} at {escape(report.get('base_url'))}, "
        f"canitoolcall {escape(report.get('canitoolcall_version'))}.",
        "",
        "| scenario | non-stream | stream | stream=non-stream |",
        "|---|---|---|---|",
    ]
    cells: dict[str, dict[str, str]] = {}
    for o in outcomes:
        cells.setdefault(str(o.get("scenario_id")), {})[_column(o)] = _cell(o)
    for sid, row in cells.items():
        out.append("| " + " | ".join([escape(sid), *(row.get(c, "-") for c in COLUMNS)]) + " |")

    problems = [o for o in outcomes if o.get("status") in ("fail", "error")]
    if problems:
        out += ["", "**Problems**", ""]
        out += [f"- {escape(_label(o))} {escape(o.get('status'))}: {_detail(o.get('detail'))}" for o in problems]
    warned = [(o, w) for o in outcomes for w in o.get("warnings") or []]
    if warned:
        out += ["", "**Warnings** (marked \\*, they do not affect the exit code)", ""]
        out += [f"- {escape(_label(o))}: {_detail(w)}" for o, w in warned]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("report", type=Path, help="JSON report written by `canitoolcall probe --json`")
    p.add_argument("--exit-code", default="?", help="the probe's exit code, quoted when there is no report")
    args = p.parse_args(argv)
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        sys.stdout.write(
            "### CanIToolCall probe: no report\n\n"
            f"`canitoolcall probe` exited with code {escape(args.exit_code)} and wrote no readable report "
            "(an unreachable endpoint or a usage error). See the step log.\n"
        )
        return 0
    sys.stdout.write(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

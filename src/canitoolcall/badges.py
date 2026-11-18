"""shields.io endpoint badges derived from the matrix.

``render_site`` writes one JSON file per cell of the newest version of each
engine (``badges/<engine>/<family>.json``) and one summary per engine
(``badges/<engine>.json``), in the format of
https://shields.io/badges/endpoint-badge (``schemaVersion`` 1, ``label``,
``message``, ``color``). Everything comes from the same matrix that feeds
``matrix.json``; no number is computed here that is not already there.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from canitoolcall.lookup import DEFAULT_SITE, newest_cell
from canitoolcall.results import Status, worst_status

if TYPE_CHECKING:
    from canitoolcall.matrix import Cell, Matrix

BADGE_SITE_URL = DEFAULT_SITE
"""Site URL used in Markdown snippets when the site is built without ``--site-url``: the published site."""

STATUS_COLORS: dict[str, str] = {
    Status.PASS.value: "brightgreen",
    Status.SOFT_PASS.value: "yellowgreen",
    Status.FAIL.value: "red",
    Status.ERROR.value: "orange",
    Status.UNSUPPORTED.value: "lightgrey",
}


def _word(status: Status) -> str:
    return status.value.replace("_", " ")


def _safe(part: str) -> str:
    from canitoolcall.matrix import _safe as safe

    return safe(part)


def engine_badge_path(engine: str) -> str:
    return f"badges/{_safe(engine)}.json"


def cell_badge_path(engine: str, family: str) -> str:
    return f"badges/{_safe(engine)}/{_safe(family)}.json"


def cell_badge(cell: Cell) -> dict[str, Any]:
    """Endpoint JSON for one cell: ``fail · 23/54 strict pass`` or ``unsupported``.

    The headline status is the worst over supported cases; the count is strict passes
    over supported cases, so it cannot be misread as a failure rate.
    """
    message = _word(cell.status)
    if cell.supported:
        message += f" · {cell.counts[Status.PASS.value]}/{cell.supported} strict pass"
    return {
        "schemaVersion": 1,
        "label": f"{cell.family} on {cell.engine}",
        "message": message,
        "color": STATUS_COLORS[cell.status.value],
    }


def engine_badge(engine: str, cells: list[Cell]) -> dict[str, Any]:
    """Endpoint JSON summarising an engine's supported families, e.g. ``7/9 families pass``."""
    supported = [c for c in cells if c.status is not Status.UNSUPPORTED]
    if not supported:
        return {"schemaVersion": 1, "label": engine, "message": "unsupported", "color": STATUS_COLORS["unsupported"]}
    passing = sum(1 for c in supported if c.status is Status.PASS)
    worst = worst_status([c.status for c in supported])
    return {
        "schemaVersion": 1,
        "label": engine,
        "message": f"{passing}/{len(supported)} families pass",
        "color": STATUS_COLORS[worst.value],
    }


def newest_versions(matrix: Matrix) -> dict[str, str]:
    """Newest version per engine: latest ``run_at``, as ``lookup.newest_cell`` picks it (versions may be hashes)."""
    by_engine: dict[str, list[dict[str, str]]] = {}
    for c in matrix.cells:
        by_engine.setdefault(c.engine, []).append({"run_at": c.run_at, "engine_version": c.engine_version})
    return {engine: str(newest_cell(cells)["engine_version"]) for engine, cells in by_engine.items()}


def badge_files(matrix: Matrix) -> dict[str, dict[str, Any]]:
    """Site-relative path to endpoint JSON, for the newest version of every engine."""
    files: dict[str, dict[str, Any]] = {}
    for engine, version in newest_versions(matrix).items():
        cells = [c for c in matrix.cells if c.engine == engine and c.engine_version == version]
        files[engine_badge_path(engine)] = engine_badge(engine, cells)
        for cell in cells:
            files[cell_badge_path(engine, cell.family)] = cell_badge(cell)
    return files


def write_badges(matrix: Matrix, out_dir: Path) -> None:
    """Write the badge files, first removing ``<out_dir>/badges`` so stale files do not linger."""
    shutil.rmtree(out_dir / "badges", ignore_errors=True)
    for rel, data in badge_files(matrix).items():
        target = out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def badge_markdown(path: str, alt: str, site_url: str | None = None) -> str:
    """The Markdown image snippet for a badge file at site-relative ``path``."""
    base = (site_url or BADGE_SITE_URL).rstrip("/") + "/"
    return f"![{alt}](https://img.shields.io/endpoint?url={quote(base + path, safe='')})"


def cell_snippets(matrix: Matrix, cell: Cell, site_url: str | None = None) -> dict[str, str]:
    """Markdown snippets for a cell page (empty unless the cell is on the engine's newest version)."""
    if newest_versions(matrix).get(cell.engine) != cell.engine_version:
        return {}
    return {
        "cell": badge_markdown(cell_badge_path(cell.engine, cell.family), f"{cell.family} on {cell.engine}", site_url),
        "engine": badge_markdown(engine_badge_path(cell.engine), cell.engine, site_url),
    }

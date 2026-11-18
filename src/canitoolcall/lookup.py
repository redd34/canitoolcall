"""``canitoolcall lookup``: answer "does family X parse on engine Y?" from a ``matrix.json``.

The source is the published matrix by default, or a local file or URL given
with ``--matrix``. Only the standard library is used for HTTP, with a timeout.
The schema is ``spec/matrix.schema.json``; this reader accepts schema major
version 1 and rejects anything else.
"""

from __future__ import annotations

import json
import re
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any
from urllib.parse import urljoin, urlsplit

from canitoolcall.results import Status

DEFAULT_SITE = "https://redd34.github.io/canitoolcall/"
DEFAULT_MATRIX = DEFAULT_SITE + "matrix.json"
SUPPORTED_MAJOR = 1
TIMEOUT = 15.0
MAX_BYTES = 20 * 1024 * 1024
MIN_PARTIAL = 3
"""Shortest query (after normalising) allowed to match by substring."""


class LookupFailure(Exception):
    """A user-facing failure (bad source, unknown name, unsupported schema).

    The message is passed through :func:`clean`, so text that came from a
    hostile matrix.json (a family slug, an engine name, the source) can never
    put control characters on the terminal, whichever ``raise`` built it.
    """

    def __init__(self, message: object) -> None:
        super().__init__(clean(message))


_STATUSES = frozenset(s.value for s in Status)


def clean(text: object) -> str:
    """Make a string from a (possibly hostile) matrix.json safe to print.

    Control characters (C0, DEL, C1, including ESC), format characters such as
    bidi overrides, and line/paragraph separators become visible ``\\xNN`` or
    ``\\uNNNN`` escapes.
    """
    out = []
    for ch in str(text):
        if unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp"):
            out.append(f"\\x{ord(ch):02x}" if ord(ch) < 0x100 else f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return "".join(out)


def _natural_key(text: str) -> tuple[tuple[int, int | str], ...]:
    return tuple((0, int(p)) if p.isdigit() else (1, p) for p in re.split(r"(\d+)", text) if p)


def _is_url(source: str) -> bool:
    return urlsplit(source).scheme in ("http", "https")


def load_matrix(source: str, timeout: float = TIMEOUT) -> dict[str, Any]:
    """Read and version-check a matrix.json from a path or an http(s) URL."""
    try:
        if _is_url(source):
            req = urllib.request.Request(source, headers={"User-Agent": "canitoolcall-lookup"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read(MAX_BYTES + 1)
        else:
            with open(source, "rb") as fh:
                raw = fh.read(MAX_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        reason = getattr(e, "reason", e)
        raise LookupFailure(f"cannot read {clean(source)}: {clean(reason)}") from e
    if len(raw) > MAX_BYTES:
        raise LookupFailure(f"{clean(source)} is larger than {MAX_BYTES} bytes")
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise LookupFailure(f"{clean(source)} is not valid JSON: {clean(e)}") from e
    check_version(data, source)
    return data  # type: ignore[no-any-return]


def check_version(data: Any, source: str = "matrix.json") -> None:
    """Validate the schema version and the shape of everything ``lookup`` reads."""
    source = clean(source)
    if not isinstance(data, dict):
        raise LookupFailure(f"{source} is not a matrix.json object")
    if "schema_version" not in data:
        raise LookupFailure(
            f"{source} has no schema_version: it was built by canitoolcall < 0.3, before matrix.json was "
            "versioned. The published site is rebuilt nightly, so the default source works from the first "
            "nightly after a release containing `lookup`; until then pass --matrix with a newer matrix.json."
        )
    version = data["schema_version"]
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+", version):
        raise LookupFailure(f"{source} has an unusable schema_version ({clean(repr(version))})")
    major = int(version.split(".")[0])
    if major != SUPPORTED_MAJOR:
        raise LookupFailure(
            f"{source} has schema_version {clean(version)}, but this canitoolcall understands "
            f"major version {SUPPORTED_MAJOR}; "
            "upgrade canitoolcall"
        )
    for key in ("families", "engines", "cells"):
        if not isinstance(data.get(key), list):
            raise LookupFailure(f"{source} is malformed: {key!r} is missing or not a list")
    if not all(isinstance(f, str) for f in data["families"]):
        raise LookupFailure(f"{source} is malformed: 'families' must be strings")
    for e in data["engines"]:
        if not isinstance(e, dict) or not isinstance(e.get("name"), str):
            raise LookupFailure(f"{source} is malformed: an engine entry has no string name")
    for c in data["cells"]:
        _check_cell(c, source)


def _check_cell(c: Any, source: str) -> None:
    if not isinstance(c, dict):
        raise LookupFailure(f"{source} is malformed: a cell is not an object")
    for key in ("family", "engine", "engine_version", "page"):
        if not isinstance(c.get(key), str):
            raise LookupFailure(f"{source} is malformed: a cell has no string {key!r}")
    if c.get("status") not in _STATUSES:
        raise LookupFailure(f"{source} has a cell with unknown status {clean(repr(c.get('status')))}")
    rate = c.get("pass_rate")
    if rate is not None and (isinstance(rate, bool) or not isinstance(rate, (int, float))):
        raise LookupFailure(f"{source} is malformed: pass_rate must be a number or null")
    counts = c.get("counts", {})
    if not isinstance(counts, dict) or not all(isinstance(v, int) and not isinstance(v, bool) for v in counts.values()):
        raise LookupFailure(f"{source} is malformed: counts must map names to integers")
    page = c["page"]
    parts = page.replace("\\", "/").split("/")
    sp = urlsplit(page)
    if sp.scheme or sp.netloc or page.startswith(("/", "\\")) or ".." in parts or page.startswith("//"):
        raise LookupFailure(f"{source} has a cell page outside the site ({clean(repr(page))})")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def resolve_family(data: Mapping[str, Any], query: str) -> tuple[str, str]:
    """Map a slug, family name or model name to a family slug.

    Returns ``(slug, how)``. Matching order: exact slug; family display name;
    a model repo id (full or the part after ``org/``) from ``family_info``;
    then a unique slug/name/model substring (only for queries of at least
    ``MIN_PARTIAL`` characters). Case-insensitive throughout.
    """
    families: list[str] = list(data["families"])
    info: Mapping[str, Any] = data.get("family_info") or {}
    q = _norm(query)
    if not q:
        raise LookupFailure("empty family or model name")

    for slug in families:
        if _norm(slug) == q:
            return slug, "family slug"
    for slug in families:
        if _norm(str(info.get(slug, {}).get("name", ""))) == q:
            return slug, "family name"
    exact = _hits(families, info, lambda m: q in {_norm(m), _norm(m.split("/", 1)[-1])}, models_only=True)
    if len(exact) == 1:
        return exact[0], "model name"
    if len(exact) > 1:
        raise _ambiguous(query, exact)

    loose = _hits(families, info, lambda s: q in _norm(s)) if len(q) >= MIN_PARTIAL else []
    if len(loose) == 1:
        return loose[0], "partial match"
    if len(loose) > 1:
        raise _ambiguous(query, loose)
    note = f" (partial matches need at least {MIN_PARTIAL} characters)" if len(q) < MIN_PARTIAL else ""
    if not info:
        note += " (this matrix.json has no model list, so only family slugs and names were matched)"
    raise LookupFailure(clean(f"unknown family or model {query!r}; families: {', '.join(families)}{note}"))


def _hits(families: list[str], info: Mapping[str, Any], match: Any, models_only: bool = False) -> list[str]:
    out: list[str] = []
    for slug in families:
        meta = info.get(slug, {})
        names = [str(m) for m in meta.get("models", [])]
        if not models_only:
            names += [slug, str(meta.get("name", ""))]
        if any(match(n) for n in names):
            out.append(slug)
    return out


def _ambiguous(query: str, slugs: list[str]) -> LookupFailure:
    return LookupFailure(clean(f"{query!r} matches several families ({', '.join(slugs)}); use a family slug"))


def lookup(
    data: Mapping[str, Any], query: str, engine: str | None = None, base_url: str = DEFAULT_SITE
) -> dict[str, Any]:
    """Answer the question; every string in the result is made print-safe. Raises :class:`LookupFailure`."""
    family, how = resolve_family(data, query)
    engines = sorted({str(e["name"]) for e in data["engines"]})
    if engine is not None and engine.lower() not in {e.lower() for e in engines}:
        raise LookupFailure(clean(f"unknown engine {engine!r}; engines in this matrix: {', '.join(engines)}"))
    cells = [
        c
        for c in data["cells"]
        if c["family"] == family and (engine is None or str(c["engine"]).lower() == engine.lower())
    ]
    if not cells:
        raise LookupFailure(f"the matrix has no cell for {family} on {engine or 'any engine'}")
    # One row per engine: the latest version, as the matrix columns sort them.
    newest: dict[str, dict[str, Any]] = {}
    for c in cells:
        key = str(c["engine"])
        if key not in newest or _natural_key(str(c["engine_version"])) >= _natural_key(
            str(newest[key]["engine_version"])
        ):
            newest[key] = c
    return {
        "query": clean(query),
        "family": clean(family),
        "matched_by": how,
        "schema_version": clean(data["schema_version"]),
        "cells": [
            {
                "engine": clean(c["engine"]),
                "engine_version": clean(c["engine_version"]),
                "status": c["status"],
                "pass_rate": c.get("pass_rate"),
                "counts": {clean(k): v for k, v in c.get("counts", {}).items()},
                "run_at": None if c.get("run_at") is None else clean(c["run_at"]),
                "url": clean(urljoin(base_url, str(c["page"]))),
            }
            for c in sorted(newest.values(), key=lambda c: str(c["engine"]))
        ],
    }


def format_table(result: Mapping[str, Any]) -> str:
    """A short human table for :func:`lookup` output."""
    lines = [clean(f"{result['family']} (matched {result['query']!r} by {result['matched_by']})")]
    rows = []
    for c in result["cells"]:
        rate = "n/a" if c["pass_rate"] is None else f"{c['pass_rate'] * 100:.0f}%"
        counts = " ".join(f"{k}={v}" for k, v in c["counts"].items() if v)
        rows.append(
            (
                clean(f"{c['engine']} {c['engine_version']}"),
                clean(c["status"]),
                rate,
                clean(counts or "-"),
                clean(c["url"]),
            )
        )
    head = ("ENGINE", "STATUS", "STRICT", "COUNTS", "PAGE")
    widths = [max(len(r[i]) for r in [head, *rows]) for i in range(4)]
    for r in [head, *rows]:
        lines.append("  ".join(r[i].ljust(widths[i]) for i in range(4)) + "  " + r[4])
    return "\n".join(lines)

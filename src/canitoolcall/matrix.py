"""Static matrix site generator (``canitoolcall matrix``).

Reads ``results/*.json`` (spec/results.schema.json) and renders a static site:

* ``index.html``: the family x engine@version grid (headline status, strict
  pass rate, soft passes shown separately, and the checks below 100 %), an
  engine x check pass-rate table, and the metadata of every run.
* ``cells/<engine>/<version>/<family>.html``: one page per cell with per-check
  pass rates and a drill-down for every non-passing fixture: the failing
  checks and strategies, the observed vs expected parse, the parser
  configuration, and a minimal repro command.
* ``data/<file>.json``: the unmodified results files the site was built from.
* ``matrix.json``: the aggregated grid, for badges and other tools.

The matrix is built ONLY from real run files. The fixture corpus is read
optionally, to show a failing fixture's raw output and expected parse next
to what the engine produced; whether the corpus still matches the run's
``fixtures_digest`` is stated on the page.

Only realistic chunking strategies count toward a status. Synthetic
strategies are reported separately (see docs/DESIGN.md): ``char:<seed>``
always, and multi-token strategies for engines that stream one token per event
(``run.synthetic_strategies`` in the results file).
"""

from __future__ import annotations

import difflib
import json
import re
import shlex
import shutil
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from canitoolcall import __version__
from canitoolcall.chunking import ChunkStrategy
from canitoolcall.fixtures import (
    Family,
    Fixture,
    default_fixtures_dir,
    fixtures_digest,
    load_families,
    load_fixtures,
    repo_root,
    spec_dir,
)
from canitoolcall.results import CaseResult, CheckResult, ParseResult, RunResults, Status, worst_status

if TYPE_CHECKING:
    import jinja2

CHECKS: tuple[str, ...] = (
    "expected_match",
    "expected_error",
    "stream_equals_nonstream",
    "split_invariance",
    "no_leakage",
    "arguments_json",
    "arguments_schema",
    "parallel_order",
)
"""All checks, in display order (spec/README.md, "Checks")."""

COUNTED: tuple[Status, ...] = (Status.PASS, Status.SOFT_PASS, Status.FAIL, Status.ERROR)
"""Statuses that count toward a rate; ``unsupported`` never does."""

STATUS_LABELS: dict[str, str] = {
    Status.PASS.value: "pass",
    Status.SOFT_PASS.value: "soft pass",
    Status.FAIL.value: "fail",
    Status.ERROR.value: "error",
    Status.UNSUPPORTED.value: "unsupported",
}

STATUS_ICONS: dict[str, str] = {
    Status.PASS.value: "\N{CHECK MARK}",
    Status.SOFT_PASS.value: "\N{ALMOST EQUAL TO}",
    Status.FAIL.value: "\N{BALLOT X}",
    Status.ERROR.value: "!",
    Status.UNSUPPORTED.value: "\N{EN DASH}",
}


# --------------------------------------------------------------------------- aggregation


def is_realistic(strategy: str, synthetic: Collection[str] = ()) -> bool:
    """True for ``nonstream`` and every strategy the engine can actually produce.

    ``synthetic`` is the run's ``synthetic_strategies``. Unknown ids are
    treated as realistic, so they are never silently hidden.
    """
    if strategy in ("nonstream", "*"):
        return True
    if strategy in synthetic:
        return False
    try:
        return ChunkStrategy.parse(strategy).realistic
    except ValueError:
        return True


def case_status(case: CaseResult, synthetic: Collection[str] = ()) -> Status:
    """A case's matrix status: its status over realistic strategies only.

    It equals ``case.status`` unless the run included synthetic strategies.
    """
    if case.status is Status.UNSUPPORTED:
        return Status.UNSUPPORTED
    realistic = [c for c in case.checks if is_realistic(c.strategy, synthetic)]
    if len(realistic) == len(case.checks):
        return case.status
    statuses = [c.status for c in realistic if c.status is not Status.UNSUPPORTED]
    if case.harness_error is not None:
        statuses.append(Status.ERROR)
    return worst_status(statuses)


def _empty_counts() -> dict[str, int]:
    return {s.value: 0 for s in Status}


@dataclass(frozen=True)
class CheckStats:
    """How one check fared over a set of cases (worst status over realistic strategies per case)."""

    check: str
    counts: Mapping[str, int]

    @property
    def applied(self) -> int:
        """Cases the check produced a verdict for."""
        return sum(self.counts[s.value] for s in COUNTED)

    @property
    def pass_rate(self) -> float | None:
        """Strict passes / applied, or None when the check never applied."""
        return self.counts[Status.PASS.value] / self.applied if self.applied else None


def check_stats(cases: Iterable[CaseResult], synthetic: Collection[str] = ()) -> tuple[CheckStats, ...]:
    """Per-check counts over supported cases; checks that never applied are omitted."""
    counts: dict[str, dict[str, int]] = {}
    for case in cases:
        if case.status is Status.UNSUPPORTED:
            continue
        per_check: dict[str, list[Status]] = {}
        for c in case.checks:
            if is_realistic(c.strategy, synthetic):
                per_check.setdefault(c.check, []).append(c.status)
        for name, statuses in per_check.items():
            counted = [s for s in statuses if s is not Status.UNSUPPORTED]
            status = worst_status(counted) if counted else Status.UNSUPPORTED
            counts.setdefault(name, _empty_counts())[status.value] += 1
    order = {name: i for i, name in enumerate(CHECKS)}
    names = sorted(counts, key=lambda n: (order.get(n, len(order)), n))
    return tuple(CheckStats(n, counts[n]) for n in names if sum(counts[n][s.value] for s in COUNTED))


@dataclass(frozen=True)
class Cell:
    """One (family, engine@version) cell."""

    family: str
    engine: str
    engine_version: str
    counts: dict[str, int]
    status: Status
    """Headline status: worst over supported cases, or UNSUPPORTED if none."""
    run_at: str
    cases: tuple[CaseResult, ...] = ()
    checks: tuple[CheckStats, ...] = ()
    stress_failures: int = 0
    """Cases whose synthetic strategies failed; never part of ``status``."""
    synthetic: tuple[str, ...] = ()
    """The run's synthetic strategies (never counted)."""

    @property
    def supported(self) -> int:
        return sum(self.counts[s.value] for s in COUNTED)

    @property
    def pass_rate(self) -> float | None:
        """Strict passes / supported cases, or None when nothing was supported."""
        return self.counts[Status.PASS.value] / self.supported if self.supported else None

    @property
    def multi_token_synthetic(self) -> tuple[str, ...]:
        """Synthetic strategies other than ``char:*`` (multi-token deltas the engine never sends)."""
        return tuple(s for s in self.synthetic if not s.startswith("char:"))

    @property
    def weak_checks(self) -> tuple[CheckStats, ...]:
        """Checks that did not strictly pass on every applicable case."""
        return tuple(c for c in self.checks if c.counts[Status.PASS.value] < c.applied)


def build_cell(family: str, run: RunResults) -> Cell:
    """Aggregate one family's cases of one run."""
    cases = tuple(c for c in run.cases if c.family == family)
    syn = run.run.synthetic_strategies
    counts = _empty_counts()
    for case in cases:
        counts[case_status(case, syn).value] += 1
    supported = [case_status(c, syn) for c in cases if case_status(c, syn) is not Status.UNSUPPORTED]
    stress = sum(
        1
        for c in cases
        if any(not is_realistic(k.strategy, syn) and k.status in (Status.FAIL, Status.ERROR) for k in c.checks)
    )
    return Cell(
        family=family,
        engine=run.engine.name,
        engine_version=run.engine.version,
        counts=counts,
        status=worst_status(supported) if supported else Status.UNSUPPORTED,
        run_at=run.run.finished_at,
        cases=cases,
        checks=check_stats(cases, syn),
        stress_failures=stress,
        synthetic=tuple(syn),
    )


@dataclass(frozen=True)
class Matrix:
    families: tuple[str, ...]
    engines: tuple[tuple[str, str], ...]
    """(engine, version) columns, sorted."""
    cells: tuple[Cell, ...]
    runs: tuple[RunResults, ...] = ()
    """The run behind each column, aligned with ``engines``."""

    def cell(self, family: str, engine: str, version: str) -> Cell | None:
        return next(
            (c for c in self.cells if c.family == family and c.engine == engine and c.engine_version == version),
            None,
        )

    def run_for(self, engine: str, version: str) -> RunResults | None:
        return next((r for r in self.runs if (r.engine.name, r.engine.version) == (engine, version)), None)


def _natural_key(text: str) -> tuple[tuple[int, int | str], ...]:
    """Sort ``0.9.1`` before ``0.10.0``; numbers sort before words."""
    return tuple((0, int(p)) if p.isdigit() else (1, p) for p in re.split(r"(\d+)", text) if p)


def _parse_time(ts: str) -> datetime:
    """Parse an ISO 8601 timestamp (``Z`` allowed on Python 3.10); unparsable sorts first."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def build_matrix(runs: Sequence[RunResults]) -> Matrix:
    """Aggregate runs into a Matrix (latest run wins per engine@version)."""
    latest: dict[tuple[str, str], RunResults] = {}
    for run in runs:
        key = (run.engine.name, run.engine.version)
        prev = latest.get(key)
        if prev is None or _parse_time(run.run.finished_at) >= _parse_time(prev.run.finished_at):
            latest[key] = run
    engines = tuple(sorted(latest, key=lambda k: (k[0], _natural_key(k[1]))))
    chosen = tuple(latest[k] for k in engines)
    families = tuple(sorted({c.family for r in chosen for c in r.cases}))
    cells = tuple(build_cell(fam, run) for fam in families for run in chosen if any(c.family == fam for c in run.cases))
    return Matrix(families=families, engines=engines, cells=cells, runs=chosen)


# --------------------------------------------------------------------------- loading


def results_files(results_dir: Path) -> list[Path]:
    """Results files in ``results_dir``: ``*.json`` and gzipped ``*.json.gz`` (committed snapshots)."""
    if not results_dir.is_dir():
        return []
    return sorted([*results_dir.glob("*.json"), *results_dir.glob("*.json.gz")], key=lambda p: p.name)


def no_results_hint(results_dir: Path) -> str:
    """Why ``results_dir`` has nothing to render, with the likely fix."""
    dated = sorted(p for p in results_dir.iterdir() if p.is_dir() and results_files(p)) if results_dir.is_dir() else []
    hint = f"; did you mean --results {dated[-1]}?" if dated else "; run `canitoolcall run --engine ...` first"
    return f"no results files (*.json, *.json.gz) in {results_dir}{hint}"


def read_results_text(path: Path) -> str:
    """Text of a results file, transparently decompressing ``*.json.gz``."""
    if path.name.endswith(".gz"):
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as f:
            return f.read()
    return path.read_text(encoding="utf-8")


def _data_name(path: Path) -> str:
    """Published name of a results file under ``data/`` (always plain JSON)."""
    return path.name[: -len(".gz")] if path.name.endswith(".gz") else path.name


def load_results(results_dir: Path) -> list[RunResults]:
    """Load and schema-check every ``*.json`` / ``*.json.gz`` in ``results_dir``.

    Raises ``FileNotFoundError`` if the directory does not exist, and
    ``ValueError`` naming the file for anything that is not a valid results file.
    """
    import jsonschema  # lazy: the main env has it; engine venvs never import this module

    if not results_dir.is_dir():
        raise FileNotFoundError(f"results directory {results_dir} does not exist")
    schema = json.loads((spec_dir() / "results.schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    runs: list[RunResults] = []
    for path in results_files(results_dir):
        try:
            data = json.loads(read_results_text(path))
        except (json.JSONDecodeError, OSError, EOFError) as e:
            raise ValueError(f"{path}: invalid JSON: {e}") from e
        errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))
        if errors:
            first = errors[0]
            more = f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""
            raise ValueError(f"{path}: not a valid results file: {first.json_path}: {first.message}{more}")
        try:
            runs.append(RunResults.from_dict(data))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{path}: {e}") from e
    return runs


# --------------------------------------------------------------------------- drill-down views


@dataclass(frozen=True)
class Outcome:
    """One distinct parse, and the strategies that produced it."""

    strategies: tuple[str, ...]
    observed_json: str
    diff: tuple[tuple[str, str], ...]
    """``(css_class, line)`` pairs of a unified diff expected -> observed; empty if equal or no expectation."""
    matches_expected: bool | None = None
    """Strict match with the fixture's ``expected``; None when there is no expectation to compare with."""


@dataclass(frozen=True)
class FixtureContext:
    """The fixture as found in the corpus at build time."""

    source: str | None
    line: int | None
    raw_output: str
    expected_json: str | None
    expected_error: str | None
    provenance_url: str | None
    """Link target: set only for http(s) URLs."""
    provenance_text: str
    """The source URL as recorded, shown as text."""
    provenance_kind: str
    record_jsonl: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class CaseView:
    fixture_id: str
    anchor: str
    status: Status
    failing: tuple[CheckResult, ...]
    stress: tuple[CheckResult, ...]
    outcomes: tuple[Outcome, ...]
    parser_config_json: str | None
    harness_error: str | None
    reason: str | None
    skipped: Mapping[str, str]
    repro: str
    fixture: FixtureContext | None


def _normalize_empty(value: str | None) -> str | None:
    return value or None


def _observed_view(result: ParseResult) -> dict[str, Any]:
    """An observed parse in the same shape as ``expected`` (strict: ``""`` shown as null)."""
    calls: list[dict[str, Any]] = []
    for tc in result.tool_calls:
        try:
            args: Any = tc.arguments()
        except json.JSONDecodeError:
            args = {"<arguments_raw, not valid JSON>": tc.arguments_raw}
        calls.append({"name": tc.name, "arguments": args})
    out: dict[str, Any] = {
        "content": _normalize_empty(result.content),
        "reasoning_content": _normalize_empty(result.reasoning_content),
        "tool_calls": calls,
    }
    if result.exception is not None:
        out["exception"] = result.exception
    return out


def _expected_view(fx: Fixture) -> dict[str, Any] | None:
    if fx.expected is None:
        return None
    return {
        "content": fx.expected.content,
        "reasoning_content": fx.expected.reasoning_content,
        "tool_calls": [{"name": tc.name, "arguments": dict(tc.arguments)} for tc in fx.expected.tool_calls],
    }


def _pretty(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)


def _diff(expected: str, observed: str) -> tuple[tuple[str, str], ...]:
    if expected == observed:
        return ()
    lines = difflib.unified_diff(expected.splitlines(), observed.splitlines(), "expected", "observed", lineterm="", n=2)
    out: list[tuple[str, str]] = []
    for line in lines:
        if line.startswith(("---", "+++")):
            cls = "meta"
        elif line.startswith("@@"):
            cls = "hunk"
        elif line.startswith("+"):
            cls = "add"
        elif line.startswith("-"):
            cls = "del"
        else:
            cls = "ctx"
        out.append((cls, line))
    return tuple(out)


def _outcomes(case: CaseResult, fx: Fixture | None) -> tuple[Outcome, ...]:
    """Group identical observed parses; failing strategies first."""
    if case.observed is None:
        return ()
    parses: list[tuple[str, ParseResult]] = [("nonstream", case.observed.nonstream), *case.observed.streams.items()]
    groups: dict[str, list[str]] = {}
    for strategy, result in parses:
        groups.setdefault(_pretty(_observed_view(result)), []).append(strategy)
    expected = _expected_view(fx) if fx is not None else None
    expected_json = _pretty(expected) if expected is not None else None
    failing = {c.strategy for c in case.checks if c.status in (Status.FAIL, Status.ERROR)}
    outcomes = [
        Outcome(
            tuple(strats),
            obs,
            _diff(expected_json, obs) if expected_json is not None else (),
            obs == expected_json if expected_json is not None else None,
        )
        for obs, strats in groups.items()
    ]
    outcomes.sort(key=lambda o: not failing.intersection(o.strategies))
    return tuple(outcomes)


def _rel_source(fx: Fixture, fixtures_dir: Path | None) -> str | None:
    if fx.source is None:
        return None
    if fixtures_dir is not None:
        try:
            return (Path(fixtures_dir.name) / fx.source.resolve().relative_to(fixtures_dir.resolve())).as_posix()
        except ValueError:
            pass
    return fx.source.as_posix()


def repro_command(case: CaseResult, engine: str, source: str | None, synthetic: Collection[str] = ()) -> str:
    """A minimal command that replays just this fixture (``--id``) with the failing strategies.

    Every interpolated value is shell-quoted: results files are untrusted input.
    """
    strategies = sorted(
        {
            c.strategy
            for c in case.checks
            if c.status in (Status.FAIL, Status.ERROR, Status.SOFT_PASS)
            and c.strategy not in ("nonstream", "*")  # "*" is split_invariance's summary row
            and is_realistic(c.strategy, synthetic)
        }
    ) or ["token" if "one" in synthetic else "one"]
    q = shlex.quote
    target = f"--fixtures {q(source)}" if source else f"--family {q(case.family)}"
    flags = " ".join(f"--strategy {q(s)}" for s in strategies)
    return f"uv run canitoolcall run --engine {q(engine)} {target} --id {q(case.fixture_id)} {flags} --observed all"


def _http_url(url: str) -> str | None:
    """``url`` if it is an absolute http(s) URL, else None (never render javascript: etc. as a link)."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    return url if parts.scheme in ("http", "https") and parts.netloc else None


def _fixture_context(fx: Fixture, fixtures_dir: Path | None) -> FixtureContext:
    expected = _expected_view(fx)
    err = fx.expected_error
    return FixtureContext(
        source=_rel_source(fx, fixtures_dir),
        line=fx.line,
        raw_output=fx.raw_output,
        expected_json=_pretty(expected) if expected is not None else None,
        expected_error=f"{err.reason} (accept: {', '.join(err.accept)})" if err is not None else None,
        provenance_url=_http_url(fx.provenance.source_url),
        provenance_text=fx.provenance.source_url,
        provenance_kind=fx.provenance.kind,
        record_jsonl=json.dumps(fx.to_dict(), ensure_ascii=False),
        tags=fx.tags,
    )


def skipped_summary(cases: Iterable[CaseResult]) -> dict[str, tuple[int, str]]:
    """``strategy -> (number of cases, first reason)`` for strategies the worker could not run."""
    out: dict[str, tuple[int, str]] = {}
    for case in cases:
        for strategy, reason in case.skipped_strategies.items():
            n, first = out.get(strategy, (0, reason))
            out[strategy] = (n + 1, first)
    return dict(sorted(out.items()))


def anchor_for(fixture_id: str) -> str:
    return "f-" + re.sub(r"[^A-Za-z0-9_-]", "-", fixture_id)


def case_view(
    case: CaseResult,
    engine: str,
    fixtures: Mapping[str, Fixture],
    fixtures_dir: Path | None,
    synthetic: Collection[str] = (),
) -> CaseView:
    fx = fixtures.get(case.fixture_id)
    source = _rel_source(fx, fixtures_dir) if fx is not None else None
    return CaseView(
        fixture_id=case.fixture_id,
        anchor=anchor_for(case.fixture_id),
        status=case_status(case, synthetic),
        failing=tuple(c for c in case.checks if is_realistic(c.strategy, synthetic) and c.status is not Status.PASS),
        stress=tuple(c for c in case.checks if not is_realistic(c.strategy, synthetic) and c.status is not Status.PASS),
        outcomes=_outcomes(case, fx),
        parser_config_json=_pretty(dict(case.parser_config)) if case.parser_config is not None else None,
        harness_error=case.harness_error,
        reason=case.reason,
        skipped=dict(case.skipped_strategies),
        repro=repro_command(case, engine, source, synthetic),
        fixture=_fixture_context(fx, fixtures_dir) if fx is not None else None,
    )


# --------------------------------------------------------------------------- rendering


def default_templates_dir() -> Path:
    """``site/templates`` in a checkout, else the copy shipped in the wheel."""
    root = repo_root()
    if root is not None and (root / "site" / "templates").is_dir():
        return root / "site" / "templates"
    return Path(__file__).resolve().parent / "_data" / "templates"


def default_spec_dir() -> Path:
    """``spec`` in a checkout, else the copy shipped in the wheel."""
    root = repo_root()
    if root is not None and (root / "spec").is_dir():
        return root / "spec"
    return Path(__file__).resolve().parent / "_data" / "spec"


def _safe(part: str) -> str:
    """A path segment that is safe on every filesystem and in URLs."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", part).lstrip(".") or "_"


def cell_path(engine: str, version: str, family: str) -> str:
    """Site-relative path of a cell page."""
    return f"cells/{_safe(engine)}/{_safe(version)}/{_safe(family)}.html"


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    pct = value * 100
    if 99.5 <= pct < 100:  # never round a failure up to 100 %
        return ">99%"
    return f"{pct:.0f}%"


def _date(ts: str) -> str:
    return ts[:10]


@dataclass
class _Corpus:
    fixtures: dict[str, Fixture] = field(default_factory=dict)
    families: dict[str, Family] = field(default_factory=dict)
    root: Path | None = None
    note: str | None = None


def _load_corpus(fixtures_dir: Path | None) -> _Corpus:
    root = fixtures_dir if fixtures_dir is not None else default_fixtures_dir()
    # Notes are published: name the corpus directory only, never an absolute local path.
    if not root.is_dir():
        return _Corpus(note=f"fixture corpus {root.name!r} not found; raw outputs are not shown")
    try:
        fixtures = {f.id: f for f in load_fixtures([root])}
        families = load_families(root)
    except (OSError, ValueError, KeyError, TypeError) as e:
        detail = type(e).__name__ if isinstance(e, OSError) else str(e).replace(str(root.resolve()), root.name)
        return _Corpus(
            root=root, note=f"could not read the fixture corpus {root.name!r} ({detail}); raw outputs are not shown"
        )
    return _Corpus(fixtures=fixtures, families=families, root=root)


def _digest_matches(run: RunResults, corpus: _Corpus) -> bool:
    ids = {c.fixture_id for c in run.cases}
    if not ids or not ids <= corpus.fixtures.keys():
        return False
    return fixtures_digest(corpus.fixtures[i] for i in ids) == run.run.fixtures_digest


def _environment(templates_dir: Path) -> jinja2.Environment:
    import jinja2

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(templates_dir)),
        autoescape=True,
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters["pct"] = _pct
    env.filters["date"] = _date
    env.filters["label"] = lambda s: STATUS_LABELS[Status(s).value]
    env.filters["icon"] = lambda s: STATUS_ICONS[Status(s).value]
    env.filters["sv"] = lambda s: Status(s).value
    env.filters["short"] = lambda s, n=12: (s or "")[:n]
    env.filters["pretty"] = lambda v: _pretty(dict(v) if isinstance(v, Mapping) else v)
    return env


MATRIX_SCHEMA_VERSION = "1.0"
"""``matrix.json`` schema version (``MAJOR.MINOR``, see ``spec/README.md``)."""


def _family_info(corpus: _Corpus) -> dict[str, Any]:
    """Display name and model repos per family, so ``lookup`` can resolve model names."""
    info: dict[str, Any] = {}
    for slug, fam in sorted(corpus.families.items()):
        models = {m.repo for m in fam.reference_models}
        models.update(m for fx in corpus.fixtures.values() if fx.family == slug for m in fx.models)
        info[slug] = {"name": fam.name, "models": sorted(models)}
    return info


def _per_model(cell: Cell, fixture_models: Mapping[str, Sequence[str]]) -> dict[str, Any] | None:
    """Status, counts and pass rate per model repo, from each case's fixture ``models``.

    None when a case's fixture is missing from the corpus: partial counts would mislead.
    """
    if not cell.cases or any(c.fixture_id not in fixture_models for c in cell.cases):
        return None
    by_model: dict[str, list[CaseResult]] = {}
    for case in cell.cases:
        for model in fixture_models[case.fixture_id]:
            by_model.setdefault(model, []).append(case)
    out: dict[str, Any] = {}
    for model, cases in sorted(by_model.items()):
        counts = _empty_counts()
        for case in cases:
            counts[case_status(case, cell.synthetic).value] += 1
        supported = [s for s in COUNTED if counts[s.value]]
        total = sum(counts[s.value] for s in supported)
        out[model] = {
            "status": (worst_status(supported) if supported else Status.UNSUPPORTED).value,
            "counts": counts,
            "pass_rate": counts[Status.PASS.value] / total if total else None,
        }
    return out


def matrix_json(
    matrix: Matrix,
    family_info: Mapping[str, Any] | None = None,
    fixture_models: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """The aggregated grid as plain JSON (written to ``matrix.json``).

    ``fixture_models`` maps fixture ids to their ``models``; with it, each cell
    whose fixtures are all known gets per-model results under ``models``.
    """
    return {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "canitoolcall_version": __version__,
        "families": list(matrix.families),
        **({"family_info": dict(family_info)} if family_info else {}),
        "engines": [
            {
                "name": r.engine.name,
                "version": r.engine.version,
                "commit": r.engine.commit,
                "run_started_at": r.run.started_at,
                "run_finished_at": r.run.finished_at,
            }
            for r in matrix.runs
        ],
        "cells": [
            {
                "family": c.family,
                "engine": c.engine,
                "engine_version": c.engine_version,
                "status": c.status.value,
                "counts": c.counts,
                "pass_rate": c.pass_rate,
                "stress_failures": c.stress_failures,
                "checks": {k.check: dict(k.counts) for k in c.checks},
                "run_at": c.run_at,
                "page": cell_path(c.engine, c.engine_version, c.family),
                **({"models": m} if (m := _per_model(c, fixture_models or {})) else {}),
            }
            for c in matrix.cells
        ],
    }


def render_site(
    results_dir: Path,
    out_dir: Path,
    templates_dir: Path | None = None,
    *,
    fixtures_dir: Path | None = None,
    site_url: str | None = None,
) -> Path:
    """Render the static site; returns the path of ``index.html``.

    ``fixtures_dir`` (default: :func:`default_fixtures_dir`) is only used to
    show raw outputs and expected parses in the drill-down; a missing corpus
    does not stop the build. Files are overwritten, never deleted.

    ``site_url`` is the absolute URL the site is published at. When given,
    pages get canonical, ``og:url`` and ``og:image`` links and a ``sitemap.xml``
    is written; without it the site stays location-independent. (No
    ``robots.txt``: crawlers only read it at a host's root, which a project
    page under a path does not own.)
    """
    base_url = _site_base_url(site_url)
    templates_dir = templates_dir or default_templates_dir()
    if not (templates_dir / "index.html").is_file():
        raise FileNotFoundError(f"no site templates in {templates_dir}")
    files = results_files(results_dir)
    runs = load_results(results_dir)
    matrix = build_matrix(runs)
    corpus = _load_corpus(fixtures_dir)
    env = _environment(templates_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    assets = templates_dir / "assets"
    if assets.is_dir():
        shutil.copytree(assets, out_dir / "assets", dirs_exist_ok=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(exist_ok=True)
    data_files: dict[tuple[str, str], str] = {}
    for path, run in zip(files, runs, strict=True):
        name = _data_name(path)
        if name == path.name:
            shutil.copyfile(path, data_dir / name)
        else:
            (data_dir / name).write_text(read_results_text(path), encoding="utf-8")
        if matrix.run_for(run.engine.name, run.engine.version) is run:
            data_files[(run.engine.name, run.engine.version)] = f"data/{name}"
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")
    schema = default_spec_dir() / "matrix.schema.json"
    if schema.is_file():  # published at the schema's $id
        (out_dir / "spec").mkdir(exist_ok=True)
        shutil.copyfile(schema, out_dir / "spec" / "matrix.schema.json")
    fixture_models = {i: fx.models for i, fx in corpus.fixtures.items()}
    (out_dir / "matrix.json").write_text(
        _pretty(matrix_json(matrix, _family_info(corpus), fixture_models)) + "\n", encoding="utf-8"
    )

    family_names = {slug: fam.name for slug, fam in corpus.families.items()}
    engine_checks = {
        (r.engine.name, r.engine.version): {k.check: k for k in check_stats(r.cases, r.run.synthetic_strategies)}
        for r in matrix.runs
    }
    all_checks = [c for c in CHECKS if any(c in m for m in engine_checks.values())]
    latest = max((r.run.finished_at for r in matrix.runs), key=_parse_time, default=None)
    common = {
        "version": __version__,
        "status_values": [s.value for s in Status],
        "family_names": family_names,
        "site_url": base_url,
    }

    index = out_dir / "index.html"
    index.write_text(
        env.get_template("index.html").render(
            root="",
            matrix=matrix,
            cell_path=cell_path,
            data_files=data_files,
            engine_checks=engine_checks,
            all_checks=all_checks,
            latest=latest,
            page_url=base_url,
            **common,
        ),
        encoding="utf-8",
    )

    cell_tpl = env.get_template("cell.html")
    for cell in matrix.cells:
        cell_run = matrix.run_for(cell.engine, cell.engine_version)
        assert cell_run is not None
        syn = cell_run.run.synthetic_strategies
        views = [
            case_view(c, cell.engine, corpus.fixtures, corpus.root, syn)
            for c in cell.cases
            if case_status(c, syn) is not Status.PASS or any(k.status is not Status.PASS for k in c.checks)
        ]
        order = {Status.FAIL: 0, Status.ERROR: 1, Status.SOFT_PASS: 2, Status.UNSUPPORTED: 3, Status.PASS: 4}
        views.sort(key=lambda v: (order[v.status], v.fixture_id))
        rel = cell_path(cell.engine, cell.engine_version, cell.family)
        page = out_dir / rel
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(
            cell_tpl.render(
                root="../" * rel.count("/"),
                cell=cell,
                run=cell_run,
                views=views,
                skipped=skipped_summary(cell.cases),
                data_file=data_files.get((cell.engine, cell.engine_version)),
                corpus_note=corpus.note,
                digest_matches=_digest_matches(cell_run, corpus),
                have_corpus=bool(corpus.fixtures),
                page_url=f"{base_url}{rel}" if base_url else None,
                **common,
            ),
            encoding="utf-8",
        )
    if base_url:
        pages = [base_url] + [f"{base_url}{cell_path(c.engine, c.engine_version, c.family)}" for c in matrix.cells]
        (out_dir / "sitemap.xml").write_text(_sitemap(pages, latest), encoding="utf-8")
    return index


def _site_base_url(site_url: str | None) -> str | None:
    """``site_url`` with a trailing slash; rejects anything but an absolute http(s) URL."""
    if site_url is None:
        return None
    parts = urlsplit(site_url)
    if parts.scheme not in ("http", "https") or not parts.netloc or parts.query or parts.fragment:
        raise ValueError(f"--site-url must be an absolute http(s) URL without query or fragment: {site_url!r}")
    return site_url if site_url.endswith("/") else site_url + "/"


def _sitemap(urls: Sequence[str], latest: str | None) -> str:
    """A sitemaps.org urlset; ``lastmod`` is the newest run's date when known."""
    from xml.sax.saxutils import escape

    when = _parse_time(latest) if latest else None
    lastmod = f"<lastmod>{when.date().isoformat()}</lastmod>" if when and when.year > 1 else ""
    rows = "".join(f"  <url><loc>{escape(u)}</loc>{lastmod}</url>\n" for u in urls)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{rows}</urlset>\n"
    )

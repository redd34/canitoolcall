"""Conformance checks applied to an engine's observations of one fixture.

Every check takes the fixture, its family metadata and the
:class:`~canitoolcall.results.Observation` (one non-streaming parse plus one
parse per chunking strategy) and returns :class:`CheckResult` rows. The runner
calls :func:`run_checks` and derives the case status with :func:`case_status`
(the worst status over rows of *realistic* strategies: ``char:*`` never counts,
and multi-token strategies do not count for engines that stream one token per
event; see :attr:`Observation.synthetic <canitoolcall.results.Observation.synthetic>`).

Comparison policy (spec/README.md, "Normalization policy soft-v1"):

* strict: ``""`` -> ``None``; content/reasoning compared exactly; tool-call
  names compared exactly; arguments compared as parsed JSON (key order and
  whitespace ignored, value types significant: ``3 != "3"``, ``true != 1``;
  ``1`` and ``1.0`` are the same JSON number).
* soft (``soft-v1``): additionally strip leading/trailing whitespace from
  content/reasoning; whitespace-only -> ``None``.
* strict-equal -> PASS; soft-only-equal -> SOFT_PASS; otherwise FAIL.

Arguments that are not valid JSON (including ``""`` for a no-argument call,
``NaN``/``Infinity``, objects with a duplicate key, and a non-string
``arguments_raw``) never equal a valid parse. Two parses that raised compare
equal when the exception *type* matches (messages may differ).

To check a single parse (e.g. from a pytest plugin), call :func:`run_checks`
with ``Observation(nonstream=result, streams={})``: the stream-comparison
checks then produce no rows.

Checks never raise on bad engine output (invalid JSON arguments, exceptions
recorded in ``ParseResult.exception``); they report FAIL with a ``detail``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from functools import lru_cache
from typing import Any

from canitoolcall.chunking import ChunkStrategy
from canitoolcall.fixtures import Expected, Family, Fixture
from canitoolcall.results import CheckResult, Observation, ParsedToolCall, ParseResult, Status, worst_status

NORMALIZATION = "soft-v1"
"""Name of the normalization policy recorded in results."""

NONSTREAM = "nonstream"
"""Strategy label used for the non-streaming parse."""

ALL_STREAMS = "*"
"""Strategy label for checks that compare all streams with each other."""

CheckFn = Callable[[Fixture, Family | None, Observation], list[CheckResult]]

_DETAIL_MAX = 160


# --------------------------------------------------------------------------- normalization


def normalize_text(value: str | None, *, soft: bool) -> str | None:
    """Apply the strict or soft (``soft-v1``) text normalization.

    >>> normalize_text("", soft=False) is None
    True
    >>> normalize_text("  \\n", soft=True) is None
    True
    >>> normalize_text(" a\\n", soft=False)
    ' a\\n'
    """
    if value is None:
        return None
    if soft:
        value = value.strip()
    return value or None


class _InvalidJSON:
    """Marker for arguments that do not decode as JSON (keeps the raw text)."""

    __slots__ = ("raw",)

    def __init__(self, raw: str) -> None:
        self.raw = raw

    @classmethod
    def of(cls, raw: object) -> _InvalidJSON:
        """Marker for any raw value; non-strings are kept as their ``repr``."""
        return cls(raw if isinstance(raw, str) else repr(raw))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _InvalidJSON) and other.raw == self.raw

    def __hash__(self) -> int:
        return hash(("invalid-json", self.raw))

    def __repr__(self) -> str:
        return f"<invalid JSON {self.raw!r}>"


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def decode_arguments(raw: object) -> Any:
    """Strictly decode ``arguments_raw``; raises ``ValueError`` if it is not JSON.

    Unlike :func:`json.loads`, ``NaN``/``Infinity`` and objects with a
    duplicate key are rejected (clients may keep either value), and a
    non-string ``raw`` (e.g. ``None`` from an engine) is invalid too.
    """
    if not isinstance(raw, str):
        raise ValueError(f"arguments are {type(raw).__name__}, not JSON text")
    return json.loads(raw, parse_constant=_reject_constant, object_pairs_hook=_reject_duplicate_keys)


def _canon_json(value: Any) -> Any:
    """Hashable, type-tagged form of a decoded JSON value.

    Tags keep ``True`` distinct from ``1`` (Python treats them as equal).
    Integral floats compare equal to ints: JSON has a single number type.
    """
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("num", value)
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return ("num", int(value))
        return ("num", value)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, Mapping):
        return ("obj", tuple(sorted(((str(k), _canon_json(v)) for k, v in value.items()), key=lambda kv: kv[0])))
    if isinstance(value, (list, tuple)):
        return ("arr", tuple(_canon_json(v) for v in value))
    raise TypeError(f"not a JSON value: {type(value).__name__}")


def canonical_arguments(raw: object) -> Any:
    """Comparable form of ``arguments_raw``: canonical JSON, or an invalid marker."""
    try:
        return _canon_json(decode_arguments(raw))
    except ValueError:
        return _InvalidJSON.of(raw)


def _exception_type(exc: str | None) -> str | None:
    if exc is None:
        return None
    return exc.split(":", 1)[0].strip() or exc


def canonical(result: ParseResult, *, soft: bool) -> dict[str, Any]:
    """Comparable form of a ParseResult: normalized text fields, tool calls as
    ``[(name, parsed_arguments_or_marker)]`` in order, and the exception type.

    Invalid JSON arguments are kept as a distinct marker so two results with
    the same invalid text compare equal but never equal a valid parse.
    """
    return {
        "content": normalize_text(result.content, soft=soft),
        "reasoning_content": normalize_text(result.reasoning_content, soft=soft),
        "tool_calls": [(tc.name, canonical_arguments(tc.arguments_raw)) for tc in result.tool_calls],
        "exception": _exception_type(result.exception),
    }


def expected_as_result(expected: Expected) -> ParseResult:
    """The ParseResult a perfect engine returns for ``expected``."""
    return ParseResult(
        content=expected.content,
        reasoning_content=expected.reasoning_content,
        tool_calls=tuple(
            ParsedToolCall(tc.name, json.dumps(tc.arguments, ensure_ascii=False)) for tc in expected.tool_calls
        ),
    )


def compare(a: ParseResult, b: ParseResult) -> Status:
    """PASS if strictly equal, SOFT_PASS if equal only under soft-v1, else FAIL."""
    if canonical(a, soft=False) == canonical(b, soft=False):
        return Status.PASS
    if canonical(a, soft=True) == canonical(b, soft=True):
        return Status.SOFT_PASS
    return Status.FAIL


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= _DETAIL_MAX else text[: _DETAIL_MAX - 3] + "..."


def describe_difference(want: ParseResult, got: ParseResult, *, want_label: str = "expected") -> str:
    """Human-readable, field-by-field difference (strict comparison)."""
    parts: list[str] = []
    if _exception_type(want.exception) != _exception_type(got.exception):
        parts.append(f"exception: {want_label} {_short(want.exception)}, got {_short(got.exception)}")
    for field in ("content", "reasoning_content"):
        w = normalize_text(getattr(want, field), soft=False)
        g = normalize_text(getattr(got, field), soft=False)
        if w != g:
            ws = " (whitespace only)" if normalize_text(w, soft=True) == normalize_text(g, soft=True) else ""
            parts.append(f"{field}{ws}: {want_label} {_short(w)}, got {_short(g)}")
    wn = [tc.name for tc in want.tool_calls]
    gn = [tc.name for tc in got.tool_calls]
    if wn != gn:
        parts.append(f"tool_calls: {want_label} {wn}, got {gn}")
    else:
        for i, (wt, gt) in enumerate(zip(want.tool_calls, got.tool_calls, strict=True)):
            if canonical_arguments(wt.arguments_raw) != canonical_arguments(gt.arguments_raw):
                w_args, g_args = _short(wt.arguments_raw), _short(gt.arguments_raw)
                parts.append(f"tool_calls[{i}].arguments: {want_label} {w_args}, got {g_args}")
    return "; ".join(parts) or "equal"


# --------------------------------------------------------------------------- helpers


def is_synthetic(strategy: str, synthetic: Collection[str] = ()) -> bool:
    """True for strategy labels that do not count toward a case's status.

    ``char:*`` never counts; ``synthetic`` adds the engine-specific ones
    (multi-token strategies on one-token-per-step engines, see
    :func:`canitoolcall.chunking.synthetic_strategies`). ``nonstream``, ``*``
    and unknown labels are treated as realistic, so a row is never silently
    hidden.
    """
    if strategy in (NONSTREAM, ALL_STREAMS):
        return False
    if strategy in synthetic:
        return True
    try:
        return not ChunkStrategy.parse(strategy).realistic
    except ValueError:
        return False


def case_status(rows: Sequence[CheckResult], synthetic: Collection[str] = ()) -> Status:
    """Worst status over the rows of realistic strategies (see :func:`is_synthetic`)."""
    if any(r.status is Status.UNSUPPORTED for r in rows):
        raise ValueError("unsupported belongs to adapter applicability, not a scored CheckResult")
    return worst_status(r.status for r in rows if not is_synthetic(r.strategy, synthetic))


def _parses(obs: Observation) -> Iterator[tuple[str, ParseResult]]:
    """Every parse, labelled: ``nonstream`` first, then streams in order."""
    yield NONSTREAM, obs.nonstream
    yield from obs.streams.items()


def _row(check: str, strategy: str, status: Status, detail: str | None = None) -> CheckResult:
    return CheckResult(check, strategy, status, detail)


def _iter_strings(value: Any) -> Iterator[str]:
    """All string values (and object keys) inside a decoded JSON value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for k, v in value.items():
            yield str(k)
            yield from _iter_strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _iter_strings(v)


def _offered_tools(fixture: Fixture) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for tool in fixture.tools:
        fn = tool.get("function", tool)
        if isinstance(fn, Mapping) and isinstance(fn.get("name"), str):
            out[fn["name"]] = fn
    return out


# --------------------------------------------------------------------------- checks


def check_expected_match(fixture: Fixture, family: Family | None, obs: Observation) -> list[CheckResult]:
    """``expected_match``: each parse (nonstream + every strategy) vs ``fixture.expected``.

    Returns no rows when the fixture uses ``expected_error`` instead.
    """
    if fixture.expected is None:
        return []
    want = expected_as_result(fixture.expected)
    rows: list[CheckResult] = []
    for label, got in _parses(obs):
        if got.exception is not None:
            rows.append(_row("expected_match", label, Status.FAIL, f"parser raised {_short(got.exception)}"))
            continue
        status = compare(want, got)
        rows.append(
            _row("expected_match", label, status, None if status is Status.PASS else describe_difference(want, got))
        )
    return rows


def _error_outcomes(fixture: Fixture, got: ParseResult) -> tuple[set[str], bool]:
    """Outcomes a parse exhibits, and whether content passthrough is only soft."""
    if got.exception is not None:
        return {"exception"}, False
    outcomes = {"no_tool_calls"}
    soft_only = False
    if normalize_text(got.content, soft=False) == normalize_text(fixture.raw_output, soft=False):
        outcomes.add("content_passthrough")
    elif normalize_text(got.content, soft=True) == normalize_text(fixture.raw_output, soft=True):
        outcomes.add("content_passthrough")
        soft_only = True
    return outcomes, soft_only


def check_expected_error(fixture: Fixture, family: Family | None, obs: Observation) -> list[CheckResult]:
    """``expected_error``: each parse's outcome must be in ``expected_error.accept``.

    Outcomes: ``exception`` (``ParseResult.exception`` set), ``no_tool_calls``
    (no calls, no exception), ``content_passthrough`` (no calls and content ==
    raw_output modulo surrounding whitespace; a whitespace-only difference is
    SOFT_PASS when it is the only accepted outcome that matched). Any returned
    tool call fails, also when the parse raised afterwards: a streaming client
    had already received that (partial) call.
    """
    err = fixture.expected_error
    if err is None:
        return []
    accept = set(err.accept)
    rows: list[CheckResult] = []
    for label, got in _parses(obs):
        if got.tool_calls:
            names = [tc.name for tc in got.tool_calls]
            raised = f" (then raised {_short(got.exception)})" if got.exception is not None else ""
            rows.append(
                _row(
                    "expected_error",
                    label,
                    Status.FAIL,
                    f"returned {len(names)} tool call(s) {names}{raised} for {err.reason}",
                )
            )
            continue
        outcomes, soft_only = _error_outcomes(fixture, got)
        matched = outcomes & accept
        if not matched:
            seen = (
                "exception " + _short(got.exception) if got.exception else "no tool calls, content not passed through"
            )
            rows.append(_row("expected_error", label, Status.FAIL, f"outcome {seen}; accepted: {sorted(accept)}"))
        elif matched == {"content_passthrough"} and soft_only:
            rows.append(_row("expected_error", label, Status.SOFT_PASS, "content passthrough differs in whitespace"))
        else:
            rows.append(_row("expected_error", label, Status.PASS))
    return rows


def check_stream_equals_nonstream(fixture: Fixture, family: Family | None, obs: Observation) -> list[CheckResult]:
    """``stream_equals_nonstream``: one row per strategy, via :func:`compare`."""
    rows: list[CheckResult] = []
    for label, got in obs.streams.items():
        status = compare(obs.nonstream, got)
        detail = None if status is Status.PASS else describe_difference(obs.nonstream, got, want_label="nonstream")
        rows.append(_row("stream_equals_nonstream", label, status, detail))
    return rows


def check_split_invariance(fixture: Fixture, family: Family | None, obs: Observation) -> list[CheckResult]:
    """``split_invariance``: all realistic streaming results equal each other.

    One row (strategy ``"*"``) naming the strategies that disagree with the
    ``one`` (or first) stream in ``detail``. No row with fewer than two
    realistic streams. Synthetic streams (``char:*`` and ``obs.synthetic``)
    are left out: they are judged only by ``stream_equals_nonstream``.
    """
    streams = {k: v for k, v in obs.streams.items() if not is_synthetic(k, obs.synthetic)}
    if len(streams) < 2:
        return []
    ref_id = "one" if "one" in streams else "token" if "token" in streams else next(iter(streams))
    ref = streams[ref_id]
    statuses: list[Status] = []
    diffs: list[str] = []
    for label, got in streams.items():
        if label == ref_id:
            continue
        st = compare(ref, got)
        statuses.append(st)
        if st is not Status.PASS:
            diffs.append(f"{label} ({st.value}): {describe_difference(ref, got, want_label=ref_id)}")
    status = worst_status(statuses)
    detail = None if not diffs else f"vs {ref_id}: " + " | ".join(diffs)
    return [_row("split_invariance", ALL_STREAMS, status, detail)]


def check_no_leakage(fixture: Fixture, family: Family | None, obs: Observation) -> list[CheckResult]:
    """``no_leakage``: no ``family.markers`` string appears in content,
    reasoning_content, tool names or argument strings (values and keys),
    unless the corresponding expected field contains it verbatim. Skipped (no
    rows) when ``family`` is None or has no markers.

    For ``expected_error`` fixtures content may contain markers (passing the
    raw text through is a graceful outcome), so only reasoning and tool calls
    are checked.
    """
    if family is None or not family.markers:
        return []
    markers = tuple(family.markers)
    exp = fixture.expected
    if exp is not None:
        allowed_content = {m for m in markers if exp.content and m in exp.content}
        allowed_reasoning = {m for m in markers if exp.reasoning_content and m in exp.reasoning_content}
        exp_strings = [s for tc in exp.tool_calls for s in _iter_strings(tc.arguments)]
        allowed_args = {m for m in markers if any(m in s for s in exp_strings)}
    else:
        allowed_content, allowed_reasoning, allowed_args = set(markers), set(), set()

    rows: list[CheckResult] = []
    for label, got in _parses(obs):
        leaks: list[str] = []

        def scan(where: str, text: str | None, allowed: set[str], leaks: list[str] = leaks) -> None:
            if text:
                leaks.extend(f"{where} contains {m!r}" for m in markers if m not in allowed and m in text)

        scan("content", got.content, allowed_content)
        scan("reasoning_content", got.reasoning_content, allowed_reasoning)
        for i, tc in enumerate(got.tool_calls):
            scan(f"tool_calls[{i}].name", tc.name, set())
            try:
                strings = list(_iter_strings(decode_arguments(tc.arguments_raw)))
            except ValueError:
                strings = [tc.arguments_raw] if isinstance(tc.arguments_raw, str) else []
            for s in strings:
                scan(f"tool_calls[{i}].arguments", s, allowed_args)
        unique = list(dict.fromkeys(leaks))
        rows.append(_row("no_leakage", label, Status.FAIL if unique else Status.PASS, "; ".join(unique) or None))
    return rows


def _argument_problem(tc: ParsedToolCall) -> str | None:
    try:
        value = decode_arguments(tc.arguments_raw)
    except ValueError as e:
        return f"{tc.name}: arguments are not valid JSON text ({_short(tc.arguments_raw)}: {e})"
    if not isinstance(value, dict):
        return f"{tc.name}: arguments decode to {type(value).__name__}, not an object ({_short(tc.arguments_raw)})"
    return None


def check_arguments_json(fixture: Fixture, family: Family | None, obs: Observation) -> list[CheckResult]:
    """``arguments_json``: every ``arguments_raw`` decodes to a JSON object.

    One row per parse that returned at least one tool call. ``""`` for a
    no-argument call fails (clients ``json.loads`` it); a double-encoded
    string (``"{\\"a\\": 1}"``) fails too.
    """
    rows: list[CheckResult] = []
    for label, got in _parses(obs):
        if not got.tool_calls:
            continue
        problems = [f"[{i}] {p}" for i, tc in enumerate(got.tool_calls) if (p := _argument_problem(tc))]
        rows.append(
            _row("arguments_json", label, Status.FAIL if problems else Status.PASS, "; ".join(problems) or None)
        )
    return rows


@lru_cache(maxsize=512)
def _validator(schema_json: str) -> Any:
    import jsonschema  # lazy: never needed inside engine venvs

    schema = json.loads(schema_json)
    cls = jsonschema.validators.validator_for(schema, default=jsonschema.Draft202012Validator)
    cls.check_schema(schema)
    return cls(schema)


def schema_errors(parameters: Mapping[str, Any] | None, arguments: Any) -> list[str]:
    """Validation messages for ``arguments`` against a tool's ``parameters`` schema.

    A missing schema accepts any object. Raises ``jsonschema.SchemaError`` if
    the schema itself is invalid.
    """
    schema = dict(parameters) if parameters else {"type": "object"}
    validator = _validator(json.dumps(schema, sort_keys=True))
    return [f"{e.json_path}: {e.message}" for e in sorted(validator.iter_errors(arguments), key=lambda e: list(e.path))]


def check_arguments_schema(fixture: Fixture, family: Family | None, obs: Observation) -> list[CheckResult]:
    """``arguments_schema``: each call names an offered tool and its arguments
    validate against that tool's ``parameters`` JSON Schema (``jsonschema``,
    imported lazily).

    One row per parse with at least one tool call. If a call's arguments equal
    the expected call's arguments at the same position, schema errors are not
    the engine's fault (the model emitted them) and the call passes. An
    invalid tool schema is a fixture problem and yields ERROR.
    """
    import jsonschema

    tools = _offered_tools(fixture)
    expected_args = (
        [_canon_json(tc.arguments) for tc in fixture.expected.tool_calls] if fixture.expected is not None else []
    )
    rows: list[CheckResult] = []
    for label, got in _parses(obs):
        if not got.tool_calls:
            continue
        problems: list[str] = []
        fixture_problems: list[str] = []
        for i, tc in enumerate(got.tool_calls):
            fn = tools.get(tc.name)
            if fn is None:
                problems.append(f"[{i}] {tc.name!r} is not an offered tool")
                continue
            if _argument_problem(tc) is not None:
                problems.append(f"[{i}] {tc.name}: arguments are not a JSON object; not validated")
                continue
            args = decode_arguments(tc.arguments_raw)
            try:
                errs = schema_errors(fn.get("parameters"), args)
            except jsonschema.SchemaError as e:
                fixture_problems.append(f"[{i}] {tc.name}: tool schema is invalid ({e.message})")
                continue
            if errs and not (i < len(expected_args) and expected_args[i] == _canon_json(args)):
                problems.append(f"[{i}] {tc.name}: " + "; ".join(errs))
        status = Status.FAIL if problems else Status.ERROR if fixture_problems else Status.PASS
        rows.append(_row("arguments_schema", label, status, "; ".join(problems + fixture_problems) or None))
    return rows


def check_parallel_order(fixture: Fixture, family: Family | None, obs: Observation) -> list[CheckResult]:
    """``parallel_order``: call count and name order equal ``expected``.

    Only for fixtures whose expected result has 2+ tool calls.
    """
    if fixture.expected is None or len(fixture.expected.tool_calls) < 2:
        return []
    want = [tc.name for tc in fixture.expected.tool_calls]
    rows: list[CheckResult] = []
    for label, got in _parses(obs):
        names = [tc.name for tc in got.tool_calls]
        if names == want:
            rows.append(_row("parallel_order", label, Status.PASS))
        elif sorted(names) == sorted(want):
            rows.append(_row("parallel_order", label, Status.FAIL, f"order: expected {want}, got {names}"))
        else:
            rows.append(
                _row(
                    "parallel_order", label, Status.FAIL, f"expected {len(want)} calls {want}, got {len(names)} {names}"
                )
            )
    return rows


ALL_CHECKS: Sequence[tuple[str, CheckFn]] = (
    ("expected_match", check_expected_match),
    ("expected_error", check_expected_error),
    ("stream_equals_nonstream", check_stream_equals_nonstream),
    ("split_invariance", check_split_invariance),
    ("no_leakage", check_no_leakage),
    ("arguments_json", check_arguments_json),
    ("arguments_schema", check_arguments_schema),
    ("parallel_order", check_parallel_order),
)
"""Registry in reporting order; names match results.schema.json."""


def run_checks(fixture: Fixture, family: Family | None, obs: Observation) -> list[CheckResult]:
    """Run every check in :data:`ALL_CHECKS` and concatenate their rows.

    Raises only on a harness problem (an invalid tool schema is reported as
    an ERROR row); callers treat an exception as a harness error.
    """
    rows: list[CheckResult] = []
    for _, fn in ALL_CHECKS:
        rows.extend(fn(fixture, family, obs))
    return rows

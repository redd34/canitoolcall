"""Unit tests for canitoolcall.checks, using hand-built ParseResults.

The fixture under test is the template-rendered Qwen3 sample (and test-only
slices of it, see core_corpus.py); the engine outputs are constructed here.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from core_corpus import sample_family, sample_record, single_call_record, truncated_record

from canitoolcall.checks import (
    ALL_CHECKS,
    NONSTREAM,
    RECOVERED_TEXT_ONLY,
    canonical,
    canonical_arguments,
    case_status,
    check_arguments_json,
    check_arguments_schema,
    check_expected_error,
    check_expected_match,
    check_no_leakage,
    check_parallel_order,
    check_split_invariance,
    check_stream_equals_nonstream,
    compare,
    compare_modes,
    decode_arguments,
    describe_difference,
    expected_as_result,
    is_synthetic,
    normalize_text,
    run_checks,
    schema_errors,
)
from canitoolcall.fixtures import Family, Fixture
from canitoolcall.results import CheckResult, Observation, ParsedToolCall, ParseResult, Status

SPEC_CHECKS = {
    "expected_match",
    "expected_error",
    "stream_equals_nonstream",
    "split_invariance",
    "no_leakage",
    "arguments_json",
    "arguments_schema",
    "parallel_order",
}


@pytest.fixture
def fx() -> Fixture:
    return Fixture.from_dict(sample_record())


@pytest.fixture
def fam() -> Family:
    return Family.from_dict(sample_family())


@pytest.fixture
def good(fx: Fixture) -> ParseResult:
    assert fx.expected is not None
    return expected_as_result(fx.expected)


def obs(nonstream: ParseResult, **streams: ParseResult) -> Observation:
    return Observation(nonstream=nonstream, streams={k.replace("_", ":"): v for k, v in streams.items()})


def by_strategy(rows: list[CheckResult]) -> dict[str, Status]:
    return {r.strategy: r.status for r in rows}


def with_call(result: ParseResult, i: int, **changes: str) -> ParseResult:
    calls = list(result.tool_calls)
    calls[i] = replace(calls[i], **changes)
    return replace(result, tool_calls=tuple(calls))


# --------------------------------------------------------------------------- normalization


@pytest.mark.parametrize(
    ("value", "soft", "want"),
    [
        (None, False, None),
        ("", False, None),
        ("", True, None),
        ("  \n", False, "  \n"),
        ("  \n", True, None),
        (" a\n", False, " a\n"),
        (" a\n", True, "a"),
        ("a  b", True, "a  b"),  # inner whitespace is never touched
    ],
)
def test_normalize_text(value: str | None, soft: bool, want: str | None) -> None:
    assert normalize_text(value, soft=soft) == want


def test_registry_matches_results_schema() -> None:
    assert [name for name, _ in ALL_CHECKS] == [
        "expected_match",
        "expected_error",
        "stream_equals_nonstream",
        "split_invariance",
        "no_leakage",
        "arguments_json",
        "arguments_schema",
        "parallel_order",
    ]
    assert set(dict(ALL_CHECKS)) == SPEC_CHECKS


# --------------------------------------------------------------------------- argument comparison


@pytest.mark.parametrize(
    ("a", "b", "equal"),
    [
        ('{"a": 1, "b": 2}', '{"b":2,"a":1}', True),  # key order and whitespace
        ('{"a": 1}', '{"a": 1.0}', True),  # one JSON number type
        ('{"a": 3}', '{"a": "3"}', False),  # types matter
        ('{"a": true}', '{"a": 1}', False),  # bool is not a number (Python says True == 1)
        ('{"a": false}', '{"a": 0}', False),
        ('{"a": null}', "{}", False),
        ('{"a": [1, 2]}', '{"a": [2, 1]}', False),  # array order matters
        ('{"a": {"x": [1, {"y": "é"}]}}', '{"a":{"x":[1,{"y":"\\u00e9"}]}}', True),  # escapes decode
        ('{"a": 1.5}', '{"a": 1.50}', True),
        ("", "{}", False),  # empty string is not a no-argument object
        ("", "", True),  # same invalid text compares equal
        ("{bad", "{bad", True),
        ("{bad", "{worse", False),
        ('{"a": NaN}', '{"a": NaN}', True),  # invalid (RFC 8259), but identical
    ],
)
def test_canonical_arguments(a: str, b: str, equal: bool) -> None:
    assert (canonical_arguments(a) == canonical_arguments(b)) is equal


def test_decode_arguments_rejects_non_json_constants() -> None:
    assert decode_arguments('{"a": 1}') == {"a": 1}
    for bad in ('{"a": NaN}', '{"a": Infinity}', "-Infinity", ""):
        with pytest.raises(ValueError):
            decode_arguments(bad)


def test_invalid_never_equals_valid() -> None:
    assert canonical_arguments('"{}"') != canonical_arguments("{}")  # double-encoded
    assert canonical_arguments("{") != canonical_arguments("{}")


# --------------------------------------------------------------------------- compare


def test_compare_strict_soft_fail(good: ParseResult) -> None:
    assert compare(good, good) is Status.PASS
    assert compare(good, replace(good, content="")) is Status.PASS  # "" == None
    assert compare(good, replace(good, content="\n\n")) is Status.SOFT_PASS
    assert compare(good, replace(good, reasoning_content="\nI should call the tools.\n")) is Status.SOFT_PASS
    assert compare(good, replace(good, reasoning_content="I should call the tools")) is Status.FAIL
    assert compare(good, replace(good, tool_calls=good.tool_calls[:1])) is Status.FAIL
    assert compare(good, replace(good, tool_calls=good.tool_calls[::-1])) is Status.FAIL
    assert compare(good, with_call(good, 0, name="get_weather2")) is Status.FAIL


def test_compare_exceptions_by_type() -> None:
    a = ParseResult(exception="ValueError: at char 3")
    assert compare(a, ParseResult(exception="ValueError: at char 9")) is Status.PASS
    assert compare(a, ParseResult(exception="KeyError: 'name'")) is Status.FAIL
    assert compare(a, ParseResult()) is Status.FAIL


def test_canonical_shape(good: ParseResult) -> None:
    c = canonical(replace(good, content=" x "), soft=True)
    assert c["content"] == "x"
    assert [name for name, _ in c["tool_calls"]] == ["get_weather", "search"]
    assert c["exception"] is None


def test_describe_difference(good: ParseResult) -> None:
    got = with_call(replace(good, content="\n"), 1, arguments_raw='{"query": "x"}')
    text = describe_difference(good, got)
    assert "content (whitespace only): expected None, got '\\n'" in text
    assert "tool_calls[1].arguments" in text
    assert "whitespace only" in describe_difference(good, replace(good, reasoning_content=" I should call the tools."))
    assert describe_difference(good, good) == "equal"
    long = "x" * 1000
    assert len(describe_difference(good, replace(good, content=long))) < 400


# --------------------------------------------------------------------------- expected_match


def test_expected_match_pass(fx: Fixture, good: ParseResult) -> None:
    rows = check_expected_match(fx, None, obs(good, one=good, token=good))
    assert by_strategy(rows) == {NONSTREAM: Status.PASS, "one": Status.PASS, "token": Status.PASS}
    assert all(r.detail is None for r in rows)


def test_expected_match_per_strategy(fx: Fixture, good: ParseResult) -> None:
    lost = replace(good, tool_calls=())
    ws = replace(good, content="\n\n")
    boom = ParseResult(exception="JSONDecodeError: Expecting value")
    rows = check_expected_match(fx, None, obs(good, one=lost, token=ws, rand_1_8=boom))
    assert by_strategy(rows) == {
        NONSTREAM: Status.PASS,
        "one": Status.FAIL,
        "token": Status.SOFT_PASS,
        "rand:1:8": Status.FAIL,
    }
    details = {r.strategy: r.detail or "" for r in rows}
    assert "tool_calls: expected ['get_weather', 'search'], got []" in details["one"]
    assert "whitespace only" in details["token"]
    assert "parser raised" in details["rand:1:8"]


def test_expected_match_arguments_compared_as_json(fx: Fixture, good: ParseResult) -> None:
    reordered = with_call(good, 1, arguments_raw='{"filters":{"max":3,"tags":["a","b"]},"query":"café \\"best\\""}')
    assert by_strategy(check_expected_match(fx, None, obs(reordered))) == {NONSTREAM: Status.PASS}
    stringly = with_call(
        good, 1, arguments_raw='{"query": "café \\"best\\"", "filters": {"tags": ["a", "b"], "max": "3"}}'
    )
    assert by_strategy(check_expected_match(fx, None, obs(stringly))) == {NONSTREAM: Status.FAIL}


def test_expected_match_skipped_for_expected_error() -> None:
    t = Fixture.from_dict(truncated_record())
    assert check_expected_match(t, None, obs(ParseResult())) == []


# --------------------------------------------------------------------------- expected_error


@pytest.fixture
def trunc() -> Fixture:
    return Fixture.from_dict(truncated_record())


def with_accept(fx: Fixture, *accept: str) -> Fixture:
    assert fx.expected_error is not None
    return replace(fx, expected_error=replace(fx.expected_error, accept=tuple(accept)))  # type: ignore[arg-type]


def test_expected_error_outcomes(trunc: Fixture) -> None:
    raw = trunc.raw_output
    rows = check_expected_error(
        with_accept(trunc, "no_tool_calls"),
        None,
        obs(
            ParseResult(content="partial"),
            one=ParseResult(exception="ValueError: x"),
            token=ParseResult(content=raw, tool_calls=(ParsedToolCall("get_weather", "{}"),)),
        ),
    )
    assert by_strategy(rows) == {NONSTREAM: Status.PASS, "one": Status.FAIL, "token": Status.FAIL}
    assert "returned 1 tool call(s)" in (rows[2].detail or "")


def test_expected_error_exception_accepted(trunc: Fixture) -> None:
    rows = check_expected_error(with_accept(trunc, "exception"), None, obs(ParseResult(exception="ValueError: x")))
    assert by_strategy(rows) == {NONSTREAM: Status.PASS}
    rows = check_expected_error(with_accept(trunc, "exception"), None, obs(ParseResult()))
    assert by_strategy(rows) == {NONSTREAM: Status.FAIL}


def test_expected_error_content_passthrough(trunc: Fixture) -> None:
    raw = trunc.raw_output
    f = with_accept(trunc, "content_passthrough")
    rows = check_expected_error(
        f,
        None,
        obs(ParseResult(content=raw), one=ParseResult(content=raw + "\n"), token=ParseResult(content="other")),
    )
    assert by_strategy(rows) == {NONSTREAM: Status.PASS, "one": Status.SOFT_PASS, "token": Status.FAIL}
    # A whitespace difference is fully fine when no_tool_calls is also accepted.
    both = with_accept(trunc, "content_passthrough", "no_tool_calls")
    assert by_strategy(check_expected_error(both, None, obs(ParseResult(content=raw + "\n")))) == {
        NONSTREAM: Status.PASS
    }


def test_expected_error_skipped_for_expected(fx: Fixture) -> None:
    assert check_expected_error(fx, None, obs(ParseResult())) == []


# --------------------------------------------------------------------------- stream comparisons


def test_stream_equals_nonstream(fx: Fixture, good: ParseResult) -> None:
    rows = check_stream_equals_nonstream(
        fx, None, obs(good, one=replace(good, tool_calls=()), token=good, rand_2_8=replace(good, content=" "))
    )
    assert by_strategy(rows) == {"one": Status.FAIL, "token": Status.PASS, "rand:2:8": Status.SOFT_PASS}
    assert "nonstream" in (rows[0].detail or "")
    assert check_stream_equals_nonstream(fx, None, obs(good)) == []


def test_split_invariance(fx: Fixture, good: ParseResult) -> None:
    rows = check_split_invariance(fx, None, obs(good, one=good, token=good, rand_1_8=good))
    assert rows == [CheckResult("split_invariance", "*", Status.PASS, None)]

    lost = replace(good, tool_calls=())
    (row,) = check_split_invariance(fx, None, obs(good, one=lost, token=good, rand_1_8=good))
    assert row.status is Status.FAIL
    assert row.detail is not None and "vs one" in row.detail and "token (fail)" in row.detail

    (row,) = check_split_invariance(fx, None, obs(good, one=good, token=replace(good, content="\n")))
    assert row.status is Status.SOFT_PASS


def test_split_invariance_reference_and_minimum(fx: Fixture, good: ParseResult) -> None:
    assert check_split_invariance(fx, None, obs(good, one=good)) == []
    (row,) = check_split_invariance(fx, None, obs(good, token=good, rand_1_8=replace(good, tool_calls=())))
    assert row.detail is not None and row.detail.startswith("vs token")


def test_split_invariance_ignores_synthetic_streams(fx: Fixture, good: ParseResult) -> None:
    broken = replace(good, tool_calls=())
    (row,) = check_split_invariance(fx, None, obs(good, one=good, token=good, char_0=broken))
    assert row.status is Status.PASS
    # ...but the synthetic stream is still compared with the non-streaming parse.
    rows = check_stream_equals_nonstream(fx, None, obs(good, char_0=broken))
    assert by_strategy(rows) == {"char:0": Status.FAIL}


# Recovery contract section 3 (docs/formats/recovery.md): for expected_error fixtures, modes are compared on
# tool calls and exception type; a recovered-text-only difference is soft_pass, anything else stays fail.


def test_cross_mode_recovered_text_only_is_soft_pass(trunc: Fixture) -> None:
    kept = ParseResult(content=trunc.raw_output)
    dropped = ParseResult(content="I will check.")
    rows = check_stream_equals_nonstream(trunc, None, obs(kept, one=dropped, token=kept))
    assert by_strategy(rows) == {"one": Status.SOFT_PASS, "token": Status.PASS}
    soft = next(r for r in rows if r.strategy == "one")
    assert soft.detail is not None and soft.detail.startswith(f"{RECOVERED_TEXT_ONLY}: content")

    (row,) = check_split_invariance(trunc, None, obs(kept, one=kept, token=dropped))
    assert row.status is Status.SOFT_PASS
    assert row.detail is not None and f"token (soft_pass): {RECOVERED_TEXT_ONLY}: " in row.detail

    reasoned = ParseResult(reasoning_content="thinking", content=None)
    assert by_strategy(check_stream_equals_nonstream(trunc, None, obs(ParseResult(), one=reasoned))) == {
        "one": Status.SOFT_PASS
    }
    assert compare_modes(trunc, kept, dropped) is Status.SOFT_PASS
    assert compare_modes(trunc, kept, kept) is Status.PASS


def test_cross_mode_call_or_raise_difference_still_fails(trunc: Fixture, good: ParseResult) -> None:
    none = ParseResult(content=trunc.raw_output)
    raised = ParseResult(exception="ValueError: unterminated")
    other_raise = ParseResult(exception="KeyError: 'name'")
    rows = check_stream_equals_nonstream(trunc, None, obs(none, one=good, token=raised))
    assert by_strategy(rows) == {"one": Status.FAIL, "token": Status.FAIL}
    assert all(RECOVERED_TEXT_ONLY not in (r.detail or "") for r in rows)
    (row,) = check_split_invariance(trunc, None, obs(none, one=raised, token=other_raise))
    assert row.status is Status.FAIL
    # Same exception type with different messages is already equal.
    assert compare_modes(trunc, raised, ParseResult(exception="ValueError: other")) is Status.PASS


def test_cross_mode_compares_calls_exactly(trunc: Fixture, good: ParseResult) -> None:
    other_args = with_call(good, 0, arguments_raw='{"city": "Bern", "unit": "c"}')
    reordered = replace(good, tool_calls=tuple(reversed(good.tool_calls)))
    fewer = replace(good, tool_calls=good.tool_calls[:1])
    for variant in (other_args, reordered, fewer, replace(other_args, content="other text")):
        assert compare_modes(trunc, good, variant) is Status.FAIL
    # Identical calls, different text: softened like the no-call case (expected_error fails the calls anyway).
    assert compare_modes(trunc, good, replace(good, content="other text")) is Status.SOFT_PASS


def test_cross_mode_raise_vs_no_raise_fails_both_directions(trunc: Fixture) -> None:
    quiet = ParseResult(content="I will check.")
    raised = ParseResult(content="I will check.", exception="ValueError: unterminated")
    assert compare_modes(trunc, quiet, raised) is Status.FAIL
    assert compare_modes(trunc, raised, quiet) is Status.FAIL


def test_cross_mode_split_invariance_mixed_and_reasoning(trunc: Fixture, good: ParseResult) -> None:
    kept = ParseResult(content=trunc.raw_output)
    (row,) = check_split_invariance(trunc, None, obs(kept, one=kept, token=ParseResult(), rand_1_8=good))
    assert row.status is Status.FAIL
    assert row.detail is not None and "token (soft_pass)" in row.detail and "rand:1:8 (fail)" in row.detail
    (row,) = check_split_invariance(trunc, None, obs(kept, one=kept, token=ParseResult(reasoning_content="r")))
    assert row.status is Status.SOFT_PASS
    # A synthetic stream is still compared with nonstream, under the same rule.
    rows = check_stream_equals_nonstream(trunc, None, obs(kept, char_0=ParseResult()))
    assert by_strategy(rows) == {"char:0": Status.SOFT_PASS}


def test_cross_mode_expected_fixture_text_difference_still_fails(fx: Fixture, good: ParseResult) -> None:
    other_text = replace(good, content="something else")
    assert by_strategy(check_stream_equals_nonstream(fx, None, obs(good, one=other_text))) == {"one": Status.FAIL}
    (row,) = check_split_invariance(fx, None, obs(good, one=good, token=other_text))
    assert row.status is Status.FAIL
    assert compare_modes(fx, good, other_text) is Status.FAIL


def test_cross_mode_soft_pass_sets_case_status(trunc: Fixture) -> None:
    kept = ParseResult(content=trunc.raw_output)
    dropped = ParseResult()
    rows = run_checks(trunc, None, obs(kept, one=kept, token=dropped, rand_1_8=kept))
    assert case_status(rows) is Status.SOFT_PASS
    assert {r.check for r in rows if r.status is Status.SOFT_PASS} == {"stream_equals_nonstream", "split_invariance"}


# --------------------------------------------------------------------------- leakage


def test_no_leakage_pass_and_fail(fx: Fixture, fam: Family, good: ParseResult) -> None:
    leaky_reasoning = replace(good, reasoning_content="I should call the tools.\n</think>")
    leaky_content = replace(good, content="<tool_call>\n")
    leaky_args = with_call(good, 0, arguments_raw='{"city": "Zürich<|im_end|>", "unit": "c"}')
    leaky_name = with_call(good, 0, name="<tool_call>get_weather")
    rows = check_no_leakage(
        fx, fam, obs(good, one=leaky_reasoning, token=leaky_content, rand_1_8=leaky_args, rand_2_8=leaky_name)
    )
    got = by_strategy(rows)
    assert got == {
        NONSTREAM: Status.PASS,
        "one": Status.FAIL,
        "token": Status.FAIL,
        "rand:1:8": Status.FAIL,
        "rand:2:8": Status.FAIL,
    }
    details = {r.strategy: r.detail for r in rows}
    assert details["one"] == "reasoning_content contains '</think>'"
    assert details["token"] == "content contains '<tool_call>'"
    assert details["rand:1:8"] == "tool_calls[0].arguments contains '<|im_end|>'"
    assert "tool_calls[0].name contains '<tool_call>'" in (details["rand:2:8"] or "")


def test_no_leakage_scans_invalid_arguments_and_keys(fx: Fixture, fam: Family, good: ParseResult) -> None:
    invalid = with_call(good, 0, arguments_raw='{"city": "x"}</tool_call>')
    key = with_call(good, 0, arguments_raw='{"<think>": "x"}')
    rows = check_no_leakage(fx, fam, obs(invalid, one=key))
    assert by_strategy(rows) == {NONSTREAM: Status.FAIL, "one": Status.FAIL}


def test_no_leakage_allows_markers_the_expected_parse_contains(fx: Fixture, fam: Family) -> None:
    assert fx.expected is not None
    call = fx.expected.tool_calls[1]
    marker_args = {**call.arguments, "query": "explain <tool_call> tags"}
    exp = replace(
        fx.expected,
        content="use <think> tags",
        tool_calls=(fx.expected.tool_calls[0], replace(call, arguments=marker_args)),
    )
    f = replace(fx, expected=exp)
    result = expected_as_result(exp)
    assert by_strategy(check_no_leakage(f, fam, obs(result))) == {NONSTREAM: Status.PASS}
    # The allowance is per field: the same marker in reasoning still leaks.
    leaked = replace(result, reasoning_content="use <think> tags")
    assert by_strategy(check_no_leakage(f, fam, obs(leaked))) == {NONSTREAM: Status.FAIL}


def test_no_leakage_expected_error_allows_passthrough(trunc: Fixture, fam: Family) -> None:
    rows = check_no_leakage(
        trunc,
        fam,
        obs(ParseResult(content=trunc.raw_output), one=ParseResult(reasoning_content="x</think>")),
    )
    assert by_strategy(rows) == {NONSTREAM: Status.PASS, "one": Status.FAIL}


def test_no_leakage_skipped_without_family(fx: Fixture, good: ParseResult) -> None:
    assert check_no_leakage(fx, None, obs(good)) == []


# --------------------------------------------------------------------------- arguments


def test_arguments_json(fx: Fixture, good: ParseResult) -> None:
    rows = check_arguments_json(
        fx,
        None,
        obs(
            good,
            one=with_call(good, 0, arguments_raw=""),
            token=with_call(good, 0, arguments_raw=json.dumps('{"city": "x"}')),  # double-encoded
            rand_1_8=with_call(good, 1, arguments_raw='{"query": "x"'),
            rand_2_8=with_call(good, 0, arguments_raw="[1]"),
            rand_3_8=ParseResult(content="no calls"),
        ),
    )
    got = by_strategy(rows)
    assert got == {
        NONSTREAM: Status.PASS,
        "one": Status.FAIL,
        "token": Status.FAIL,
        "rand:1:8": Status.FAIL,
        "rand:2:8": Status.FAIL,
    }  # no row for a parse without calls
    details = {r.strategy: r.detail or "" for r in rows}
    assert "not valid JSON" in details["one"]
    assert "decode to str, not an object" in details["token"]
    assert details["rand:1:8"].startswith("[1] search")
    assert "decode to list" in details["rand:2:8"]


def test_arguments_schema(fx: Fixture, good: ParseResult) -> None:
    rows = check_arguments_schema(
        fx,
        None,
        obs(
            good,
            one=with_call(good, 0, arguments_raw='{"city": "Zürich", "unit": "kelvin"}'),
            token=with_call(good, 0, arguments_raw='{"unit": "c"}'),
            rand_1_8=with_call(good, 0, name="get_wether"),
            rand_2_8=with_call(good, 0, arguments_raw="oops"),
        ),
    )
    got = by_strategy(rows)
    assert got == {
        NONSTREAM: Status.PASS,
        "one": Status.FAIL,
        "token": Status.FAIL,
        "rand:1:8": Status.FAIL,
        "rand:2:8": Status.FAIL,
    }
    details = {r.strategy: r.detail or "" for r in rows}
    assert "'kelvin' is not one of" in details["one"]
    assert "'city' is a required property" in details["token"]
    assert "not an offered tool" in details["rand:1:8"]
    assert "not validated" in details["rand:2:8"]


def test_arguments_schema_forgives_what_the_model_emitted(fx: Fixture) -> None:
    # If the expected (model-emitted) arguments violate the schema, returning
    # them faithfully is not the engine's fault.
    assert fx.expected is not None
    bad_args = {"city": "Zürich", "unit": "kelvin"}
    exp = replace(fx.expected, tool_calls=(replace(fx.expected.tool_calls[0], arguments=bad_args),))
    f = replace(fx, expected=exp)
    assert by_strategy(check_arguments_schema(f, None, obs(expected_as_result(exp)))) == {NONSTREAM: Status.PASS}


def test_arguments_schema_invalid_tool_schema_is_error(fx: Fixture, good: ParseResult) -> None:
    tools = (
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "no-such-type"}}},
        fx.tools[1],
    )
    rows = check_arguments_schema(replace(fx, tools=tools), None, obs(good))
    assert by_strategy(rows) == {NONSTREAM: Status.ERROR}
    assert "tool schema is invalid" in (rows[0].detail or "")


def test_schema_errors_without_parameters() -> None:
    assert schema_errors(None, {"anything": 1}) == []
    assert schema_errors(None, [1]) != []


def test_parallel_order(fx: Fixture, good: ParseResult) -> None:
    rows = check_parallel_order(
        fx,
        None,
        obs(
            good,
            one=replace(good, tool_calls=good.tool_calls[::-1]),
            token=replace(good, tool_calls=good.tool_calls[:1]),
        ),
    )
    assert by_strategy(rows) == {NONSTREAM: Status.PASS, "one": Status.FAIL, "token": Status.FAIL}
    assert (rows[1].detail or "").startswith("order:")
    assert "expected 2 calls" in (rows[2].detail or "")


def test_parallel_order_only_for_multi_call() -> None:
    single = Fixture.from_dict(single_call_record())
    assert check_parallel_order(single, None, obs(ParseResult())) == []


# --------------------------------------------------------------------------- aggregate


def test_run_checks_all_pass(fx: Fixture, fam: Family, good: ParseResult) -> None:
    rows = run_checks(fx, fam, obs(good, one=good, special=good, token=good))
    assert {r.check for r in rows} == SPEC_CHECKS - {"expected_error"}
    assert case_status(rows) is Status.PASS


def test_run_checks_single_parse(fx: Fixture, fam: Family, good: ParseResult) -> None:
    rows = run_checks(fx, fam, Observation(nonstream=good, streams={}))
    assert {r.strategy for r in rows} == {NONSTREAM}
    assert {r.check for r in rows} == {
        "expected_match",
        "no_leakage",
        "arguments_json",
        "arguments_schema",
        "parallel_order",
    }


def test_case_status_ignores_synthetic_rows() -> None:
    rows = [
        CheckResult("expected_match", NONSTREAM, Status.PASS),
        CheckResult("expected_match", "char:0", Status.FAIL),
        CheckResult("split_invariance", "*", Status.SOFT_PASS),
    ]
    assert case_status(rows) is Status.SOFT_PASS
    assert case_status([*rows, CheckResult("no_leakage", "token", Status.FAIL)]) is Status.FAIL
    assert case_status([]) is Status.PASS


@pytest.mark.parametrize(
    ("label", "synthetic"),
    [
        ("nonstream", False),
        ("*", False),
        ("one", False),
        ("special", False),
        ("rand:1:8", False),
        ("char:0", True),
        ("char:7", True),
        ("weird", False),
    ],
)
def test_is_synthetic(label: str, synthetic: bool) -> None:
    assert is_synthetic(label) is synthetic


def test_checks_never_raise_on_garbage(fx: Fixture, fam: Family) -> None:
    garbage = ParseResult(
        content="\x00<tool_call>",
        reasoning_content="",
        tool_calls=(ParsedToolCall("", ""), ParsedToolCall("search", "null"), ParsedToolCall("x", '{"a": NaN}')),
        exception=None,
    )
    rows = run_checks(fx, fam, obs(garbage, one=ParseResult(exception="Boom"), token=garbage))
    assert case_status(rows) is Status.FAIL


def test_checks_never_raise_on_non_string_arguments(fx: Fixture, fam: Family, good: ParseResult) -> None:
    # SGLang's FunctionResponse.arguments is Optional; a worker reply can carry null.
    for raw in (None, 5, ["a"]):
        bad = ParseResult(tool_calls=(ParsedToolCall("get_weather", raw),))  # type: ignore[arg-type]
        rows = run_checks(fx, fam, obs(bad, one=bad))
        assert case_status(rows) is Status.FAIL
        assert any(r.check == "arguments_json" and "not valid JSON" in (r.detail or "") for r in rows)
    assert canonical_arguments(None) != canonical_arguments("null")


def test_duplicate_keys_are_invalid_json() -> None:
    assert canonical_arguments('{"a":1,"a":2}') != canonical_arguments('{"a":2}')
    with pytest.raises(ValueError, match="duplicate key"):
        decode_arguments('{"a": 1, "b": {"c": 1, "c": 1}}')
    assert canonical_arguments('{"a": {"b": 1}}') == canonical_arguments('{"a":{"b":1}}')


def test_expected_error_fails_a_call_returned_before_an_exception(trunc: Fixture) -> None:
    partial = ParseResult(tool_calls=(ParsedToolCall("get_weather", '{"city": "Par'),), exception="ValueError: cut")
    rows = check_expected_error(with_accept(trunc, "exception"), None, obs(partial))
    assert by_strategy(rows) == {NONSTREAM: Status.FAIL}
    assert "then raised" in (rows[0].detail or "")


def test_engine_specific_synthetic_strategies_do_not_count(fx: Fixture, good: ParseResult) -> None:
    """A one-token-per-step engine (Ollama) that fails only when a delta holds the whole output."""
    bad = replace(good, tool_calls=())
    observation = Observation(
        nonstream=good,
        streams={"one": bad, "token": good, "rand:1:8": bad},
        synthetic=frozenset({"one", "rand:1:8"}),
    )
    rows = run_checks(fx, None, observation)
    assert case_status(rows) is Status.FAIL  # without the engine's synthetic set
    assert case_status(rows, observation.synthetic) is Status.PASS
    # split_invariance compares realistic streams only: here just "token", so no row.
    assert not [r for r in rows if r.check == "split_invariance"]
    assert is_synthetic("one", {"one"}) and not is_synthetic("token", {"one"})

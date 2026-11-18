"""Probe tests for the OpenAI Responses API (``--api responses``), against the mock in conftest.py."""

from __future__ import annotations

import json
from typing import Any

import pytest

from canitoolcall import cli
from canitoolcall.probe import (
    BUILTIN_SCENARIOS,
    EQUIVALENCE_CHECK,
    ProbeOutcome,
    ProbeReport,
    _ResponsesAccumulator,
    probe,
)

SECRET = "sk-test-DO-NOT-LEAK-0123456789"


def _sc(sid: str) -> list[Any]:
    return [s for s in BUILTIN_SCENARIOS if s.id == sid]


def outcome(report: ProbeReport, sid: str, *, stream: bool, api: str = "responses") -> ProbeOutcome:
    for o in report.outcomes:
        if o.scenario_id == sid and o.api == api and o.check != EQUIVALENCE_CHECK and o.stream == stream:
            return o
    raise KeyError((sid, stream, api))


# --------------------------------------------------------------------------- request shape


def test_responses_body_uses_responses_shapes() -> None:
    forced = _sc("forced-tool-choice")[0].request_body("m", stream=True, api="responses")
    assert "messages" not in forced and forced["stream"] is True and forced["temperature"] == 0
    assert forced["input"] == [{"role": "user", "content": "Tell me something about Lisbon."}]
    assert forced["tool_choice"] == {"type": "function", "name": "get_weather"}
    weather = forced["tools"][0]
    assert weather["type"] == "function" and weather["name"] == "get_weather" and "function" not in weather
    assert weather["parameters"]["required"] == ["city"]

    followup = _sc("tool-result-followup")[0].request_body("m", stream=False, api="responses")
    assert followup["input"][1:] == [
        {"type": "function_call", "call_id": "call_probe_1", "name": "get_weather", "arguments": '{"city": "Paris"}'},
        {"type": "function_call_output", "call_id": "call_probe_1", "output": '{"temp_c": 18, "sky": "clear"}'},
    ]
    parallel = _sc("parallel-calls")[0].request_body("m", stream=False, api="responses")
    assert parallel["parallel_tool_calls"] is True and parallel["tool_choice"] == "auto"
    # the chat body is unchanged
    assert "messages" in _sc("single-call")[0].request_body("m", stream=False)


# --------------------------------------------------------------------------- pass path


def test_conforming_responses_server_passes_everything(mock_openai: Any) -> None:
    report = probe(mock_openai.base_url, "mock-model", apis=("responses",), timeout_s=10)
    n = len(BUILTIN_SCENARIOS)
    assert len(report.outcomes) == 3 * n
    bad = [(o.scenario_id, o.stream, o.check, o.detail) for o in report.outcomes if o.status != "pass"]
    assert bad == []
    assert all(not o.warnings for o in report.outcomes)
    assert {o.api for o in report.outcomes} == {"responses"}
    assert all("input" in b and "messages" not in b for _, b in mock_openai.requests)
    assert len(mock_openai.requests) == 2 * n


def test_responses_stream_accumulation_matches_nonstream(mock_openai: Any) -> None:
    report = probe(mock_openai.base_url, "m", apis=("responses",), scenarios=_sc("unicode-args"))
    ns, st = (outcome(report, "unicode-args", stream=s).observed for s in (False, True))
    assert ns is not None and ns == st
    assert json.loads(ns.tool_calls[0].arguments_raw or "")["body"] == "Café ☕ in 東京 — naïve résumé 🚀"


def test_responses_reasoning_and_content_items_are_read(mock_openai: Any) -> None:
    report = probe(
        mock_openai.base_url, "m", apis=("responses",), scenarios=_sc("reasoning-then-call") + _sc("no-call")
    )
    for stream in (False, True):
        r = outcome(report, "reasoning-then-call", stream=stream).observed
        assert r is not None and r.reasoning_content == "9.9 = 9.90 and 9.90 > 9.11."
        c = outcome(report, "no-call", stream=stream).observed
        assert c is not None and c.content == "42" and c.tool_calls == ()


def test_both_apis_run_and_render_an_api_column(mock_openai: Any) -> None:
    report = probe(mock_openai.base_url, "m", apis=("chat", "responses"), scenarios=_sc("single-call"))
    assert [(o.api, o.stream, o.check) for o in report.outcomes] == [
        ("chat", False, "scenario"),
        ("chat", True, "scenario"),
        ("chat", True, EQUIVALENCE_CHECK),
        ("responses", False, "scenario"),
        ("responses", True, "scenario"),
        ("responses", True, EQUIVALENCE_CHECK),
    ]
    assert all(o.status == "pass" for o in report.outcomes)
    lines = report.render_text().splitlines()
    assert next(line for line in lines if line.startswith("api")).split() == [
        "api",
        "scenario",
        "non-stream",
        "stream",
        "stream=non-stream",
    ]
    assert [line.split() for line in lines if "single-call" in line] == [
        ["chat", "single-call", "pass", "pass", "pass"],
        ["responses", "single-call", "pass", "pass", "pass"],
    ]
    assert {o["api"] for o in report.to_dict()["outcomes"]} == {"chat", "responses"}


# --------------------------------------------------------------------------- fail paths


def test_marker_leak_in_responses_content_fails(mock_openai: Any) -> None:
    mock_openai.override("no-call", stream=True, content="<|im_end|>42")
    report = probe(mock_openai.base_url, "m", apis=("responses",), scenarios=_sc("no-call"))
    assert outcome(report, "no-call", stream=False).status == "pass"
    o = outcome(report, "no-call", stream=True)
    assert o.status == "fail" and "marker '<|im_end|>' leaked into content" in (o.detail or "")
    assert "no-call [responses stream] fail: marker '<|im_end|>'" in report.render_text()


def test_done_arguments_that_disagree_with_deltas_fail(mock_openai: Any) -> None:
    mock_openai.override("single-call", done_arguments="{}")
    report = probe(mock_openai.base_url, "m", apis=("responses",), scenarios=_sc("single-call"))
    assert outcome(report, "single-call", stream=False).status == "pass"
    o = outcome(report, "single-call", stream=True)
    assert o.status == "fail" and "response.function_call_arguments.done" in (o.detail or "")


def test_stream_without_response_completed_fails(mock_openai: Any) -> None:
    mock_openai.override("single-call", omit_completed=True)
    report = probe(mock_openai.base_url, "m", apis=("responses",), scenarios=_sc("single-call"))
    o = outcome(report, "single-call", stream=True)
    assert o.status == "fail" and "response.completed" in (o.detail or "")


def test_missing_call_id_fails_in_both_modes(mock_openai: Any) -> None:
    mock_openai.override("single-call", omit_id=True)
    report = probe(mock_openai.base_url, "m", apis=("responses",), scenarios=_sc("single-call"))
    for stream in (False, True):
        o = outcome(report, "single-call", stream=stream)
        assert o.status == "fail" and "call_id" in (o.detail or ""), o.detail


def test_arguments_as_object_fails(mock_openai: Any) -> None:
    mock_openai.override("single-call", stream=False, tool_calls=(("get_weather", {"city": "Paris"}),))
    report = probe(mock_openai.base_url, "m", apis=("responses",), scenarios=_sc("single-call"), stream_modes=(False,))
    o = outcome(report, "single-call", stream=False)
    assert o.status == "fail" and "not a JSON string" in (o.detail or "")


def test_ignored_stream_flag_fails(mock_openai: Any) -> None:
    mock_openai.override("single-call", ignore_stream=True)
    report = probe(mock_openai.base_url, "m", apis=("responses",), scenarios=_sc("single-call"))
    o = outcome(report, "single-call", stream=True)
    assert o.status == "fail" and "stream=true was ignored" in (o.detail or "")


def test_error_event_is_an_error(mock_openai: Any) -> None:
    mock_openai.override("single-call", error_event=True)
    report = probe(mock_openai.base_url, "m", apis=("responses",), scenarios=_sc("single-call"))
    o = outcome(report, "single-call", stream=True)
    assert o.status == "error" and "boom" in (o.detail or "")


# --------------------------------------------------------------------------- key safety + CLI


def test_responses_key_is_sent_and_redacted(mock_openai: Any) -> None:
    body = json.dumps({"error": {"message": f"Incorrect API key provided: {SECRET}"}})
    mock_openai.override("single-call", status=401, error_body=body)
    report = probe(mock_openai.base_url, "m", api_key=SECRET, apis=("responses",), scenarios=_sc("single-call"))
    assert {h.get("Authorization") for h, _ in mock_openai.requests} == {f"Bearer {SECRET}"}
    o = outcome(report, "single-call", stream=False)
    assert o.status == "error" and "HTTP 401" in (o.detail or "") and "***" in (o.detail or "")
    assert SECRET not in json.dumps(report.to_dict()) and SECRET not in report.render_text()


@pytest.mark.parametrize(
    ("api", "want"), [("chat", {"messages"}), ("responses", {"input"}), ("both", {"messages", "input"})]
)
def test_cli_api_flag_selects_the_endpoints(
    mock_openai: Any, api: str, want: set[str], capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["probe", "--base-url", mock_openai.base_url, "--model", "m", "--no-stream"]
    assert cli.main([*argv, "--api", api]) == cli.EXIT_OK
    assert {("messages" if "messages" in b else "input") for _, b in mock_openai.requests} == want
    capsys.readouterr()


def test_cli_default_is_chat_and_responses_failures_exit_1(
    mock_openai: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["probe", "--base-url", mock_openai.base_url, "--model", "m"]
    mock_openai.override("single-call", omit_completed=True)  # only the Responses stream is affected
    assert cli.main(argv) == cli.EXIT_OK
    assert all("messages" in b for _, b in mock_openai.requests)
    assert cli.main([*argv, "--api", "both"]) == cli.EXIT_FAILURES
    out = capsys.readouterr().out
    assert "single-call [responses stream] fail" in out


# --------------------------------------------------------------------------- stream accumulation units


def _call_events(*, added: bool = True, item_done: bool = True) -> list[dict[str, Any]]:
    item = {"type": "function_call", "id": "fc_0", "call_id": "call_0", "name": "get_weather", "arguments": ""}
    events: list[dict[str, Any]] = []
    if added:
        events.append({"type": "response.output_item.added", "output_index": 0, "item": item})
    events += [
        {"type": "response.function_call_arguments.delta", "output_index": 0, "item_id": "fc_0", "delta": '{"city": '},
        {"type": "response.function_call_arguments.delta", "output_index": 0, "item_id": "fc_0", "delta": '"Paris"}'},
        {
            "type": "response.function_call_arguments.done",
            "output_index": 0,
            "item_id": "fc_0",
            "arguments": '{"city": "Paris"}',
        },
    ]
    if item_done:
        events.append(
            {"type": "response.output_item.done", "output_index": 0, "item": {**item, "arguments": '{"city": "Paris"}'}}
        )
    events.append({"type": "response.completed", "response": {"status": "completed", "output": []}})
    return events


def _accumulate(events: list[dict[str, Any]]) -> Any:
    acc = _ResponsesAccumulator()
    for ev in events:
        acc.add_chunk(ev)
    return acc.finish(0.0)


def test_accumulator_conforming_events() -> None:
    r = _accumulate(_call_events())
    assert r.problems == () and r.warnings == ()
    assert [(c.name, c.arguments_raw) for c in r.result.tool_calls] == [("get_weather", '{"city": "Paris"}')]


def test_accumulator_delta_before_item_added_fails() -> None:
    r = _accumulate(_call_events(added=False))
    assert any("before its function_call item was added" in p for p in r.problems)


def test_accumulator_missing_output_item_done_is_a_warning() -> None:
    r = _accumulate(_call_events(item_done=False))
    assert r.problems == ()
    assert r.warnings == ("no response.output_item.done event for the function_call items",)

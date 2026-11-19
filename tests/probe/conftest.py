"""A local mock OpenAI-compatible server for the probe tests.

The server answers ``POST /chat/completions`` with canned assistant messages,
rendered either as one ``chat.completion`` JSON body or as an SSE stream of
``chat.completion.chunk`` deltas (content/reasoning split into small pieces,
tool calls announced by index and id, then argument fragments), the way
OpenAI-compatible servers stream. Quirk flags reproduce protocol mistakes so
the probe's diagnostics can be tested. No real endpoint or key is used.

It also answers ``POST /responses`` (the OpenAI Responses API) from the same
canned turns: one ``response`` object with ``reasoning`` / ``message`` /
``function_call`` output items, or the SSE event sequence from OpenAI's
function-calling guide (``response.output_item.added``, argument deltas,
``response.function_call_arguments.done``, ``response.output_item.done``,
``response.completed``). See docs/probe-responses-api.md for the sources.
"""

from __future__ import annotations

import itertools
import json
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from canitoolcall.probe import BUILTIN_SCENARIOS, Scenario


@dataclass(frozen=True)
class Canned:
    """One canned assistant turn plus rendering quirks."""

    content: str | None = None
    reasoning: str | None = None
    tool_calls: tuple[tuple[str, Any], ...] = ()
    """(name, arguments); arguments is the JSON string sent (or an object with ``args_as_object``)."""
    finish_reason: str | None = None
    status: int = 200
    error_body: str | None = None
    reasoning_key: str = "reasoning_content"
    fragment: int = 3
    """Characters per streamed content/reasoning/arguments delta."""
    omit_index: bool = False
    repeat_name: bool = False
    omit_id: bool = False
    ignore_stream: bool = False
    error_event: bool = False
    done_arguments: str | None = None
    """Responses stream: arguments sent in ``response.function_call_arguments.done`` instead of the real ones."""
    omit_completed: bool = False
    """Responses stream: end without the ``response.completed`` event."""


def _chunks(text: str, n: int) -> Iterator[str]:
    for i in range(0, len(text), n):
        yield text[i : i + n]


def render_json(c: Canned, model: str) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": c.content}
    if c.reasoning is not None:
        msg[c.reasoning_key] = c.reasoning
    if c.tool_calls:
        msg["tool_calls"] = []
        for i, (name, args) in enumerate(c.tool_calls):
            call: dict[str, Any] = {"type": "function", "function": {"name": name, "arguments": args}}
            if not c.omit_id:
                call["id"] = f"call_{i}"
            msg["tool_calls"].append(call)
    finish = c.finish_reason or ("tool_calls" if c.tool_calls else "stop")
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
    }


def render_sse(c: Canned, model: str) -> Iterator[str]:
    def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
        body = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

    yield ": keep-alive comment\n\n"
    yield chunk({"role": "assistant", "content": ""})
    if c.error_event:
        yield f"data: {json.dumps({'error': {'message': 'boom'}})}\n\n"
        return
    for piece in _chunks(c.reasoning or "", c.fragment):
        yield chunk({c.reasoning_key: piece})
    for piece in _chunks(c.content or "", c.fragment):
        yield chunk({"content": piece})
    for i, (name, args) in enumerate(c.tool_calls):
        head: dict[str, Any] = {"type": "function", "function": {"name": name, "arguments": ""}}
        if not c.omit_index:
            head["index"] = i
        if not c.omit_id:
            head["id"] = f"call_{i}"
        yield chunk({"tool_calls": [head]})
        for piece in _chunks(args, c.fragment):
            fn: dict[str, Any] = {"arguments": piece}
            if c.repeat_name:
                fn["name"] = name
            part: dict[str, Any] = {"function": fn}
            if not c.omit_index:
                part["index"] = i
            yield chunk({"tool_calls": [part]})
    yield chunk({}, c.finish_reason or ("tool_calls" if c.tool_calls else "stop"))
    usage = {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "choices": [], "usage": {"total_tokens": 1}}
    yield f"data: {json.dumps(usage)}\n\n"
    yield "data: [DONE]\n\n"


def _output_items(c: Canned) -> list[dict[str, Any]]:
    """The canned turn as Responses API output items."""
    items: list[dict[str, Any]] = []
    if c.reasoning is not None:
        part = {"type": "reasoning_text", "text": c.reasoning}
        items.append({"type": "reasoning", "id": "rs_0", "summary": [], "content": [part]})
    if c.content is not None:
        text = {"type": "output_text", "text": c.content, "annotations": []}
        items.append({"type": "message", "id": "msg_0", "role": "assistant", "status": "completed", "content": [text]})
    for i, (name, args) in enumerate(c.tool_calls):
        call: dict[str, Any] = {"type": "function_call", "id": f"fc_{i}", "name": name, "arguments": args}
        if not c.omit_id:
            call["call_id"] = f"call_{i}"
        items.append({**call, "status": "completed"})
    return items


def render_responses_json(c: Canned, model: str) -> dict[str, Any]:
    out = _output_items(c)
    return {
        "id": "resp_mock",
        "object": "response",
        "model": model,
        "status": "completed",
        "error": None,
        "output": out,
    }


def render_responses_sse(c: Canned, model: str) -> Iterator[str]:
    seq = itertools.count()

    def ev(type_: str, **fields: Any) -> str:
        body = {"type": type_, "sequence_number": next(seq), **fields}
        return f"event: {type_}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"

    resp: dict[str, Any] = {"id": "resp_mock", "object": "response", "model": model, "status": "in_progress"}
    yield ev("response.created", response={**resp, "output": []})
    if c.error_event:
        yield ev("error", code="mock", message="boom", param=None)
        return
    items = _output_items(c)
    for idx, item in enumerate(items):
        ids = {"item_id": item["id"], "output_index": idx}
        if item["type"] == "reasoning":
            yield ev("response.output_item.added", output_index=idx, item={**item, "content": []})
            for piece in _chunks(c.reasoning or "", c.fragment):
                yield ev("response.reasoning_text.delta", **ids, content_index=0, delta=piece)
            yield ev("response.reasoning_text.done", **ids, content_index=0, text=c.reasoning)
        elif item["type"] == "message":
            yield ev("response.output_item.added", output_index=idx, item={**item, "content": []})
            part = {"type": "output_text", "text": "", "annotations": []}
            yield ev("response.content_part.added", **ids, content_index=0, part=part)
            for piece in _chunks(c.content or "", c.fragment):
                yield ev("response.output_text.delta", **ids, content_index=0, delta=piece, logprobs=[])
            yield ev("response.output_text.done", **ids, content_index=0, text=c.content, logprobs=[])
            yield ev("response.content_part.done", **ids, content_index=0, part={**part, "text": c.content})
        else:
            yield ev(
                "response.output_item.added", output_index=idx, item={**item, "arguments": "", "status": "in_progress"}
            )
            for piece in _chunks(item["arguments"], c.fragment):
                yield ev("response.function_call_arguments.delta", **ids, delta=piece)
            done = item["arguments"] if c.done_arguments is None else c.done_arguments
            yield ev("response.function_call_arguments.done", **ids, arguments=done)
        yield ev("response.output_item.done", output_index=idx, item=item)
    if not c.omit_completed:
        yield ev("response.completed", response={**resp, "status": "completed", "output": items})


def _args(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False)


GOOD_ANSWERS: dict[str, Canned] = {
    "single-call": Canned(tool_calls=(("get_weather", _args({"city": "Paris"})),)),
    "parallel-calls": Canned(
        tool_calls=(("get_weather", _args({"city": "Paris"})), ("get_weather", _args({"city": "Tokyo"})))
    ),
    "no-call": Canned(content="42"),
    "nested-args": Canned(
        tool_calls=(
            (
                "create_event",
                _args(
                    {
                        "title": "Design review",
                        "start": "2026-10-01T15:00:00Z",
                        "attendees": [
                            {"name": "Ana", "email": "ana@example.com"},
                            {"name": "Bo", "email": "bo@example.com"},
                        ],
                        "location": {"city": "Berlin", "room": "4.12"},
                    }
                ),
            ),
        )
    ),
    "unicode-args": Canned(
        tool_calls=(("save_note", _args({"title": "Grüße aus Zürich", "body": "Café ☕ in 東京 — naïve résumé 🚀"})),),
        fragment=1,
    ),
    "empty-args": Canned(tool_calls=(("get_server_time", "{}"),)),
    "forced-tool-choice": Canned(tool_calls=(("get_weather", _args({"city": "Lisbon"})),)),
    "tool-result-followup": Canned(content="It is 18 °C and clear in Paris."),
    "reasoning-then-call": Canned(
        reasoning="9.9 = 9.90 and 9.90 > 9.11.", tool_calls=(("record_answer", _args({"answer": "9.9"})),)
    ),
}
"""A conforming server's answer for every builtin scenario (hand-written mock data)."""

Responder = Callable[[str, bool, dict[str, Any]], Canned]
"""(scenario id, stream, request body) -> canned answer."""


def scenario_id_for(body: dict[str, Any], scenarios: tuple[Scenario, ...] = BUILTIN_SCENARIOS) -> str:
    api, key = ("chat", "messages") if "messages" in body else ("responses", "input")
    for sc in scenarios:
        if body[key] == sc.request_body("m", stream=False, api=api)[key]:
            return sc.id
    raise KeyError("unknown scenario")


def good_responder(sid: str, stream: bool, body: dict[str, Any]) -> Canned:
    return GOOD_ANSWERS[sid]


@dataclass
class MockOpenAI:
    base_url: str
    responder: Responder = good_responder
    requests: list[tuple[dict[str, str], dict[str, Any]]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def override(self, sid: str, *, stream: bool | None = None, **changes: Any) -> None:
        """Answer scenario ``sid`` (in one or both modes) with ``changes`` applied to the good answer."""
        previous = self.responder

        def responder(s: str, st: bool, body: dict[str, Any]) -> Canned:
            base = previous(s, st, body)
            if s == sid and (stream is None or st == stream):
                return replace(base, **changes)
            return base

        self.responder = responder


def _handler(mock: MockOpenAI) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # quiet
            pass

        def _send(self, status: int, ctype: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = json.loads(raw)
            with mock.lock:
                mock.requests.append((dict(self.headers.items()), body))
            renderers = {
                "/v1/chat/completions": (render_json, render_sse),
                "/v1/responses": (render_responses_json, render_responses_sse),
            }
            if self.path not in renderers:
                self._send(404, "application/json", b'{"error": {"message": "not found"}}')
                return
            as_json, as_sse = renderers[self.path]
            stream = bool(body.get("stream"))
            canned = mock.responder(scenario_id_for(body), stream, body)
            if canned.status != 200:
                err = canned.error_body or json.dumps({"error": {"message": "mock failure"}})
                self._send(canned.status, "application/json", err.encode())
                return
            if stream and not canned.ignore_stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Connection", "close")
                self.end_headers()
                for event in as_sse(canned, body["model"]):
                    self.wfile.write(event.encode("utf-8"))
                    self.wfile.flush()
                self.close_connection = True
                return
            payload = json.dumps(as_json(canned, body["model"]), ensure_ascii=False).encode("utf-8")
            self._send(200, "application/json", payload)

    return Handler


@pytest.fixture
def mock_openai() -> Iterator[MockOpenAI]:
    """A running mock server; ``mock.base_url`` ends in ``/v1``."""
    mock = MockOpenAI(base_url="")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(mock))
    server.daemon_threads = True
    mock.base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    try:
        yield mock
    finally:
        server.shutdown()
        server.server_close()

"""Live probe: check a running OpenAI-compatible endpoint in about a minute.

``canitoolcall probe --base-url http://localhost:8000/v1 --model M`` sends a
small set of scripted tool-use scenarios (single call, parallel calls, no-call
answer, nested and unicode arguments, a no-argument tool, forced
``tool_choice``, a follow-up after a tool result, reasoning + call), each
non-streaming and streaming, and reports pass/fail per scenario plus a
stream-vs-non-stream equivalence row per scenario.

Unlike the offline suite this exercises the whole stack (template, sampling,
parser, server), so outcomes depend on the model's behaviour. Checks are
therefore structural (a call to the right tool with schema-valid arguments,
stream == non-stream shape, no marker leakage into content, tool names or
arguments) rather than exact-text. In ``reasoning_content`` reasoning
delimiters (``<think>``, Harmony channel markup) still fail, but tool-call
markers are only a warning: the expected reasoning is unknown, and models often
draft their call (``<tool_call>{...}</tool_call>``) while thinking.

Uses only the standard library HTTP client (``urllib``) so it runs anywhere;
SSE is parsed by hand. The API key is sent only in the ``Authorization``
header, only to ``--base-url``: redirects are never followed (urllib would
resend the header to the new host), and the key is never logged, written to
reports, or echoed in error details.
"""

from __future__ import annotations

import base64
import contextlib
import contextvars
import http.client
import ipaddress
import json
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal

from canitoolcall import __version__
from canitoolcall.results import ParsedToolCall, ParseResult

ProbeStatus = Literal["pass", "fail", "error", "skip"]

SCENARIO_CHECK = "scenario"
"""``ProbeOutcome.check`` for a single request judged against its scenario."""
EQUIVALENCE_CHECK = "stream_equals_nonstream"
"""``ProbeOutcome.check`` for the structural stream-vs-non-stream comparison."""

# Every marker below is quoted from docs/formats/<family>.md, which were rendered
# from the official templates/encoders. A server that returns any of them in
# content, a tool name or an argument value is leaking the raw model format
# through its parser. In reasoning_content the two groups are judged apart:
#
# * TOOL_CALL_MARKERS spell a tool call. Models often draft their call as text
#   while thinking, and the probe cannot know the expected reasoning, so in
#   reasoning_content they are only a warning (REASONING_MARKER_WARNING).
# * REASONING_DELIMITERS open or close the reasoning itself, or are
#   channel/turn control markup. A correct reasoning parser always strips its
#   own delimiters (e.g. sglang#35083: glm45 left '\n<think>' in the
#   reasoning), so they fail in reasoning_content too.

TOOL_CALL_MARKERS: tuple[str, ...] = (
    # qwen3-hermes / qwen3-xml / glm
    "<tool_call>",
    "</tool_call>",
    "<function=",
    "<parameter=",
    "<arg_key>",
    "<arg_value>",
    # gpt-oss (Harmony): only used in tool-call messages
    "<|constrain|>",
    "<|call|>",
    # deepseek
    "<｜tool▁calls▁begin｜>",  # noqa: RUF001
    "<｜tool▁call▁begin｜>",  # noqa: RUF001
    "<｜tool▁sep｜>",  # noqa: RUF001
    "<｜DSML｜",  # noqa: RUF001
    # kimi K2
    "<|tool_calls_section_begin|>",
    "<|tool_call_begin|>",
    "<|tool_call_argument_begin|>",
    # mistral
    "[TOOL_CALLS]",
    "[ARGS]",
    # llama
    "<|python_tag|>",
    # gemma 4
    "<|tool_call>",
    "<tool_call|>",
    '<|"|>',
)
"""Markers that spell a tool call: a warning in ``reasoning_content``, a failure elsewhere."""

REASONING_DELIMITERS: tuple[str, ...] = (
    # reasoning tags (qwen3, deepseek, glm, kimi K2, older magistral; mistral v13+)
    "<think>",
    "</think>",
    "[THINK]",
    "[/THINK]",
    # gpt-oss (Harmony) channel and message markup
    "<|start|>",
    "<|channel|>",
    "<|message|>",
    "<|end|>",
    "<|return|>",
    # kimi K3 generic element markup (also opens and closes `think`)
    "<|open|>",
    "<|close|>",
    "<|sep|>",
    # gemma 4 thought channel
    "<|channel>",
    "<channel|>",
    # turn control tokens (qwen chatml, llama)
    "<|im_end|>",
    "<|eom_id|>",
    "<|eot_id|>",
)
"""Reasoning delimiters and channel/turn control markup: a failure everywhere,
including ``reasoning_content`` (the reasoning parser must strip them)."""

LEAK_MARKERS: tuple[str, ...] = TOOL_CALL_MARKERS + REASONING_DELIMITERS
"""Every format marker; none may appear in content, tool names or argument values."""


REASONING_MARKER_WARNING = (
    "reasoning mentions tool-call markers {markers} "
    "(usually the model drafting its call while thinking; not a parser leak)"
)
"""Warning for tool-call markers in ``reasoning_content``. The probe cannot know
the expected reasoning text, so they do not fail a scenario on their own."""


# --------------------------------------------------------------------------- model


@dataclass(frozen=True)
class ProbeExpectation:
    """Structural expectation for a scenario."""

    tool_names: tuple[str, ...] = ()
    """Tools that must be called, in order (as a subsequence); empty means no call expected."""
    min_calls: int = 0
    required_argument_keys: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    """Per tool name: argument keys every call to that tool must carry."""
    expect_reasoning: bool | None = None
    """True: ``reasoning_content`` must be present; False: absent; None: not judged."""
    expect_content: bool | None = None
    """True: non-empty ``content`` required; False: must be empty; None: not judged."""
    argument_contains: tuple[str, ...] = ()
    """Substrings that must survive verbatim in some string argument value."""


@dataclass(frozen=True)
class Scenario:
    id: str
    description: str
    messages: tuple[Mapping[str, Any], ...]
    tools: tuple[Mapping[str, Any], ...]
    expect: ProbeExpectation
    tool_choice: str | Mapping[str, Any] = "auto"
    extra_body: Mapping[str, Any] = field(default_factory=dict)

    def request_body(self, model: str, *, stream: bool) -> dict[str, Any]:
        """The /chat/completions JSON body for this scenario."""
        body: dict[str, Any] = {
            "model": model,
            "messages": [dict(m) for m in self.messages],
            "tools": [dict(t) for t in self.tools],
            "tool_choice": self.tool_choice if isinstance(self.tool_choice, str) else dict(self.tool_choice),
            "temperature": 0,
            "stream": stream,
        }
        body.update(self.extra_body)
        return body


@dataclass(frozen=True)
class ProbeOutcome:
    scenario_id: str
    stream: bool
    status: ProbeStatus
    detail: str | None
    latency_s: float
    observed: ParseResult | None = None
    check: str = SCENARIO_CHECK
    """:data:`SCENARIO_CHECK` or :data:`EQUIVALENCE_CHECK`."""
    warnings: tuple[str, ...] = ()
    """Protocol oddities that clients may trip over but that do not fail the scenario."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "check": self.check,
            "stream": self.stream,
            "status": self.status,
            "detail": self.detail,
            "warnings": list(self.warnings),
            "latency_s": round(self.latency_s, 3),
            "observed": self.observed.to_dict() if self.observed is not None else None,
        }


_STATUSES: tuple[ProbeStatus, ...] = ("pass", "fail", "error", "skip")


@dataclass(frozen=True)
class ProbeReport:
    base_url: str
    model: str
    started_at: str
    outcomes: tuple[ProbeOutcome, ...]
    finished_at: str | None = None

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = dict.fromkeys(_STATUSES, 0)
        for o in self.outcomes:
            counts[o.status] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "canitoolcall_version": __version__,
            "base_url": self.base_url,
            "model": self.model,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "summary": self.summary(),
            "outcomes": [o.to_dict() for o in self.outcomes],
        }

    def render_text(self) -> str:
        """Human-readable table for the terminal."""
        cells: dict[str, dict[str, str]] = {}
        order: list[str] = []
        for o in self.outcomes:
            if o.scenario_id not in cells:
                cells[o.scenario_id] = {}
                order.append(o.scenario_id)
            col = "equiv" if o.check == EQUIVALENCE_CHECK else ("stream" if o.stream else "nonstream")
            cells[o.scenario_id][col] = o.status + ("*" if o.warnings else "")
        headers = ("scenario", "non-stream", "stream", "stream=non-stream")
        rows = [
            (sid, cells[sid].get("nonstream", "-"), cells[sid].get("stream", "-"), cells[sid].get("equiv", "-"))
            for sid in order
        ]
        widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]

        def line(values: Sequence[str]) -> str:
            return "  ".join(v.ljust(w) for v, w in zip(values, widths, strict=True)).rstrip()

        out = [f"canitoolcall probe  {self.base_url}  model={self.model}", "", line(headers)]
        out.append(line(["-" * w for w in widths]))
        out.extend(line(r) for r in rows)
        s = self.summary()
        warned = [o for o in self.outcomes if o.warnings]
        summary = "summary: " + ", ".join(f"{s[k]} {k}" for k in _STATUSES)
        if warned:
            summary += f" ({len(warned)} with warnings, marked *)"
        out += ["", summary]
        problems = [o for o in self.outcomes if o.status in ("fail", "error")]
        if problems:
            out += ["", "problems:"]
            out += [f"  {_label(o)} {o.status}: {o.detail}" for o in problems]
        if warned:
            out += ["", "warnings (*, do not affect the exit code):"]
            out += [f"  {_label(o)}: {w}" for o in warned for w in o.warnings]
        return "\n".join(out)


def _label(o: ProbeOutcome) -> str:
    mode = "stream=non-stream" if o.check == EQUIVALENCE_CHECK else ("stream" if o.stream else "non-stream")
    return f"{o.scenario_id} [{mode}]"


# --------------------------------------------------------------------------- scenarios


def _tool(name: str, description: str, properties: Mapping[str, Any], required: Sequence[str] = ()) -> dict[str, Any]:
    params: dict[str, Any] = {"type": "object", "properties": dict(properties)}
    if required:
        params["required"] = list(required)
    return {"type": "function", "function": {"name": name, "description": description, "parameters": params}}


_WEATHER = _tool(
    "get_weather",
    "Get the current weather for a city.",
    {
        "city": {"type": "string", "description": "City name, e.g. Paris"},
        "unit": {"type": "string", "enum": ["c", "f"]},
    },
    ["city"],
)
_SERVER_TIME = _tool("get_server_time", "Return the server's current time. Takes no arguments.", {})
_SAVE_NOTE = _tool(
    "save_note",
    "Save a note with a title and a body, stored verbatim.",
    {"title": {"type": "string"}, "body": {"type": "string"}},
    ["title", "body"],
)
_CREATE_EVENT = _tool(
    "create_event",
    "Create a calendar event.",
    {
        "title": {"type": "string"},
        "start": {"type": "string", "description": "ISO 8601 date-time"},
        "attendees": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
                "required": ["name", "email"],
            },
        },
        "location": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "room": {"type": "string"}},
            "required": ["city"],
        },
    },
    ["title", "attendees", "location"],
)
_RECORD_ANSWER = _tool(
    "record_answer",
    "Record the final answer to a question.",
    {"answer": {"type": "string"}},
    ["answer"],
)


def _user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


BUILTIN_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        id="single-call",
        description="One obvious call with a single string argument.",
        messages=(_user("What's the weather in Paris right now? Use the tool."),),
        tools=(_WEATHER,),
        expect=ProbeExpectation(tool_names=("get_weather",), required_argument_keys={"get_weather": ("city",)}),
    ),
    Scenario(
        id="parallel-calls",
        description="Two independent calls to the same tool in one turn.",
        messages=(
            _user(
                "Get the current weather in Paris and in Tokyo. Call get_weather once per city, "
                "both calls in this single turn."
            ),
        ),
        tools=(_WEATHER,),
        expect=ProbeExpectation(
            tool_names=("get_weather", "get_weather"), min_calls=2, required_argument_keys={"get_weather": ("city",)}
        ),
        extra_body={"parallel_tool_calls": True},
    ),
    Scenario(
        id="no-call",
        description="Tools are offered but the answer needs none: plain content, no calls.",
        messages=(_user("What is 17 + 25? Reply with just the number and do not use any tool."),),
        tools=(_WEATHER,),
        expect=ProbeExpectation(expect_content=True),
    ),
    Scenario(
        id="nested-args",
        description="Arguments with an array of objects and a nested object, validated against the schema.",
        messages=(
            _user(
                "Create an event titled 'Design review' starting 2026-10-01T15:00:00Z with attendees "
                "Ana (ana@example.com) and Bo (bo@example.com), located in Berlin, room 4.12."
            ),
        ),
        tools=(_CREATE_EVENT,),
        expect=ProbeExpectation(
            tool_names=("create_event",),
            required_argument_keys={"create_event": ("title", "attendees", "location")},
        ),
    ),
    Scenario(
        id="unicode-args",
        description="Non-ASCII, CJK and emoji must survive into the arguments byte-exact.",
        messages=(
            _user("Save a note titled 'Grüße aus Zürich' whose body is exactly: Café ☕ in 東京 — naïve résumé 🚀"),
        ),
        tools=(_SAVE_NOTE,),
        expect=ProbeExpectation(
            tool_names=("save_note",),
            required_argument_keys={"save_note": ("title", "body")},
            argument_contains=("東京", "🚀"),
        ),
    ),
    Scenario(
        id="empty-args",
        description="A tool without parameters: arguments must still be a JSON object ('{}'), not ''.",
        messages=(_user("What time is it on the server? Use the tool."),),
        tools=(_SERVER_TIME,),
        expect=ProbeExpectation(tool_names=("get_server_time",)),
    ),
    Scenario(
        id="forced-tool-choice",
        description="tool_choice names a function; the server must return a call to it.",
        messages=(_user("Tell me something about Lisbon."),),
        tools=(_WEATHER, _SAVE_NOTE),
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        expect=ProbeExpectation(tool_names=("get_weather",), required_argument_keys={"get_weather": ("city",)}),
    ),
    Scenario(
        id="tool-result-followup",
        description="After a tool result the model answers in content without calling again.",
        messages=(
            _user("What's the weather in Paris?"),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_probe_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_probe_1", "content": '{"temp_c": 18, "sky": "clear"}'},
        ),
        tools=(_WEATHER,),
        expect=ProbeExpectation(expect_content=True),
    ),
    Scenario(
        id="reasoning-then-call",
        description=(
            "Reasoning (if the model has it) must arrive in reasoning_content, never as tags in content, "
            "followed by a call."
        ),
        messages=(
            _user(
                "Think it through step by step: which number is larger, 9.11 or 9.9? "
                "Then record the larger number with the record_answer tool."
            ),
        ),
        tools=(_RECORD_ANSWER,),
        expect=ProbeExpectation(tool_names=("record_answer",), required_argument_keys={"record_answer": ("answer",)}),
    ),
)
"""The default scenario set."""


# --------------------------------------------------------------------------- HTTP


class ProbeRequestError(Exception):
    """The request failed (transport, HTTP status, or an unreadable response)."""


@dataclass(frozen=True)
class ChatResponse:
    """One /chat/completions exchange, accumulated into a :class:`ParseResult`."""

    result: ParseResult
    latency_s: float
    finish_reason: str | None = None
    problems: tuple[str, ...] = ()
    """OpenAI-protocol violations that break clients (fail the scenario)."""
    warnings: tuple[str, ...] = ()
    """Protocol oddities some clients tolerate (reported, do not fail)."""


def _endpoint(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def check_base_url(url: str) -> str:
    """Validate ``--base-url``: an ``http(s)://host[:port]/path`` URL without
    userinfo, query or fragment. Returns it unchanged; raises ``ValueError``.
    """
    try:
        p = urllib.parse.urlsplit(url)
        port = p.port  # raises ValueError for a bad port
    except ValueError as e:
        raise ValueError(f"invalid URL {url!r}: {e}") from None
    if port == 0:
        raise ValueError("base URL port must not be 0")
    if p.scheme not in ("http", "https"):
        raise ValueError(f"base URL must start with http:// or https:// (got {safe_url(url) or url!r})")
    if not p.hostname:
        raise ValueError("base URL has no host")
    if p.username is not None or p.password is not None:
        raise ValueError("base URL must not contain user:password@; pass the key with --api-key-env")
    if p.query or p.fragment:
        raise ValueError("base URL must not contain a query or fragment")
    check_url_path(url)
    return url


_CONTROL = re.compile(r"[\x00-\x20\x7f-\x9f]")
_UNRESERVED = r"A-Za-z0-9._~\-"
_SAFE_PATH = re.compile(
    rf"(?:[{_UNRESERVED}/]|%(?:2[de]|3[0-9]|4[1-9a-f]|5[0-9af]|6[1-9a-f]|7[0-9ae]))*", re.IGNORECASE
)
"""Path characters allowed in a base URL: unreserved characters, '/', and ``%XX`` escapes that decode to
unreserved characters only (so '%' itself, i.e. double encoding, is never accepted)."""


def check_url_path(url: str) -> None:
    """Reject URLs whose path could be read differently by the allowlist and by the server.

    Refused: control characters or spaces anywhere, backslashes, encoded slashes, backslashes and NULs
    (``%2f``, ``%5c``, ``%00``, any case), and ``.`` / ``..`` segments, raw or percent-encoded (``%2e``),
    even double-encoded. ``;`` is refused, and so is any character outside ``[A-Za-z0-9._~-/]`` other than
    ``%XX`` escapes of unreserved characters. Raises ``ValueError``.
    """
    if _CONTROL.search(url) or "\\" in url:
        raise ValueError("base URL must not contain control characters, spaces or backslashes")
    path = urllib.parse.urlsplit(url).path
    if re.search(r"%(?:2f|5c|00)", path, re.IGNORECASE):
        raise ValueError("base URL path must not contain encoded slashes, backslashes or NULs")
    if ";" in path:
        raise ValueError("base URL path must not contain ';' (matrix parameters)")
    if not _SAFE_PATH.fullmatch(path):
        raise ValueError("base URL path may only contain letters, digits, '._~-/' and %XX escapes of those")
    for segment in path.split("/"):
        if urllib.parse.unquote(segment) in (".", ".."):
            raise ValueError("base URL path must not contain '.' or '..' segments")


def is_loopback(url: str) -> bool:
    """True if ``url``'s host is ``localhost`` or a loopback IP address."""
    host = (urllib.parse.urlsplit(url).hostname or "").rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def insecure_key_transport(url: str) -> bool:
    """True if a key sent to ``url`` would cross the network in cleartext (http, not loopback)."""
    return urllib.parse.urlsplit(url).scheme == "http" and not is_loopback(url)


def check_reachable(base_url: str, timeout_s: float = 10.0) -> None:
    """Open (and close) a TCP connection to the endpoint's host; sends nothing.

    Raises :class:`ProbeRequestError` with one readable line if it cannot connect.
    """
    p = urllib.parse.urlsplit(base_url)
    port = p.port or (443 if p.scheme == "https" else 80)
    try:
        with socket.create_connection((p.hostname or "", port), timeout=timeout_s):
            pass
    except OSError as e:
        raise ProbeRequestError(f"cannot reach {safe_url(base_url)}: {e.strerror or e}") from None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects: urllib's default handler resends ``Authorization`` to any host."""

    def redirect_request(
        self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> urllib.request.Request | None:
        raise ProbeRequestError(
            f"HTTP {code} redirect to {safe_url(newurl)} not followed (point --base-url at the final URL)"
        )


_OPENER = urllib.request.build_opener(_NoRedirect)

ConnectFn = Callable[..., socket.socket]
"""``socket.create_connection``-compatible: ``(address, timeout, source_address) -> socket``."""


def guarded_opener(connect: ConnectFn) -> urllib.request.OpenerDirector:
    """An opener that makes every TCP connection through ``connect`` instead of ``socket.create_connection``.

    ``connect`` resolves and vets the address and connects to the vetted IP itself, so there is no gap between
    the check and the connection (no DNS-rebinding window). Redirects are refused and environment proxies are
    ignored (a proxy would connect on the caller's behalf, past the vetting). TLS still verifies the hostname.
    """

    def watched_connect(address: tuple[str, int], timeout: Any = None, source_address: Any = None) -> socket.socket:
        dog = _WATCHDOG.get()
        if dog is None:
            return connect(address, timeout, source_address)
        # The connect (and the TLS handshake after it) may take at most the time left to the deadline.
        left = dog.remaining()
        sock = connect(address, left if timeout is None else min(timeout, left), source_address)
        dog.register(sock)
        return sock

    class _Http(http.client.HTTPConnection):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._create_connection = watched_connect

        def connect(self) -> None:
            super().connect()
            if (dog := _WATCHDOG.get()) is not None and self.sock is not None:
                dog.register(self.sock)

    class _Https(http.client.HTTPSConnection):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._create_connection = watched_connect

        def connect(self) -> None:
            super().connect()
            if (dog := _WATCHDOG.get()) is not None and self.sock is not None:
                dog.register(self.sock)  # the TLS socket replaces the raw one

    class _HttpHandler(urllib.request.HTTPHandler):
        def http_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
            return self.do_open(_Http, req)

    class _HttpsHandler(urllib.request.HTTPSHandler):
        def https_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
            return self.do_open(_Https, req, context=getattr(self, "_context", None))

    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect, _HttpHandler, _HttpsHandler)


def safe_url(url: str) -> str:
    """``url`` without userinfo, query or fragment (they may carry credentials)."""
    p = urllib.parse.urlsplit(url)
    host = p.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    netloc = host + (f":{p.port}" if p.port else "")
    return urllib.parse.urlunsplit((p.scheme, netloc, p.path, "", ""))


_MIN_REDACT = 8
"""Shorter keys are not redacted: replacing a 1-3 character string mangles messages."""


def key_forms(key: str) -> list[str]:
    """The key as it could appear in an echoed header or body: raw, ``Bearer`` form, base64 and percent-encoded."""
    forms: set[str] = set()
    for text in (key, f"Bearer {key}", f"Basic {key}"):
        raw = text.encode("utf-8")
        forms.update((text, urllib.parse.quote(text, safe=""), urllib.parse.quote_plus(text)))
        for enc in (base64.b64encode, base64.urlsafe_b64encode):
            b = enc(raw).decode("ascii")
            forms.update((b, b.rstrip("=")))
    return sorted(forms, key=len, reverse=True)  # longest first, so "Bearer <key>" goes before "<key>"


def _redact(text: str, secret: str | None) -> str:
    if not secret or len(secret) < _MIN_REDACT:
        return text
    for form in key_forms(secret):
        text = text.replace(form, "***")
    return text


_MIN_PREFIX = 4


def _clip_redacted(raw: bytes, n: int, secret: str | None) -> str:
    """Redact the whole of ``raw`` (which may extend past ``n`` by the longest key form), then keep ``n`` bytes.

    Redacting before truncating means a key straddling byte ``n`` is replaced whole; any trailing run that is
    still a prefix (>= 4 chars) of a key form is dropped as well.
    """
    text = _redact(raw.decode("utf-8", "replace"), secret)
    text = text.encode("utf-8")[:n].decode("utf-8", "ignore")
    if secret and len(secret) >= _MIN_REDACT:
        forms = key_forms(secret)
        cut = True
        while cut:
            cut = False
            for form in forms:
                for k in range(min(len(form) - 1, len(text)), _MIN_PREFIX - 1, -1):
                    if text.endswith(form[:k]):
                        text, cut = text[:-k], True
                        break
    return text


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


class _Accumulator:
    """OpenAI-client-style stream accumulation (content/reasoning concatenated,
    tool-call deltas merged by ``index``: name set once, arguments concatenated)."""

    def __init__(self) -> None:
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.calls: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None
        self.problems: list[str] = []
        self.warnings: list[str] = []

    def _note(self, bucket: list[str], msg: str) -> None:
        if msg not in bucket:
            bucket.append(msg)

    def add_chunk(self, chunk: Mapping[str, Any]) -> None:
        if "error" in chunk:
            raise ProbeRequestError(f"error event in stream: {json.dumps(chunk['error'], ensure_ascii=False)[:300]}")
        for choice in chunk.get("choices") or ():
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta") or {}
            if (c := _text(delta.get("content"))) is not None:
                self.content.append(c)
            reasoning = _text(delta.get("reasoning_content"))
            if reasoning is None:
                reasoning = _text(delta.get("reasoning"))
            if reasoning is not None:
                self.reasoning.append(reasoning)
            for tc in delta.get("tool_calls") or ():
                self._add_tool_delta(tc)
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]

    def _add_tool_delta(self, tc: Mapping[str, Any]) -> None:
        idx = tc.get("index")
        if not isinstance(idx, int):
            self._note(self.problems, "tool-call delta without an integer 'index' (clients cannot merge deltas)")
            last = max(self.calls, default=None)
            if last is None or (tc.get("id") is not None and tc.get("id") != self.calls[last]["id"]):
                idx = len(self.calls)
            else:
                idx = last
        slot = self.calls.setdefault(idx, {"id": None, "name": None, "args": []})
        if tc.get("id"):
            if slot["id"] is not None and slot["id"] != tc["id"]:
                self._note(self.warnings, f"tool call {idx}: id changed mid-stream")
            slot["id"] = slot["id"] or tc["id"]
        fn = tc.get("function") or {}
        name = fn.get("name")
        if name:
            if slot["name"] is None:
                slot["name"] = name
            elif name == slot["name"]:
                self._note(
                    self.warnings,
                    f"tool call {idx}: name repeated in a later delta "
                    "(clients that concatenate name fragments see a doubled name)",
                )
            else:
                self._note(self.problems, f"tool call {idx}: conflicting names {slot['name']!r} / {name!r}")
        args = fn.get("arguments")
        if isinstance(args, str):
            slot["args"].append(args)
        elif args is not None:
            self._note(self.problems, f"tool call {idx}: arguments delta is {type(args).__name__}, not a string")
            slot["args"].append(json.dumps(args, ensure_ascii=False))

    def finish(self, latency_s: float) -> ChatResponse:
        calls: list[ParsedToolCall] = []
        for idx in sorted(self.calls):
            slot = self.calls[idx]
            if slot["name"] is None:
                self._note(self.problems, f"tool call {idx}: no function name in any delta")
            if slot["id"] is None:
                self._note(self.warnings, f"tool call {idx}: no id (needed to send the tool result back)")
            calls.append(ParsedToolCall(slot["name"] or "", "".join(slot["args"])))
        _check_finish_reason(self.finish_reason, bool(calls), self.warnings)
        return ChatResponse(
            result=ParseResult(
                content="".join(self.content) or None,
                reasoning_content="".join(self.reasoning) or None,
                tool_calls=tuple(calls),
            ),
            latency_s=latency_s,
            finish_reason=self.finish_reason,
            problems=tuple(self.problems),
            warnings=tuple(self.warnings),
        )


def _check_finish_reason(reason: str | None, has_calls: bool, warnings: list[str]) -> None:
    if has_calls and reason != "tool_calls":
        warnings.append(f"finish_reason is {reason!r} although tool calls were returned (OpenAI sends 'tool_calls')")


def iter_sse_data(lines: Iterable[bytes]) -> Iterator[str]:
    """Yield the ``data`` payload of each server-sent event (multi-line data joined by ``\\n``)."""
    data: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
            continue
        if line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        if name == "data":
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        yield "\n".join(data)


def _parse_message(payload: Mapping[str, Any], latency_s: float) -> ChatResponse:
    if "error" in payload and not payload.get("choices"):
        raise ProbeRequestError(f"error response: {json.dumps(payload['error'], ensure_ascii=False)[:300]}")
    choices = payload.get("choices") or []
    if not choices:
        raise ProbeRequestError("response has no choices")
    choice = choices[0]
    msg = choice.get("message") or {}
    problems: list[str] = []
    warnings: list[str] = []
    calls: list[ParsedToolCall] = []
    for i, tc in enumerate(msg.get("tool_calls") or ()):
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if not isinstance(args, str):
            problems.append(f"tool call {i}: arguments is {type(args).__name__}, not a JSON string")
            args = "" if args is None else json.dumps(args, ensure_ascii=False)
        if not tc.get("id"):
            warnings.append(f"tool call {i}: no id (needed to send the tool result back)")
        if not fn.get("name"):
            problems.append(f"tool call {i}: no function name")
        calls.append(ParsedToolCall(fn.get("name") or "", args))
    reasoning = _text(msg.get("reasoning_content"))
    if reasoning is None:
        reasoning = _text(msg.get("reasoning"))
    reason = choice.get("finish_reason")
    _check_finish_reason(reason, bool(calls), warnings)
    return ChatResponse(
        result=ParseResult(content=_text(msg.get("content")), reasoning_content=reasoning, tool_calls=tuple(calls)),
        latency_s=latency_s,
        finish_reason=reason,
        problems=tuple(problems),
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class ResponseLimits:
    """Optional caps on one response (the CLI sets none; the MCP server sets all)."""

    max_bytes: int | None = None
    """Total bytes read from the body (JSON body, or all SSE lines)."""
    deadline_s: float | None = None
    """Wall-clock budget for the whole request, however slowly the server drips bytes."""
    error_bytes: int = 300
    """Bytes of an HTTP error body kept in the error message."""
    ends_at: float | None = None
    """``time.monotonic()`` time after which no request may still run (an overall cap on a whole probe)."""

    def for_request(self) -> ResponseLimits:
        """These limits for a request starting now: ``deadline_s`` shortened to the time left before ``ends_at``."""
        if self.ends_at is None:
            return self
        left = self.ends_at - time.monotonic()
        if left <= 0:
            raise ProbeRequestError("probe exceeded its overall time limit")
        return replace(self, deadline_s=left if self.deadline_s is None else min(self.deadline_s, left), ends_at=None)


class _Budget:
    def __init__(self, limits: ResponseLimits | None) -> None:
        self.limit = limits.max_bytes if limits else None
        self.deadline = time.monotonic() + limits.deadline_s if limits and limits.deadline_s else None
        self.used = 0

    def spend(self, n: int) -> None:
        self.used += n
        if self.limit is not None and self.used > self.limit:
            raise ProbeRequestError(f"response larger than {self.limit} bytes")
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise ProbeRequestError("request exceeded its overall deadline")

    def lines(self, resp: Any) -> Iterator[bytes]:
        """Lines of the body; reads whatever has arrived (``read1``) so the deadline is checked on every packet."""
        buf = b""
        while chunk := resp.read1(8192):
            self.spend(len(chunk))
            buf += chunk
            *done, buf = buf.split(b"\n")
            for line in done:
                yield line + b"\n"
        if buf:
            yield buf

    def read_all(self, resp: Any) -> bytes:
        parts: list[bytes] = []
        while chunk := resp.read1(65536):
            self.spend(len(chunk))
            parts.append(chunk)
        return b"".join(parts)


class _Watchdog:
    """Shuts down the request's sockets when the wall-clock deadline passes, so a blocked connect, header
    read, body read or error-body read ends at the deadline however slowly the server drips bytes."""

    def __init__(self, seconds: float) -> None:
        self.deadline = time.monotonic() + seconds
        self.fired = False
        self._socks: list[socket.socket] = []
        self._lock = threading.Lock()
        self._timer = threading.Timer(seconds, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def remaining(self) -> float:
        if self.fired:
            raise TimeoutError("request exceeded its overall deadline")
        return max(self.deadline - time.monotonic(), 0.001)

    def register(self, sock: socket.socket) -> None:
        with self._lock:
            self._socks.append(sock)
            fired = self.fired
        if fired:
            self._kill(sock)

    @staticmethod
    def _kill(sock: socket.socket) -> None:
        with contextlib.suppress(OSError, ValueError):
            sock.shutdown(socket.SHUT_RDWR)

    def _fire(self) -> None:
        with self._lock:
            self.fired = True
            socks = list(self._socks)
        for sock in socks:
            self._kill(sock)

    def cancel(self) -> None:
        self._timer.cancel()


_WATCHDOG: contextvars.ContextVar[_Watchdog | None] = contextvars.ContextVar("canitoolcall_watchdog", default=None)


def chat_response(
    base_url: str,
    model: str,
    scenario: Scenario,
    *,
    stream: bool,
    api_key: str | None = None,
    timeout_s: float = 60.0,
    opener: urllib.request.OpenerDirector | None = None,
    limits: ResponseLimits | None = None,
) -> ChatResponse:
    """One /chat/completions request with protocol diagnostics.

    Raises :class:`ProbeRequestError` on transport/HTTP failures; its message
    never contains ``api_key``. With ``limits.deadline_s`` the deadline covers everything from connect
    to the last byte (error bodies included) for openers made by :func:`guarded_opener`.
    """
    limits = limits.for_request() if limits else None
    dog = _Watchdog(limits.deadline_s) if limits and limits.deadline_s and opener is not None else None
    token = _WATCHDOG.set(dog)
    try:
        out = _chat_response(base_url, model, scenario, stream, api_key, timeout_s, opener, limits)
    except ProbeRequestError:
        if dog is not None and dog.fired:
            raise ProbeRequestError("request exceeded its overall deadline") from None
        raise
    finally:
        _WATCHDOG.reset(token)
        if dog is not None:
            dog.cancel()
    if dog is not None and dog.fired:  # sockets were cut mid-body: the response may be partial
        raise ProbeRequestError("request exceeded its overall deadline")
    return out


def _chat_response(
    base_url: str,
    model: str,
    scenario: Scenario,
    stream: bool,
    api_key: str | None,
    timeout_s: float,
    opener: urllib.request.OpenerDirector | None,
    limits: ResponseLimits | None,
) -> ChatResponse:
    body = json.dumps(scenario.request_body(model, stream=stream), ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream" if stream else "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    t0 = time.monotonic()
    budget = _Budget(limits)
    try:
        req = urllib.request.Request(_endpoint(base_url), data=body, headers=headers, method="POST")
        with (opener or _OPENER).open(req, timeout=timeout_s) as resp:
            ctype = resp.headers.get("Content-Type", "")
            if stream and "text/event-stream" in ctype:
                acc = _Accumulator()
                for data in iter_sse_data(budget.lines(resp)):
                    if data.strip() == "[DONE]":
                        break
                    acc.add_chunk(json.loads(data))
                return acc.finish(time.monotonic() - t0)
            payload = json.loads(budget.read_all(resp).decode("utf-8"))
            out = _parse_message(payload, time.monotonic() - t0)
            if stream:
                note = f"stream=true was ignored (Content-Type {ctype or 'missing'!r}, not text/event-stream)"
                return ChatResponse(out.result, out.latency_s, out.finish_reason, (*out.problems, note), out.warnings)
            return out
    except urllib.error.HTTPError as e:
        n = limits.error_bytes if limits else 300
        extra = max((len(f.encode("utf-8")) for f in key_forms(api_key)), default=0) if api_key else 0
        raw = e.read(n + extra) if e.fp is not None else b""
        snippet = _clip_redacted(raw, n, api_key)
        reason = f" {e.reason}" if limits is None else ""
        raise ProbeRequestError(_redact(f"HTTP {e.code}{reason}: {snippet}".strip(), api_key)) from None
    except ProbeRequestError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ProbeRequestError(_redact(f"request failed: {e}", api_key)) from None
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ProbeRequestError(f"unreadable response: {e}") from None
    except http.client.HTTPException as e:  # InvalidURL, BadStatusLine, LineTooLong, IncompleteRead ...
        raise ProbeRequestError(_redact(f"invalid HTTP exchange: {type(e).__name__}", api_key)) from None
    except ValueError as e:  # e.g. "unknown url type" from Request()
        raise ProbeRequestError(_redact(f"invalid request URL: {e}", api_key)) from None


def chat(
    base_url: str, model: str, scenario: Scenario, *, stream: bool, api_key: str | None = None, timeout_s: float = 60.0
) -> tuple[ParseResult, float]:
    """One /chat/completions request; returns the accumulated parse and latency.

    Streaming responses are accumulated OpenAI-client style (content and
    reasoning deltas concatenated; tool-call deltas merged by index).
    """
    r = chat_response(base_url, model, scenario, stream=stream, api_key=api_key, timeout_s=timeout_s)
    return r.result, r.latency_s


# --------------------------------------------------------------------------- judging


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def _found_markers(texts: Iterable[str], markers: Sequence[str]) -> list[str]:
    return sorted({m for t in texts for m in markers if m in t})


def _find_leaks(field_name: str, texts: Iterable[str], markers: Sequence[str]) -> list[str]:
    return [f"marker {m!r} leaked into {field_name}" for m in _found_markers(texts, markers)]


def _is_subsequence(needle: Sequence[str], hay: Sequence[str]) -> bool:
    it = iter(hay)
    return all(any(n == h for h in it) for n in needle)


def evaluate(
    scenario: Scenario,
    result: ParseResult,
    *,
    markers: Sequence[str] = LEAK_MARKERS,
    reasoning_delimiters: Sequence[str] = REASONING_DELIMITERS,
) -> tuple[ProbeStatus, str | None]:
    """Judge one observed result against ``scenario.expect``.

    Same as :func:`evaluate_with_warnings` without the warnings.
    """
    status, detail, _ = evaluate_with_warnings(
        scenario, result, markers=markers, reasoning_delimiters=reasoning_delimiters
    )
    return status, detail


def evaluate_with_warnings(
    scenario: Scenario,
    result: ParseResult,
    *,
    markers: Sequence[str] = LEAK_MARKERS,
    reasoning_delimiters: Sequence[str] = REASONING_DELIMITERS,
) -> tuple[ProbeStatus, str | None, tuple[str, ...]]:
    """Judge one observed result against ``scenario.expect``; returns
    ``(status, detail, warnings)``.

    Always checked: every call names an offered tool, its arguments decode to
    a JSON object valid against the tool's ``parameters`` schema, no format
    marker leaks into content, tool names or argument values, and no U+FFFD
    replacement character appears. Then the scenario's expectation.

    In ``reasoning_content``, ``reasoning_delimiters`` (``<think>``, Harmony
    channel markup, ...) fail: the reasoning parser must strip them. Other
    markers (tool-call markup) do not fail on their own: the expected reasoning
    is unknown and models often draft their call while thinking. They are
    returned as a warning, and added as a hint to the detail when the expected
    calls are missing or wrong (the call may be stuck in the reasoning).
    """
    if result.exception:
        return "fail", f"exception: {result.exception}", ()
    exp = scenario.expect
    problems: list[str] = []
    warnings: list[str] = []
    offered = {t["function"]["name"]: t["function"].get("parameters") or {} for t in scenario.tools}
    names = [c.name for c in result.tool_calls]
    decoded: list[tuple[str, dict[str, Any]]] = []
    for i, call in enumerate(result.tool_calls):
        if call.name not in offered:
            problems.append(f"call {i} names {call.name!r}, which is not an offered tool")
        try:
            args = call.arguments()
        except json.JSONDecodeError:
            problems.append(f"call {i} ({call.name}) arguments are not valid JSON: {(call.arguments_raw or '')[:80]!r}")
            continue
        if not isinstance(args, dict):
            problems.append(f"call {i} ({call.name}) arguments are {type(args).__name__}, not a JSON object")
            continue
        decoded.append((call.name, args))
        if call.name in offered:
            problems += [f"call {i} ({call.name}): {m}" for m in _schema_errors(args, offered[call.name])]

    problems += _find_leaks("content", [result.content or ""], markers)
    found_in_reasoning = _found_markers([result.reasoning_content or ""], markers)
    problems += [
        f"reasoning delimiter {m!r} leaked into reasoning_content (the reasoning parser did not strip it)"
        for m in found_in_reasoning
        if m in reasoning_delimiters
    ]
    in_reasoning = [m for m in found_in_reasoning if m not in reasoning_delimiters]
    if in_reasoning:
        warnings.append(REASONING_MARKER_WARNING.format(markers=", ".join(map(repr, in_reasoning))))
    problems += _find_leaks("a tool name", names, markers)
    problems += _find_leaks("tool arguments", [s for _, a in decoded for s in _strings(a)], markers)
    all_text = [
        result.content or "",
        result.reasoning_content or "",
        *(c.arguments_raw or "" for c in result.tool_calls),
    ]
    if any("�" in t for t in all_text):
        problems.append("U+FFFD replacement character in output (broken UTF-8 decoding)")

    calls_wrong = False
    if exp.tool_names:
        need = max(exp.min_calls, len(exp.tool_names))
        if len(names) < need:
            problems.append(f"expected at least {need} tool call(s) {list(exp.tool_names)}, got {names}")
            calls_wrong = True
        elif not _is_subsequence(exp.tool_names, names):
            problems.append(f"expected calls {list(exp.tool_names)} in order, got {names}")
            calls_wrong = True
    elif names:
        problems.append(f"expected no tool call, got {names}")
    for tool, keys in exp.required_argument_keys.items():
        for name, args in decoded:
            if name == tool and (missing := [k for k in keys if k not in args]):
                problems.append(f"{tool} call is missing argument(s) {missing}")
    if exp.argument_contains:
        values = [s for _, a in decoded for s in _strings(a)]
        lost = [s for s in exp.argument_contains if not any(s in v for v in values)]
        if lost and decoded:
            problems.append(f"argument values lost {lost}; got {[a for _, a in decoded]}")
    if exp.expect_content is True and not (result.content or "").strip():
        problems.append("expected a content answer, got none")
    if exp.expect_content is False and (result.content or "").strip():
        problems.append(f"expected no content, got {(result.content or '')[:80]!r}")
    if exp.expect_reasoning is True and not result.reasoning_content:
        problems.append("expected reasoning_content, got none")
    if exp.expect_reasoning is False and result.reasoning_content:
        problems.append("expected no reasoning_content")

    if problems:
        if calls_wrong and in_reasoning:
            problems.append(
                f"hint: reasoning_content contains {', '.join(map(repr, in_reasoning))}, "
                "so the call may have been written inside the reasoning"
            )
        return "fail", "; ".join(problems), tuple(warnings)
    return "pass", _describe(result), tuple(warnings)


def _describe(result: ParseResult) -> str:
    parts = []
    if result.tool_calls:
        calls = []
        for c in result.tool_calls:
            try:
                args = c.arguments()
                keys = ",".join(args) if isinstance(args, dict) else "?"
            except json.JSONDecodeError:
                keys = "?"
            calls.append(f"{c.name}({keys})")
        parts.append(" ".join(calls))
    if result.content and result.content.strip():
        parts.append(f"content {len(result.content)} chars")
    parts.append("reasoning_content: " + ("yes" if result.reasoning_content else "no"))
    return "; ".join(parts)


def _schema_errors(args: Mapping[str, Any], schema: Mapping[str, Any]) -> list[str]:
    import jsonschema  # lazy: main env only

    validator = jsonschema.Draft202012Validator(schema)
    return [f"{e.json_path}: {e.message}" for e in validator.iter_errors(args)][:3]


def _shape(result: ParseResult) -> dict[str, Any]:
    calls = []
    for c in result.tool_calls:
        try:
            args = c.arguments()
            keys: Any = sorted(args) if isinstance(args, dict) else "<non-object>"
        except json.JSONDecodeError:
            keys = "<invalid json>"
        calls.append((c.name, keys))
    return {
        "tool_calls": calls,
        "content": bool((result.content or "").strip()),
        "reasoning_content": bool(result.reasoning_content),
    }


def compare_modes(nonstream: ParseResult, stream: ParseResult) -> tuple[ProbeStatus, str | None]:
    """Structural stream-vs-non-stream comparison: tool names and argument keys
    in order, and presence of content and reasoning. Values are not compared
    (two sampled generations may word things differently)."""
    a, b = _shape(nonstream), _shape(stream)
    diffs = [f"{k}: non-stream {a[k]!r} vs stream {b[k]!r}" for k in a if a[k] != b[k]]
    if diffs:
        return "fail", "; ".join(diffs) + " (both sampled at temperature 0; rerun to rule out nondeterminism)"
    return "pass", None


# --------------------------------------------------------------------------- driver


def _run_one(
    base_url: str,
    model: str,
    scenario: Scenario,
    stream: bool,
    api_key: str | None,
    timeout_s: float,
    opener: urllib.request.OpenerDirector | None = None,
    limits: ResponseLimits | None = None,
) -> ProbeOutcome:
    try:
        resp = chat_response(
            base_url, model, scenario, stream=stream, api_key=api_key, timeout_s=timeout_s, opener=opener, limits=limits
        )
    except ProbeRequestError as e:
        return ProbeOutcome(scenario.id, stream, "error", _redact(str(e), api_key), 0.0)
    status, detail, judged = evaluate_with_warnings(scenario, resp.result)
    if resp.problems:
        protocol = "; ".join(resp.problems)
        detail = f"{protocol}; {detail}" if status == "fail" else protocol
        status = "fail"
    warnings = (*resp.warnings, *judged)
    return ProbeOutcome(scenario.id, stream, status, detail, resp.latency_s, resp.result, warnings=warnings)


def probe(
    base_url: str,
    model: str,
    *,
    api_key: str | None = None,
    scenarios: Sequence[Scenario] | None = None,
    stream_modes: Sequence[bool] = (False, True),
    timeout_s: float = 60.0,
    concurrency: int = 4,
    opener: urllib.request.OpenerDirector | None = None,
    limits: ResponseLimits | None = None,
) -> ProbeReport:
    """Run every scenario in every stream mode against the endpoint.

    Requests run on ``concurrency`` threads; the report order is fixed
    (scenario order, then non-stream, stream, and the equivalence row when
    both modes ran).
    """
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    todo = [(sc, mode) for sc in (BUILTIN_SCENARIOS if scenarios is None else scenarios) for mode in stream_modes]
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        done = list(
            pool.map(lambda job: _run_one(base_url, model, job[0], job[1], api_key, timeout_s, opener, limits), todo)
        )

    outcomes: list[ProbeOutcome] = []
    for sc in BUILTIN_SCENARIOS if scenarios is None else scenarios:
        mine = {o.stream: o for (s, _), o in zip(todo, done, strict=True) if s is sc}
        outcomes += [mine[m] for m in stream_modes]
        if False in mine and True in mine:
            ns, st = mine[False], mine[True]
            if ns.observed is None or st.observed is None:
                outcomes.append(ProbeOutcome(sc.id, True, "skip", "a request errored", 0.0, check=EQUIVALENCE_CHECK))
            else:
                status, detail = compare_modes(ns.observed, st.observed)
                outcomes.append(ProbeOutcome(sc.id, True, status, detail, 0.0, check=EQUIVALENCE_CHECK))
    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return ProbeReport(safe_url(base_url), model, started, tuple(outcomes), finished_at=finished)

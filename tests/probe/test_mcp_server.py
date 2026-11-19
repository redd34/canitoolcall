"""In-process MCP client <-> `canitoolcall mcp` server tests.

The matrix is a tiny hand-written file in a tmp dir (mock data, not results) and the probe talks to the mock
OpenAI-compatible server from ``conftest.py``. No real network and no real key.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anyio
import pytest

pytest.importorskip("mcp")

from mcp import Client

from canitoolcall.cli import EXIT_ERROR, main
from canitoolcall.mcp_server import DENIED, MAX_ERROR_DETAIL, ProbePolicy, build_server
from canitoolcall.probe import BUILTIN_SCENARIOS

from .conftest import MockOpenAI

MATRIX = {
    "schema_version": "1.0",
    "canitoolcall_version": "0.0.0",
    "families": ["alpha", "beta"],
    "family_info": {
        "alpha": {"name": "Alpha Family", "models": ["acme/Alpha-7B"]},
        "beta": {"name": "Beta Family", "models": ["acme/Beta-1B"]},
    },
    "engines": [
        {"name": "vllm", "version": "1.0", "commit": None, "run_started_at": "t", "run_finished_at": "t"},
        {"name": "sglang", "version": "2.0", "commit": None, "run_started_at": "t", "run_finished_at": "t"},
    ],
    "cells": [
        {
            "family": "alpha",
            "engine": "vllm",
            "engine_version": "1.0",
            "status": "pass",
            "pass_rate": 1.0,
            "counts": {"pass": 3},
            "run_at": "t",
            "page": "cells/vllm/1.0/alpha.html",
        },
        {
            "family": "alpha",
            "engine": "sglang",
            "engine_version": "2.0",
            "status": "fail",
            "pass_rate": 0.5,
            "counts": {"pass": 1, "fail": 1},
            "run_at": "t",
            "page": "cells/sglang/2.0/alpha.html",
        },
    ],
}


@pytest.fixture
def matrix_path(tmp_path: Path) -> Path:
    p = tmp_path / "matrix.json"
    p.write_text(json.dumps(MATRIX), encoding="utf-8")
    return p


def call(
    tool: str, args: dict[str, Any] | None = None, *, matrix: Path, policy: ProbePolicy | None = None
) -> tuple[bool, Any, str]:
    """Run one tool call through an in-process client; returns (is_error, structured content, text)."""
    out: list[tuple[bool, Any, str]] = []

    async def go() -> None:
        async with Client(build_server(str(matrix), policy)) as client:
            r = await client.call_tool(tool, args or {})
            text = " ".join(getattr(c, "text", "") for c in r.content)
            out.append((bool(r.is_error), r.structured_content, text))

    anyio.run(go)
    return out[0]


def test_lists_the_four_tools(matrix_path: Path) -> None:
    names: list[str] = []

    async def go() -> None:
        async with Client(build_server(str(matrix_path))) as client:
            tools = (await client.list_tools()).tools
            names.extend(t.name for t in tools)
            probe_tool = next(t for t in tools if t.name == "probe")
            desc = probe_tool.description or ""
            assert "SENDS REQUESTS" in desc
            assert f"the {len(BUILTIN_SCENARIOS)} built-in" in desc and "about 10" not in desc

    anyio.run(go)
    assert sorted(names) == ["list_engines", "list_families", "lookup", "probe"]


def test_lookup_matches_the_cli_answer(matrix_path: Path) -> None:
    from canitoolcall.lookup import lookup as lookup_matrix

    err, data, _ = call("lookup", {"name": "acme/alpha-7b", "engine": "VLLM"}, matrix=matrix_path)
    assert not err
    assert data["family"] == "alpha" and data["matched_by"] == "model name"
    assert [c["engine"] for c in data["cells"]] == ["vllm"]
    assert data == lookup_matrix(MATRIX, "acme/alpha-7b", "VLLM")

    err, data, _ = call("lookup", {"name": "alpha"}, matrix=matrix_path)
    assert not err and [c["status"] for c in data["cells"]] == ["fail", "pass"]  # sglang, vllm


def test_lookup_reports_per_model_results(tmp_path: Path) -> None:
    matrix = json.loads(json.dumps(MATRIX))
    matrix["cells"][0]["models"] = {"acme/Alpha-7B": {"status": "fail", "pass_rate": 0.0, "counts": {"fail": 1}}}
    p = tmp_path / "matrix.json"
    p.write_text(json.dumps(matrix), encoding="utf-8")

    err, data, _ = call("lookup", {"name": "acme/alpha-7b"}, matrix=p)
    assert not err and data["model"] == "acme/Alpha-7B"
    assert [(c["engine"], c["scope"], c["status"]) for c in data["cells"]] == [
        ("sglang", "family", "fail"),
        ("vllm", "model", "fail"),
    ]


def test_lookup_errors_are_tool_errors(matrix_path: Path) -> None:
    err, _, text = call("lookup", {"name": "nope"}, matrix=matrix_path)
    assert err and "unknown family or model" in text
    err, _, text = call("lookup", {"name": "beta"}, matrix=matrix_path)
    assert err and "no cell" in text


def test_bad_matrix_is_a_tool_error(tmp_path: Path) -> None:
    err, _, text = call("list_engines", matrix=tmp_path / "missing.json")
    assert err and "cannot read" in text


def test_list_families_and_engines(matrix_path: Path) -> None:
    err, data, _ = call("list_families", matrix=matrix_path)
    assert not err
    assert data["families"][0] == {"slug": "alpha", "name": "Alpha Family", "models": ["acme/Alpha-7B"]}
    err, data, _ = call("list_engines", matrix=matrix_path)
    assert not err
    assert data["engines"] == [{"name": "sglang", "versions": ["2.0"]}, {"name": "vllm", "versions": ["1.0"]}]


def test_probe_against_mock_endpoint(matrix_path: Path, mock_openai: MockOpenAI) -> None:
    args = {"base_url": mock_openai.base_url, "model": "mock-model", "stream": False}
    err, data, _ = call("probe", args, matrix=matrix_path)
    assert not err
    assert data["ok"] is True and data["model"] == "mock-model"
    assert data["summary"]["fail"] == 0 and data["summary"]["pass"] > 0
    assert mock_openai.requests and all(not body.get("stream") for _, body in mock_openai.requests)


def test_probe_reports_failures(matrix_path: Path, mock_openai: MockOpenAI) -> None:
    mock_openai.override("single-call", stream=False, tool_calls=())
    args = {"base_url": mock_openai.base_url, "model": "m", "stream": False}
    err, data, _ = call("probe", args, matrix=matrix_path)
    assert not err and data["ok"] is False and data["summary"]["fail"] >= 1


SECRET = "sk-test-not-a-real-key-123"
PUBLIC_IP = "93.184.216.34"  # a public address; never contacted (connections are redirected to the local mock)


def _bind(mock: MockOpenAI, var: str = "MCP_TEST_KEY") -> ProbePolicy:
    return ProbePolicy.parse(allow_api_key_envs=[f"{var}={mock.base_url}"])


def test_probe_sends_a_bound_key_but_never_returns_it(
    matrix_path: Path, mock_openai: MockOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MCP_TEST_KEY", SECRET)
    args = {"base_url": mock_openai.base_url, "model": "m", "api_key_env": "MCP_TEST_KEY", "stream": False}
    err, data, text = call("probe", args, matrix=matrix_path, policy=_bind(mock_openai))
    assert not err
    assert any(h.get("Authorization") == f"Bearer {SECRET}" for h, _ in mock_openai.requests)
    assert SECRET not in json.dumps(data) and SECRET not in text


def test_probe_sends_no_key_by_default(
    matrix_path: Path, mock_openai: MockOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    # CANITOOLCALL_API_KEY is no longer implicitly granted: naming it without an operator binding is refused.
    monkeypatch.setenv("CANITOOLCALL_API_KEY", SECRET)
    args = {"base_url": mock_openai.base_url, "model": "m", "api_key_env": "CANITOOLCALL_API_KEY", "stream": False}
    err, _, text = call("probe", args, matrix=matrix_path)
    assert err and DENIED in text and SECRET not in text
    assert not mock_openai.requests
    # With no api_key_env nothing is attached even though the variable is set.
    err, _, _ = call("probe", {"base_url": mock_openai.base_url, "model": "m", "stream": False}, matrix=matrix_path)
    assert not err and mock_openai.requests and all("Authorization" not in h for h, _ in mock_openai.requests)


def test_probe_refuses_unbound_env_vars(
    matrix_path: Path, mock_openai: MockOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOME_OTHER_SECRET", "hunter2")
    args = {"base_url": mock_openai.base_url, "model": "m", "api_key_env": "SOME_OTHER_SECRET"}
    err, _, text = call("probe", args, matrix=matrix_path, policy=_bind(mock_openai))
    assert err and DENIED in text and "hunter2" not in text
    assert not mock_openai.requests


def test_probe_refuses_a_key_bound_to_another_url(
    matrix_path: Path, mock_openai: MockOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MCP_TEST_KEY", SECRET)
    policy = ProbePolicy.parse(allow_api_key_envs=["MCP_TEST_KEY=https://api.example.com/v1"])
    args = {"base_url": mock_openai.base_url, "model": "m", "api_key_env": "MCP_TEST_KEY"}
    err, _, text = call("probe", args, matrix=matrix_path, policy=policy)
    assert err and DENIED in text and SECRET not in text
    assert not mock_openai.requests
    # The exact bound URL only: another path or port is not it (a trailing slash and host case are).
    assert not policy.key_for("MCP_TEST_KEY", "https://api.example.com/v2")
    assert not policy.key_for("MCP_TEST_KEY", "https://api.example.com/v1/sub")
    assert not policy.key_for("MCP_TEST_KEY", "https://api.example.com:8443/v1")
    assert policy.key_for("MCP_TEST_KEY", "https://API.example.com/v1/")


@pytest.mark.parametrize(
    "base_url",
    ["http://example.invalid/v1", "https://10.0.0.5/v1", "http://169.254.169.254/latest", "http://[fd00:ec2::254]/v1"],
)
def test_probe_refuses_non_loopback_targets_without_resolving(
    matrix_path: Path, monkeypatch: pytest.MonkeyPatch, base_url: str
) -> None:
    import socket

    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("no DNS lookup or connection may happen for a refused target")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    err, _, text = call("probe", {"base_url": base_url, "model": "m"}, matrix=matrix_path)
    assert err and DENIED in text


def test_probe_refuses_key_over_plain_http_to_remote_host(matrix_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_TEST_KEY", SECRET)
    policy = ProbePolicy(key_bindings={"MCP_TEST_KEY": "http://models.example:8000/v1"})  # parse() refuses this
    args = {"base_url": "http://models.example:8000/v1", "model": "m", "api_key_env": "MCP_TEST_KEY"}
    err, _, text = call("probe", args, matrix=matrix_path, policy=policy)
    assert err and "plain http" in text and SECRET not in text


def test_policy_parse_rejects_a_key_bound_to_plain_http_remote_host() -> None:
    with pytest.raises(ValueError, match="plain http"):
        ProbePolicy.parse(allow_api_key_envs=["MCP_TEST_KEY=http://models.example:8000/v1"])
    ProbePolicy.parse(allow_api_key_envs=["A=http://127.0.0.1:8000/v1", "B=https://models.example/v1"])
    ProbePolicy.parse(allow_base_urls=["http://models.example:8000/v1"])  # no key, so plain http is fine


@pytest.fixture
def fake_dns(monkeypatch: pytest.MonkeyPatch, mock_openai: MockOpenAI) -> Callable[[str], list[str]]:
    """Make every name resolve to a chosen address; connections to it go to the local mock server instead."""
    import socket

    connected: list[str] = []
    real = socket.create_connection
    port = int(mock_openai.base_url.split(":")[2].split("/")[0])

    def install(ip: str) -> list[str]:
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        real_gai = socket.getaddrinfo

        def gai(host: Any, p: Any, *a: Any, **k: Any) -> Any:
            if host == "127.0.0.1":  # the redirected connection below resolves for real
                return real_gai(host, p, *a, **k)
            return [(family, socket.SOCK_STREAM, 6, "", (ip, p))]

        monkeypatch.setattr(socket, "getaddrinfo", gai)

        def create(address: tuple[str, int], *a: Any, **k: Any) -> Any:
            connected.append(address[0])
            return real(("127.0.0.1", port), *a, **k)

        monkeypatch.setattr(socket, "create_connection", create)
        return connected

    return install


@pytest.mark.parametrize(
    "ip",
    [
        "10.0.0.5",
        "192.168.1.9",
        "172.16.0.1",
        "169.254.169.254",
        "169.254.10.10",
        "100.64.0.1",
        "224.0.0.1",
        "0.0.0.0",
        "127.0.0.1",
        "::1",
        "fd00:ec2::254",
        "fe80::1",
        "::ffff:10.0.0.5",
        "::",
    ],
)
def test_probe_refuses_hostnames_that_resolve_to_internal_addresses(
    matrix_path: Path, mock_openai: MockOpenAI, fake_dns: Callable[[str], list[str]], ip: str
) -> None:
    connected = fake_dns(ip)
    policy = ProbePolicy.parse(allow_base_urls=["http://models.example:8000/v1"])
    args = {"base_url": "http://models.example:8000/v1", "model": "m"}
    err, _, text = call("probe", args, matrix=matrix_path, policy=policy)
    assert err and DENIED in text
    assert ip not in text
    assert not connected and not mock_openai.requests


def test_probe_allows_an_allowlisted_hostname_with_a_public_address(
    matrix_path: Path, mock_openai: MockOpenAI, fake_dns: Callable[[str], list[str]]
) -> None:
    connected = fake_dns(PUBLIC_IP)
    policy = ProbePolicy.parse(allow_base_urls=["http://models.example:8000/v1"])
    for url, ok in [
        ("http://models.example:8000/v1", True),
        ("http://models.example:8000/v1/", True),
        ("http://MODELS.example:8000/v1/sub", True),
        ("http://models.example:8000/v2", False),
        ("http://models.example:8000/v10", False),
        ("http://models.example:9000/v1", False),
        ("https://models.example:8000/v1", False),
    ]:
        assert policy.target_allowed(url) is ok, url
    args = {"base_url": "http://models.example:8000/v1", "model": "m", "stream": False}
    err, data, text = call("probe", args, matrix=matrix_path, policy=policy)
    assert not err and data["ok"] is True, text
    assert set(connected) == {PUBLIC_IP}  # connected to the vetted address, not the name


def test_probe_trusts_an_allowlisted_ip_literal(
    matrix_path: Path, mock_openai: MockOpenAI, fake_dns: Callable[[str], list[str]]
) -> None:
    connected = fake_dns("10.0.0.5")
    policy = ProbePolicy.parse(allow_base_urls=["http://10.0.0.5:8000/v1"])
    args = {"base_url": "http://10.0.0.5:8000/v1", "model": "m", "stream": False}
    err, _, _ = call("probe", args, matrix=matrix_path, policy=policy)
    assert not err and connected
    # a different private address is still refused
    other = {"base_url": "http://10.0.0.6:8000/v1", "model": "m"}
    err, _, text = call("probe", other, matrix=matrix_path, policy=policy)
    assert err and DENIED in text


def test_probe_loopback_hostname_must_resolve_to_loopback(
    matrix_path: Path, mock_openai: MockOpenAI, fake_dns: Callable[[str], list[str]]
) -> None:
    connected = fake_dns("10.0.0.5")
    err, _, text = call("probe", {"base_url": "http://localhost:8000/v1", "model": "m"}, matrix=matrix_path)
    assert err and DENIED in text and "10.0.0.5" not in text
    assert not connected


def test_probe_does_not_follow_redirects_or_leak_the_key(
    matrix_path: Path, mock_openai: MockOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Redirect(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(307)
            self.send_header("Location", mock_openai.base_url + "/chat/completions")
            self.send_header("Content-Length", "0")
            self.end_headers()

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
        monkeypatch.setenv("MCP_TEST_KEY", SECRET)
        policy = ProbePolicy.parse(allow_api_key_envs=[f"MCP_TEST_KEY={url}"])
        args = {"base_url": url, "model": "m", "api_key_env": "MCP_TEST_KEY", "stream": False}
        err, data, text = call("probe", args, matrix=matrix_path, policy=policy)
    finally:
        srv.shutdown()
        srv.server_close()
    assert not err and data["ok"] is False  # every scenario errored: redirects are refused
    assert not mock_openai.requests  # the other origin never saw the request or the key
    assert SECRET not in json.dumps(data) and SECRET not in text


def test_policy_parse_rejects_bad_flags() -> None:
    for bad in (["NOEQUALS"], ["=http://127.0.0.1/v1"], ["V="], ["V=ftp://x/v1"], ["V=http://u:p@h/v1"]):
        with pytest.raises(ValueError):
            ProbePolicy.parse(allow_api_key_envs=bad)
    with pytest.raises(ValueError):
        ProbePolicy.parse(allow_api_key_envs=["V=http://a/v1", "V=http://b/v1"])
    with pytest.raises(ValueError):
        ProbePolicy.parse(allow_base_urls=["not a url"])


def test_cli_rejects_a_malformed_policy_flag(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["mcp", "--allow-api-key-env", "JUSTAVAR"]) == EXIT_ERROR
    assert "VAR=BASE_URL" in capsys.readouterr().err


@pytest.mark.parametrize(
    "args",
    [
        {"base_url": "ftp://x/v1", "model": "m"},
        {"base_url": "http://user:pw@127.0.0.1:1/v1", "model": "m"},
        {"base_url": "http://127.0.0.1:1/v1", "model": " "},
        {"base_url": "http://127.0.0.1:1/v1", "model": "m", "timeout_s": 10_000},
        {"base_url": "http://127.0.0.1:1/v1", "model": "m", "concurrency": 100},
    ],
)
def test_probe_validates_arguments(matrix_path: Path, args: dict[str, Any]) -> None:
    err, _, text = call("probe", args, matrix=matrix_path)
    assert err and "pw" not in text


def test_cli_reports_missing_extra(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def fake(name: str, *a: Any) -> Any:
        return None if name == "mcp" else real_find_spec(name, *a)

    monkeypatch.setattr(importlib.util, "find_spec", fake)
    assert main(["mcp"]) == EXIT_ERROR
    assert "canitoolcall[mcp]" in capsys.readouterr().err


def test_cli_does_not_swallow_other_import_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import canitoolcall.mcp_server as ms

    def broken(*a: Any, **k: Any) -> None:
        raise ImportError("some other broken import")

    monkeypatch.setattr(ms, "serve", broken)
    with pytest.raises(ImportError, match="some other broken import"):
        main(["mcp"])


# --------------------------------------------------------------------------- hardening


class RawServer:
    """A throwaway 127.0.0.1 server whose POST handler is a plain function ``(handler, request headers) -> None``."""

    def __init__(self, respond: Callable[[Any, dict[str, str]], None]) -> None:
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        outer = self
        self.requests: list[dict[str, str]] = []

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                headers = dict(self.headers.items())
                outer.requests.append(headers)
                try:
                    respond(self, headers)
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.base_url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"
        threading.Thread(target=self.srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    def close(self) -> None:
        self.srv.shutdown()
        self.srv.server_close()


def send_json(h: Any, status: int, payload: Any, ctype: str = "application/json") -> None:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    h.send_response(status)
    h.send_header("Content-Type", ctype)
    h.send_header("Content-Length", str(len(body)))
    h.end_headers()
    h.wfile.write(body)


def completion(content: str) -> dict[str, Any]:
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}


@pytest.fixture
def raw_server() -> Any:
    made: list[RawServer] = []

    def make(respond: Callable[[Any, dict[str, str]], None]) -> RawServer:
        made.append(RawServer(respond))
        return made[-1]

    yield make
    for m in made:
        m.close()


def test_key_echoed_in_response_content_is_redacted_everywhere(
    matrix_path: Path, raw_server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import base64
    from urllib.parse import quote

    key = "sk-echo key+/=ABC123"
    forms = [
        key,
        f"Bearer {key}",
        base64.b64encode(key.encode()).decode(),
        base64.b64encode(f"Bearer {key}".encode()).decode(),
        quote(key, safe=""),
        quote(f"Bearer {key}", safe=""),
    ]

    def respond(h: Any, headers: dict[str, str]) -> None:
        assert headers["Authorization"] == f"Bearer {key}"
        send_json(h, 200, completion(" | ".join(forms) + " " + headers["Authorization"]))

    srv = raw_server(respond)
    monkeypatch.setenv("MCP_TEST_KEY", key)
    policy = ProbePolicy.parse(allow_api_key_envs=[f"MCP_TEST_KEY={srv.base_url}"])
    args = {"base_url": srv.base_url, "model": "m", "api_key_env": "MCP_TEST_KEY", "stream": False}
    err, data, text = call("probe", args, matrix=matrix_path, policy=policy)
    assert not err and data["outcomes"]
    blob = json.dumps(data) + text
    for f in forms:
        assert f not in blob
        assert json.dumps(f)[1:-1] not in blob
    assert "***" in blob  # the echo really reached the report and was redacted


def test_short_keys_are_never_sent(matrix_path: Path, mock_openai: MockOpenAI, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_TEST_KEY", "short7!")
    args = {"base_url": mock_openai.base_url, "model": "m", "api_key_env": "MCP_TEST_KEY", "stream": False}
    err, _, text = call("probe", args, matrix=matrix_path, policy=_bind(mock_openai))
    assert err and DENIED in text and "short7!" not in text
    assert not mock_openai.requests


@pytest.mark.parametrize(
    "path",
    [
        "/v1/../admin",
        "/v1/./x",
        "/v1/%2e%2e/admin",
        "/v1/%2E%2E/admin",
        "/v1/.%2e/admin",
        "/v1/%252e%252e/admin",
        "/v1%2fadmin",
        "/v1%2Fadmin",
        "/v1%5cadmin",
        "/v1%5Cadmin",
        "/v1\\admin",
        "/v1\r\nX-Evil: 1",
        "/v1\nadmin",
        "/v1/\x00x",
        "/v1/%00x",
        "/v1 /x",
    ],
)
def test_base_urls_with_ambiguous_paths_are_rejected(matrix_path: Path, mock_openai: MockOpenAI, path: str) -> None:
    from canitoolcall.mcp_server import _covers
    from canitoolcall.probe import check_base_url

    with pytest.raises(ValueError):
        check_base_url("https://api.example.com" + path)
    with pytest.raises(ValueError):
        _covers("https://api.example.com/v1", "https://api.example.com" + path)
    with pytest.raises(ValueError):
        ProbePolicy.parse(allow_base_urls=["https://api.example.com" + path])
    # And through the tool, against an operator-allowed prefix and against loopback.
    policy = ProbePolicy.parse(allow_base_urls=["https://api.example.com/v1"])
    for base in ("https://api.example.com" + path, mock_openai.base_url.removesuffix("/v1") + path):
        err, _, _ = call("probe", {"base_url": base, "model": "m"}, matrix=matrix_path, policy=policy)
        assert err
    assert not mock_openai.requests


def test_port_zero_is_rejected(matrix_path: Path) -> None:
    err, _, text = call("probe", {"base_url": "http://127.0.0.1:0/v1", "model": "m"}, matrix=matrix_path)
    assert err and "port" in text
    with pytest.raises(ValueError):
        ProbePolicy.parse(allow_base_urls=["https://api.example.com:0/v1"])


@pytest.mark.parametrize(
    "ip",
    [
        "::10.0.0.5",  # IPv4-compatible
        "::a9fe:a9fe",  # IPv4-compatible 169.254.169.254
        "::ffff:169.254.169.254",
        "::ffff:127.0.0.1",
        "64:ff9b::a00:5",  # NAT64 -> 10.0.0.5
        "64:ff9b::a9fe:a9fe",  # NAT64 -> metadata
        "64:ff9b:1::a00:5",  # local-use NAT64
        "2002:a00:5::1",  # 6to4 -> 10.0.0.5
        "2002:a9fe:a9fe::1",  # 6to4 -> metadata
        "2001:0:4136:e378:8000:63bf:3fff:fdd2",  # Teredo
        "fec0::1",  # site-local
        "fd00::1",
        "ff02::1",
        "100::1",  # discard-only
        "2001:db8::1",
        "::2",
    ],
)
def test_unsafe_ipv6_forms_are_classified_unsafe(ip: str) -> None:
    import ipaddress

    from canitoolcall.mcp_server import _unsafe_address

    assert _unsafe_address(ipaddress.ip_address(ip))


def test_public_addresses_stay_safe() -> None:
    import ipaddress

    from canitoolcall.mcp_server import _unsafe_address

    for ip in ("8.8.8.8", "::ffff:8.8.8.8", "2606:4700:4700::1111", "2a00:1450:4001::200e"):
        assert not _unsafe_address(ipaddress.ip_address(ip))


@pytest.mark.parametrize("ip", ["64:ff9b::a00:5", "2002:a00:5::1", "::10.0.0.5", "fec0::1", "::ffff:169.254.169.254"])
def test_probe_refuses_hostnames_resolving_to_translation_addresses(
    matrix_path: Path, mock_openai: MockOpenAI, fake_dns: Callable[[str], list[str]], ip: str
) -> None:
    connected = fake_dns(ip)
    policy = ProbePolicy.parse(allow_base_urls=["https://api.example.com/v1"])
    args = {"base_url": "https://api.example.com/v1", "model": "m"}
    err, _, text = call("probe", args, matrix=matrix_path, policy=policy)
    assert err and ip not in text
    assert not connected and not mock_openai.requests


def test_invalid_url_from_the_worker_becomes_a_generic_error(
    matrix_path: Path, mock_openai: MockOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    import http.client

    import canitoolcall.probe as probe_mod

    def boom(*a: Any, **k: Any) -> Any:
        raise http.client.InvalidURL("URL can't contain control characters")

    monkeypatch.setattr(probe_mod, "probe", boom)
    err, _, text = call("probe", {"base_url": mock_openai.base_url, "model": "m"}, matrix=matrix_path)
    assert err and "probe failed" in text and "control characters" not in text

    # And inside chat_response the same exception is a per-scenario error, not a crash.
    class Opener:
        def open(self, *a: Any, **k: Any) -> Any:
            raise http.client.InvalidURL("bad")

    with pytest.raises(probe_mod.ProbeRequestError):
        probe_mod.chat_response(
            mock_openai.base_url,
            "m",
            BUILTIN_SCENARIOS[0],
            stream=False,
            opener=Opener(),  # type: ignore[arg-type]
        )


def test_oversized_responses_are_cut_off(matrix_path: Path, raw_server: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import canitoolcall.mcp_server as mod

    monkeypatch.setattr(mod, "MAX_RESPONSE_BYTES", 20_000)

    def respond(h: Any, headers: dict[str, str]) -> None:
        send_json(h, 200, completion("x" * 200_000))

    srv = raw_server(respond)
    err, data, _ = call("probe", {"base_url": srv.base_url, "model": "m", "stream": False}, matrix=matrix_path)
    assert not err and data["ok"] is False
    assert all(o["status"] == "error" and "larger than 20000 bytes" in o["detail"] for o in data["outcomes"])


def test_oversized_sse_streams_are_cut_off(matrix_path: Path, raw_server: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import canitoolcall.mcp_server as mod

    monkeypatch.setattr(mod, "MAX_RESPONSE_BYTES", 20_000)

    def respond(h: Any, headers: dict[str, str]) -> None:
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Connection", "close")
        h.end_headers()
        chunk = {"choices": [{"index": 0, "delta": {"content": "y" * 1000}}]}
        for _ in range(200):
            h.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        h.close_connection = True

    srv = raw_server(respond)
    err, data, _ = call("probe", {"base_url": srv.base_url, "model": "m", "stream": True}, matrix=matrix_path)
    assert not err
    streamed = [o for o in data["outcomes"] if o["stream"] and o["status"] == "error"]
    assert streamed and all("larger than" in o["detail"] for o in streamed)


def test_a_dripping_stream_hits_the_overall_deadline(matrix_path: Path, raw_server: Any) -> None:
    import time

    def respond(h: Any, headers: dict[str, str]) -> None:
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Connection", "close")
        h.end_headers()
        chunk = {"choices": [{"index": 0, "delta": {"content": "."}}]}
        for _ in range(400):  # 20 s of bytes, each gap far shorter than the socket timeout
            h.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            h.wfile.flush()
            time.sleep(0.05)
        h.close_connection = True

    srv = raw_server(respond)
    t0 = time.monotonic()
    args = {"base_url": srv.base_url, "model": "m", "stream": False, "timeout_s": 0.4, "concurrency": 8}
    err, data, _ = call("probe", args, matrix=matrix_path)
    assert not err
    assert time.monotonic() - t0 < 15
    assert any(
        "overall deadline" in (o.get("detail") or "") or "unreadable" in (o.get("detail") or "")
        for o in data["outcomes"]
    )


def test_http_error_detail_is_trimmed_and_redacted(
    matrix_path: Path, raw_server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def respond(h: Any, headers: dict[str, str]) -> None:
        body = (headers["Authorization"] + " " + "z" * 5000).encode()
        send_json(h, 500, body, "text/plain")

    srv = raw_server(respond)
    monkeypatch.setenv("MCP_TEST_KEY", SECRET)
    policy = ProbePolicy.parse(allow_api_key_envs=[f"MCP_TEST_KEY={srv.base_url}"])
    args = {"base_url": srv.base_url, "model": "m", "api_key_env": "MCP_TEST_KEY", "stream": False}
    err, data, text = call("probe", args, matrix=matrix_path, policy=policy)
    assert not err
    for o in data["outcomes"]:
        assert o["detail"].startswith("HTTP 500") and len(o["detail"]) <= 240
    assert SECRET not in json.dumps(data) + text


def test_matrix_source_secrets_are_scrubbed_from_errors() -> None:
    source = "http://alice:hunter2pw@127.0.0.1:1/matrix.json?token=sekrit-token#frag-secret"
    out: list[str] = []
    for tool, args in (("lookup", {"name": "alpha"}), ("list_families", {}), ("list_engines", {})):

        async def go(tool: str = tool, args: dict[str, Any] = args) -> None:
            async with Client(build_server(source)) as client:
                r = await client.call_tool(tool, args)
                assert r.is_error
                out.append(" ".join(getattr(c, "text", "") for c in r.content))

        anyio.run(go)
    for text in out:
        assert "cannot read" in text
        for secret in ("alice", "hunter2pw", "sekrit-token", "frag-secret"):
            assert secret not in text


@pytest.mark.parametrize(
    "path",
    [
        "/v1/..;x",
        "/v1;a=b/x",
        "/v1/%2e%2e",
        "/v1/%252e%252e",
        "/v1/%25252e%25252e",
        "/v1/%2525252e%2525252e",
        "/v1/%252525252e",
        "/v1/%2E./x",
        "/v1/.%2e",
        "/v1/%2",
        "/v1/%zz",
        "/v1/%25",
        "/v1/a%3bb",
        "/v1/a:b",
        "/v1/a%7Cb",
    ],
)
def test_path_segments_are_checked_after_full_decoding(path: str) -> None:
    from canitoolcall.probe import check_base_url

    with pytest.raises(ValueError):
        check_base_url("https://api.example.com" + path)


@pytest.mark.parametrize("path", ["", "/", "/v1", "/openai/v1/", "/a_b/c-d/e.f~g", "/v%31", "/api%2Dv1"])
def test_plain_paths_are_still_accepted(path: str) -> None:
    from canitoolcall.probe import check_base_url

    assert check_base_url("https://api.example.com" + path)


def _assert_all_error_by_deadline(data: dict[str, Any], elapsed: float, bound: float) -> None:
    assert elapsed < bound, elapsed
    assert data["outcomes"]
    assert all(o["status"] == "error" for o in data["outcomes"])
    assert all("overall deadline" in (o.get("detail") or "") for o in data["outcomes"])


def test_dripped_response_headers_hit_the_deadline(matrix_path: Path, raw_server: Any) -> None:
    import time

    def respond(h: Any, headers: dict[str, str]) -> None:
        h.wfile.write(b"HTTP/1.1 200 OK\r\n")
        for _ in range(400):  # 20 s of header lines, each gap far shorter than the socket timeout
            h.wfile.write(b"X-Pad: 1\r\n")
            h.wfile.flush()
            time.sleep(0.05)
        h.close_connection = True

    srv = raw_server(respond)
    t0 = time.monotonic()
    args = {"base_url": srv.base_url, "model": "m", "stream": False, "timeout_s": 0.4, "concurrency": 8}
    err, data, _ = call("probe", args, matrix=matrix_path)
    assert not err
    _assert_all_error_by_deadline(data, time.monotonic() - t0, 8)


def test_dripped_error_body_hits_the_deadline(matrix_path: Path, raw_server: Any) -> None:
    import time

    def respond(h: Any, headers: dict[str, str]) -> None:
        h.send_response(500)
        h.send_header("Content-Type", "text/plain")
        h.send_header("Content-Length", "100000")
        h.end_headers()
        for _ in range(400):
            h.wfile.write(b"e")
            h.wfile.flush()
            time.sleep(0.05)
        h.close_connection = True

    srv = raw_server(respond)
    t0 = time.monotonic()
    args = {"base_url": srv.base_url, "model": "m", "stream": False, "timeout_s": 0.4, "concurrency": 8}
    err, data, _ = call("probe", args, matrix=matrix_path)
    assert not err
    _assert_all_error_by_deadline(data, time.monotonic() - t0, 8)


def test_key_straddling_the_error_boundary_is_redacted(
    matrix_path: Path, raw_server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from canitoolcall.mcp_server import MAX_ERROR_DETAIL

    def respond(h: Any, headers: dict[str, str]) -> None:
        # The key starts 5 bytes before the cut of the body snippet.
        send_json(h, 500, ("x" * (MAX_ERROR_DETAIL - 5) + SECRET + "tail").encode(), "text/plain")

    srv = raw_server(respond)
    monkeypatch.setenv("MCP_TEST_KEY", SECRET)
    policy = ProbePolicy.parse(allow_api_key_envs=[f"MCP_TEST_KEY={srv.base_url}"])
    args = {"base_url": srv.base_url, "model": "m", "api_key_env": "MCP_TEST_KEY", "stream": False}
    err, data, text = call("probe", args, matrix=matrix_path, policy=policy)
    assert not err
    for o in data["outcomes"]:
        assert o["detail"].startswith("HTTP 500")
    blob = json.dumps(data) + text
    for k in range(4, len(SECRET) + 1):
        assert SECRET[:k] not in blob


def test_clip_redacted_drops_trailing_key_prefixes() -> None:
    from canitoolcall.probe import _clip_redacted

    # the key crosses the cut: redacted whole, then truncated
    assert _clip_redacted(b"x" * 195 + SECRET.encode(), 200, SECRET) == "x" * 195 + "***"
    # a raw cut inside the key leaves a prefix of >= 4 characters: dropped
    assert _clip_redacted(b"abc " + SECRET[:9].encode(), 200, SECRET) == "abc "
    assert _clip_redacted(b"abc " + SECRET[:3].encode(), 200, SECRET) == "abc " + SECRET[:3]


@pytest.mark.parametrize("inside", [4, 6, 11])
def test_key_cut_by_the_error_payload_snippet_is_redacted(
    matrix_path: Path, raw_server: Any, monkeypatch: pytest.MonkeyPatch, inside: int
) -> None:
    """Regression guard: a key straddling the error-detail cut must not survive in part.

    Today every snippet cut (the probe's 300 characters) is followed by the 216-byte trim, which removes the
    fragment either way; this keeps that true if either limit changes. ``_drop_key_tail`` is tested directly below.
    """
    key = "sk-SnippetKey0123456789"
    shown = MAX_ERROR_DETAIL + 16  # what _trim_errors keeps
    pad = shown - len('error response: {"message": "') - inside  # the cut lands `inside` characters into the key

    def respond(h: Any, headers: dict[str, str]) -> None:
        send_json(h, 200, {"error": {"message": "A" * pad + key + " tail"}})

    srv = raw_server(respond)
    monkeypatch.setenv("MCP_TEST_KEY", key)
    policy = ProbePolicy.parse(allow_api_key_envs=[f"MCP_TEST_KEY={srv.base_url}"])
    args = {"base_url": srv.base_url, "model": "m", "api_key_env": "MCP_TEST_KEY", "stream": False}
    _, data, text = call("probe", args, matrix=matrix_path, policy=policy)
    blob = json.dumps(data) + text
    assert "A" * 50 in blob  # the snippet really reached the report
    assert key[:inside] not in blob


def test_drop_key_tail_leaves_ordinary_text_alone() -> None:
    from canitoolcall.mcp_server import _drop_key_tail, _key_forms

    forms = _key_forms("sk-SnippetKey0123456789")
    assert _drop_key_tail("send it as Bearer", forms) == "send it as Bearer"
    assert _drop_key_tail("ends with sk-S", forms) == "ends with ***"
    assert _drop_key_tail("auth: Bearer sk-Sni", forms) == "auth: ***"
    assert _drop_key_tail("ends with sk-", forms) == "ends with sk-"


def test_a_probe_call_has_an_overall_time_cap(
    matrix_path: Path, raw_server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    import canitoolcall.mcp_server as ms

    def respond(h: Any, headers: dict[str, str]) -> None:
        time.sleep(3)  # each request is well inside its own 2 x timeout_s deadline
        send_json(h, 200, completion("late"))

    assert 60 <= ms.MAX_PROBE_S <= 600
    monkeypatch.setattr(ms, "MAX_PROBE_S", 1.0)
    srv = raw_server(respond)
    t0 = time.monotonic()
    args = {"base_url": srv.base_url, "model": "m", "stream": False, "timeout_s": 5, "concurrency": 1}
    err, data, _ = call("probe", args, matrix=matrix_path)
    assert time.monotonic() - t0 < 6  # 9 requests x 3 s without the cap
    assert not err and data["ok"] is False
    assert all(o["status"] == "error" for o in data["outcomes"])
    assert len(srv.requests) == 1  # nothing new is started once the cap has passed


def test_concurrent_probe_calls_are_bounded(matrix_path: Path, raw_server: Any) -> None:
    import time

    from canitoolcall.mcp_server import MAX_PARALLEL_PROBES

    def respond(h: Any, headers: dict[str, str]) -> None:
        time.sleep(1)
        send_json(h, 200, completion("ok"))

    srv = raw_server(respond)
    args = {"base_url": srv.base_url, "model": "m", "stream": False, "timeout_s": 5, "concurrency": 8}
    results: list[tuple[bool, str]] = []

    async def go() -> None:
        async with Client(build_server(str(matrix_path))) as client:

            async def one() -> None:
                r = await client.call_tool("probe", args)
                results.append((bool(r.is_error), " ".join(getattr(c, "text", "") for c in r.content)))

            async with anyio.create_task_group() as tg:
                for _ in range(MAX_PARALLEL_PROBES + 1):
                    tg.start_soon(one)

    anyio.run(go)
    refused = [text for is_err, text in results if is_err]
    assert len(results) == MAX_PARALLEL_PROBES + 1
    assert len(refused) == 1 and "already running" in refused[0]

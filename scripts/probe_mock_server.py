"""Run the probe tests' mock OpenAI-compatible server (tests/probe/conftest.py) on its own.

For the GitHub Action self-test (.github/workflows/action-selftest.yml) and for trying
`canitoolcall probe` without a model. It serves hand-written canned answers, never a model:

    uv run python scripts/probe_mock_server.py --port 8011
    uv run canitoolcall probe --base-url http://127.0.0.1:8011/v1 --model mock-model

The first line on stdout is "mock OpenAI-compatible server listening on <base URL>".
Needs the dev environment (`uv sync`), because conftest.py imports pytest.
"""

from __future__ import annotations

import argparse
import hmac
import importlib.util
import os
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import ModuleType

CONFTEST = Path(__file__).resolve().parents[1] / "tests" / "probe" / "conftest.py"


def _load_mock() -> ModuleType:
    spec = importlib.util.spec_from_file_location("probe_mock_conftest", CONFTEST)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8011, help="0 picks a free port (default 8011)")
    p.add_argument(
        "--omit-index",
        action="append",
        default=[],
        metavar="SCENARIO",
        help="stream this scenario's tool-call deltas without 'index', so the probe fails it (repeatable)",
    )
    p.add_argument(
        "--require-key-env",
        metavar="VAR",
        help="answer 401 unless the request carries 'Authorization: Bearer $VAR'",
    )
    args = p.parse_args(argv)
    if args.require_key_env and not os.environ.get(args.require_key_env):
        p.error(f"--require-key-env: ${args.require_key_env} is not set")

    mod = _load_mock()
    mock = mod.MockOpenAI(base_url="")
    for sid in args.omit_index:
        mock.override(sid, stream=True, omit_index=True)
    base = mod._handler(mock)
    expected = f"Bearer {os.environ[args.require_key_env]}" if args.require_key_env else None

    class Handler(base):  # type: ignore[misc,valid-type]
        def do_POST(self) -> None:
            if expected is not None and not hmac.compare_digest(self.headers.get("Authorization", ""), expected):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self._send(401, "application/json", b'{"error": {"message": "missing or wrong API key"}}')
                return
            super().do_POST()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    host, port = server.server_address[:2]
    print(f"mock OpenAI-compatible server listening on http://{host}:{port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

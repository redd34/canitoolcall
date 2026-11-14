"""JSON-lines worker that runs one adapter inside its engine's interpreter.

The runner (main env) starts::

    PYTHONPATH=<repo>/src <engine-python> -m canitoolcall.adapters.worker <engine>

and exchanges one JSON object per line on stdin/stdout. stderr is free-form
logging. Protocol (version 1):

``{"op": "hello"}``
    -> ``{"ok": true, "protocol": 1, "engine": str, "version": str,
    "pinned_version": str, "tokens_per_step": "one"|"many", "commit": str|null,
    "python": str, "details": {...}}``

``{"op": "replay", "fixture": <fixture record>, "family": <family.json>|null,
"strategies": ["one", "special", "token", "rand:1:8", ...]}``
    -> ``{"ok": true, "fixture_id": str, "supported": bool, "reason": str|null,
    "parser_config": {...}, "nonstream": <ParseResult>, "streams": {id: <ParseResult>},
    "skipped": {id: reason}}``
    (``nonstream``/``streams``/``parser_config``/``skipped`` omitted when unsupported)

``{"op": "shutdown"}``
    -> ``{"ok": true}`` then exit 0.

Any harness failure answers ``{"ok": false, "error": "<traceback>"}`` and the
worker keeps serving. An engine that is not installed answers
``{"ok": false, "unavailable": true, "error": "<one line>"}``. Standard library only.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import traceback
from collections.abc import Collection, Mapping, Sequence
from typing import IO, Any

from canitoolcall.adapters import load_adapter
from canitoolcall.adapters.base import Adapter, AdapterUnavailable, ReplayInput
from canitoolcall.chunking import ChunkStrategy, split, split_text
from canitoolcall.fixtures import Family, Fixture

PROTOCOL_VERSION = 1


def hello(adapter: Adapter) -> dict[str, Any]:
    return {
        "ok": True,
        "protocol": PROTOCOL_VERSION,
        "engine": adapter.name,
        "version": adapter.version(),
        "pinned_version": adapter.pinned_version,
        "tokens_per_step": adapter.tokens_per_step,
        "commit": adapter.commit(),
        "python": platform.python_version(),
        "details": adapter.engine_details(),
    }


def replay(
    adapter: Adapter,
    fixture: Fixture,
    family: Family | None,
    strategies: Sequence[ChunkStrategy],
) -> dict[str, Any]:
    """Replay one fixture: one non-streaming parse plus one stream per strategy.

    * ``one``/``token``/``rand``: groups of :meth:`Adapter.units`.
    * ``special``: needs :meth:`Adapter.special_token_ids`; skipped (with a
      reason) when the adapter cannot say which ids are special.
    * ``char``: synthetic text deltas, only through
      :meth:`Adapter.parse_stream_text`; skipped otherwise. The generic worker
      never fabricates token ids for character splits.

    Skipped strategies are listed in ``skipped`` so the gap is visible in results.
    """
    raw = ReplayInput.from_fixture(fixture, family)
    sup = adapter.supports(raw.family, raw.model)
    if sup:
        # An override can narrow support, but cannot widen an existing refusal.
        sup = adapter.supports_replay(raw)
    if not sup:
        return {"ok": True, "fixture_id": fixture.id, "supported": False, "reason": sup.reason}
    tools = list(fixture.tools)
    units = adapter.units(raw)
    special: Collection[int] | None = None
    special_known = False
    streams: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for strat in strategies:
        if strat.id in streams or strat.id in skipped:
            continue
        if strat.kind == "char":
            if not adapter.supports_text_deltas:
                skipped[strat.id] = "synthetic character split: adapter has no text-delta streaming path"
                continue
            result = adapter.parse_stream_text(raw, split_text(raw.text, strat), tools)
        elif strat.kind == "special":
            if not special_known:
                special, special_known = adapter.special_token_ids(raw), True
            if special is None:
                skipped[strat.id] = "adapter does not expose the engine's special token ids"
                continue
            result = adapter.parse_stream(raw, split(units, strat, special=frozenset(special)), tools)
        else:
            result = adapter.parse_stream(raw, split(units, strat), tools)
        streams[strat.id] = result.to_dict()
    nonstream = adapter.parse(raw, tools).to_dict()
    # parser_config comes last: some engines (llama.cpp) only know their full
    # parser configuration after a parse with this fixture's tools.
    config = adapter.parser_config(raw)
    if raw.finish is not None or raw.assumes is not None:
        config = dict(config)
        conditions: dict[str, Any] = {}
        if raw.finish is not None:
            conditions["finish"] = raw.finish.to_dict()
        if raw.assumes is not None:
            conditions["assumes"] = list(raw.assumes)
        config["replay_conditions"] = conditions
    return {
        "ok": True,
        "fixture_id": fixture.id,
        "supported": True,
        "reason": None,
        "parser_config": config,
        "nonstream": nonstream,
        "streams": streams,
        "skipped": skipped,
    }


def handle(adapter: Adapter, msg: Mapping[str, Any]) -> dict[str, Any]:
    """Dispatch one protocol message (never raises)."""
    try:
        op = msg.get("op")
        if op == "hello":
            return hello(adapter)
        if op == "replay":
            fam = msg.get("family")
            return replay(
                adapter,
                Fixture.from_dict(msg["fixture"]),
                Family.from_dict(fam) if fam else None,
                [ChunkStrategy.parse(s) for s in msg.get("strategies", [])],
            )
        if op == "shutdown":
            return {"ok": True}
        return {"ok": False, "error": f"unknown op {op!r}"}
    except AdapterUnavailable as e:
        return {"ok": False, "unavailable": True, "error": str(e)}
    except Exception:
        return {"ok": False, "error": traceback.format_exc()}


def serve(adapter: Adapter, stdin: IO[str], stdout: IO[str]) -> int:
    """Serve the protocol until ``shutdown`` or EOF. Returns the exit code."""
    try:
        for line in stdin:
            if not line.strip():
                continue
            msg: Any = None
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as e:
                reply: dict[str, Any] = {"ok": False, "error": f"bad JSON: {e}"}
            else:
                reply = (
                    handle(adapter, msg)
                    if isinstance(msg, dict)
                    else {"ok": False, "error": "message must be a JSON object"}
                )
            stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            stdout.flush()
            if isinstance(msg, dict) and msg.get("op") == "shutdown":
                break
    finally:
        adapter.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m canitoolcall.adapters.worker <engine>", file=sys.stderr)
        return 2
    # Keep the protocol channel clean: engines print on import, and compiled
    # extensions may write straight to file descriptor 1. Move the protocol to
    # a private duplicate of fd 1 and point fd 1 (and sys.stdout) at stderr.
    sys.stdout.flush()
    saved_fd1 = os.dup(1)
    proto_out = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
    os.dup2(2, 1)
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    try:
        adapter = load_adapter(args[0])
        return serve(adapter, sys.stdin, proto_out)
    finally:
        proto_out.close()
        os.dup2(saved_fd1, 1)
        os.close(saved_fd1)
        sys.stdout = old_stdout


if __name__ == "__main__":
    raise SystemExit(main())

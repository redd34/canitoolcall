"""CanIToolCall: a neutral conformance suite for tool-call and reasoning parsers.

Recorded raw model outputs (fixtures) are replayed offline through each
inference engine's *own* parser code, non-streaming and as many chunked
streams, and the parsed results are checked against the expected parse.

Import rule: ``fixtures``, ``chunking``, ``results``, ``adapters.base`` and
``adapters.worker`` must import with the standard library only, because the
adapter worker runs inside each engine's isolated virtualenv.
"""

from __future__ import annotations

__version__ = "0.3.0"

SPEC_VERSION = "0.2"
"""Fixture spec version this package reads and writes (see spec/README.md)."""

__all__ = ["SPEC_VERSION", "__version__"]

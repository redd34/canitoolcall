"""Engine adapters and their registry.

Each engine has exactly one module here (``vllm.py``, ``sglang.py``,
``llamacpp.py``, ``ollama.py``, ``transformers.py``, ``mlx_lm.py``) defining one
:class:`~canitoolcall.adapters.base.Adapter` subclass. Adapter modules must NOT
import their engine at module import time; import it inside methods so the
registry can be listed from the main dev env.

Adapters execute inside an isolated per-engine interpreter (see
:func:`engine_python`), driven over JSON lines by
:mod:`canitoolcall.adapters.worker`.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from canitoolcall.adapters.base import Adapter, AdapterUnavailable, ReplayInput, Support

ENGINES: dict[str, str] = {
    "vllm": "canitoolcall.adapters.vllm:VllmAdapter",
    "sglang": "canitoolcall.adapters.sglang:SglangAdapter",
    "llamacpp": "canitoolcall.adapters.llamacpp:LlamaCppAdapter",
    "ollama": "canitoolcall.adapters.ollama:OllamaAdapter",
    "transformers": "canitoolcall.adapters.transformers:TransformersAdapter",
    "mlx_lm": "canitoolcall.adapters.mlx_lm:MlxLmAdapter",
}
"""Engine name -> ``module:Class``. Core-owned; adapter builders never edit it."""

PYTHON_ENV = "CANITOOLCALL_{ENGINE}_PYTHON"
"""Per-engine interpreter override, e.g. ``CANITOOLCALL_VLLM_PYTHON``."""


def is_adapter_spec(engine: str) -> bool:
    """True for an explicit ``package.module:Class`` adapter spec (not a registry name)."""
    mod, sep, cls = engine.partition(":")
    return bool(sep and mod and cls.isidentifier() and all(p.isidentifier() for p in mod.split(".")))


def adapter_class(engine: str) -> type[Adapter]:
    """Import and return the adapter class for ``engine`` (does not import the engine).

    ``engine`` is a registry name (see :data:`ENGINES`) or an explicit
    ``package.module:Class`` spec, which lets out-of-tree and test adapters run
    through the same worker. The module must be importable by the interpreter
    that loads it.
    """
    if engine in ENGINES:
        target = ENGINES[engine]
    elif is_adapter_spec(engine):
        target = engine
    else:
        raise ValueError(f"unknown engine {engine!r}; known: {', '.join(ENGINES)} (or pass package.module:Class)")
    mod_name, cls_name = target.split(":")
    cls = getattr(importlib.import_module(mod_name), cls_name)
    if not (isinstance(cls, type) and issubclass(cls, Adapter)):
        raise TypeError(f"{target} is not an Adapter subclass")
    return cls


def load_adapter(engine: str) -> Adapter:
    """Instantiate the adapter for ``engine`` in the *current* interpreter."""
    return adapter_class(engine)()


@dataclass(frozen=True)
class EngineSetup:
    """Where an engine's interpreter comes from, and whether it was set up at all."""

    python: Path
    source: str
    """``env`` ($CANITOOLCALL_<ENGINE>_PYTHON), ``venv`` (.venvs/<engine>) or ``current`` (fallback)."""

    @property
    def configured(self) -> bool:
        return self.source != "current"


def engine_setup(engine: str, repo_root: Path | None = None) -> EngineSetup:
    """Resolve the interpreter for ``engine`` like :func:`engine_python`, and say how."""
    if repo_root is None:
        from canitoolcall.fixtures import repo_root as _root

        repo_root = _root()
    if is_adapter_spec(engine):
        return EngineSetup(Path(sys.executable), "current")
    env = os.environ.get(PYTHON_ENV.format(ENGINE=engine.upper()))
    if env:
        return EngineSetup(Path(env), "env")
    if repo_root is not None:
        venv_py = repo_root / ".venvs" / engine / "bin" / "python"
        if venv_py.exists():
            return EngineSetup(venv_py, "venv")
    return EngineSetup(Path(sys.executable), "current")


def engine_python(engine: str, repo_root: Path | None = None) -> Path:
    """Interpreter that has ``engine`` installed.

    Resolution order: ``$CANITOOLCALL_<ENGINE>_PYTHON``, then
    ``<repo>/.venvs/<engine>/bin/python`` (built by ``scripts/engines/<engine>.sh``),
    then the current interpreter.
    """
    if is_adapter_spec(engine):
        return Path(sys.executable)
    env = os.environ.get(PYTHON_ENV.format(ENGINE=engine.upper()))
    if env:
        return Path(env)
    if repo_root is not None:
        venv_py = repo_root / ".venvs" / engine / "bin" / "python"
        if venv_py.exists():
            return venv_py
    return Path(sys.executable)


__all__ = [
    "ENGINES",
    "PYTHON_ENV",
    "Adapter",
    "AdapterUnavailable",
    "EngineSetup",
    "ReplayInput",
    "Support",
    "adapter_class",
    "engine_python",
    "engine_setup",
    "is_adapter_spec",
    "load_adapter",
]

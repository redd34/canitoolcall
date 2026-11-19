#!/usr/bin/env bash
# Build the isolated mlx-lm replay env: .venvs/mlx_lm
#
#   mlx-lm==0.32.2 (PyPI; git tag v0.32.2 = ml-explore/mlx-lm@93e6d4a6), Python 3.12.
#   macOS arm64: mlx==0.32.4 with its Metal backend (mlx-metal), as `pip install mlx-lm` gives.
#   Linux: mlx-lm[cpu], i.e. mlx==0.32.4 + mlx-cpu==0.32.4 (manylinux_2_35 wheels).
#
# The adapter imports mlx_lm.server (which imports mlx.core at module level) to
# drive the server's own handle_completion loop, so mlx must import here. Weights
# are never loaded and no model ever runs: replay needs the tokenizer only.
#
# Then warms the HF cache with every pinned tokenizer/config/template file (never
# weights) through scripts/engines/prefetch_hf.py, so replays can run with
# HF_HUB_OFFLINE=1. Pass --no-prefetch to skip that step.
#
# Exit codes: 0 ready; 3 mlx (or mlx_lm.server) cannot be imported on this
# platform, so the engine is reported as not set up (the venv is removed);
# anything else is a setup failure.
#
# Usage: scripts/engines/mlx_lm.sh [--no-prefetch]
# Override the interpreter used by the runner with $CANITOOLCALL_MLX_LM_PYTHON.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$ROOT/.venvs/mlx_lm"
PY="$VENV/bin/python"
MLX_LM_VERSION="0.32.2"
MLX_VERSION="0.32.4"

prefetch=1
for arg in "$@"; do
  case "$arg" in
    --no-prefetch) prefetch=0 ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

command -v uv >/dev/null || { echo "uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }

# mlx publishes macOS wheels for arm64 only, and Linux wheels for x86_64 and aarch64.
case "$(uname -s)/$(uname -m)" in
  Darwin/arm64) spec="mlx-lm==${MLX_LM_VERSION}" ;;
  Linux/x86_64 | Linux/aarch64) spec="mlx-lm[cpu]==${MLX_LM_VERSION}" ;;  # mlx-lm requires mlx only on Darwin
  *) echo "mlx_lm: not set up: mlx has no wheels for $(uname -s) $(uname -m)" >&2; exit 3 ;;
esac

if [[ ! -x "$PY" ]]; then
  uv python install 3.12
  uv venv -p 3.12 "$VENV"
fi
# transformers/tokenizers/huggingface-hub: mlx-lm leaves them unpinned (transformers>=5.7.0);
# these are the versions the other replay envs use. tiktoken: the Kimi tokenizers need it.
uv pip install -p "$PY" "$spec" "mlx==${MLX_VERSION}" \
  "transformers==5.17.0" "tokenizers==0.23.2" "huggingface-hub==1.33.0" "tiktoken==0.14.0"

# Sanity: pinned versions, and the server module imports (it needs mlx.core).
set +e
"$PY" - "$MLX_LM_VERSION" "$MLX_VERSION" <<'EOF'
import sys
from importlib import metadata

want_lm, want_mlx = sys.argv[1:3]
assert metadata.version("mlx-lm") == want_lm, f"mlx-lm {metadata.version('mlx-lm')} != {want_lm}"
assert metadata.version("mlx") == want_mlx, f"mlx {metadata.version('mlx')} != {want_mlx}"
try:
    import mlx.core as mx
    import mlx_lm.server  # noqa: F401
except Exception as e:  # ImportError, or a missing shared library / backend
    print(f"mlx cannot be imported on this platform: {type(e).__name__}: {e}", file=sys.stderr)
    raise SystemExit(3)
print(f"mlx-lm {want_lm} OK (mlx {mx.__version__}, default device {mx.default_device()})")
EOF
code=$?
set -e
if [[ "$code" == 3 ]]; then
  rm -rf "$VENV"
  echo "mlx_lm: not set up on $(uname -s) $(uname -m) (removed $VENV)" >&2
fi
[[ "$code" == 0 ]] || exit "$code"

if [[ "$prefetch" == 1 ]]; then
  "$PY" "$ROOT/scripts/engines/prefetch_hf.py"
fi

echo "ready: $PY"

#!/usr/bin/env bash
# Build the isolated HF transformers replay env: .venvs/transformers
#
#   transformers==5.23.0 + jinja2 + jmespath, Python 3.12, NO torch.
#   (tokenizer.parse_response / ResponseParser and the tokenizers DecodeStream
#   only need the tokenizer stack; torch is never imported.)
#
# Then warms the HF cache with the tokenizer/generation-config files (never weights) of
# every repo the adapter supports, at the revisions pinned in
# src/canitoolcall/adapters/transformers.py, so tests can run with
# HF_HUB_OFFLINE=1. Pass --no-prefetch to skip that step.
#
# Usage: scripts/engines/transformers.sh [--no-prefetch]
# Override the interpreter used by the runner with $CANITOOLCALL_TRANSFORMERS_PYTHON.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$ROOT/.venvs/transformers"
PY="$VENV/bin/python"
TRANSFORMERS_VERSION="5.23.0"

prefetch=1
for arg in "$@"; do
  case "$arg" in
    --no-prefetch) prefetch=0 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

command -v uv >/dev/null || { echo "uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }

if [[ ! -x "$PY" ]]; then
  uv python install 3.12
  uv venv -p 3.12 "$VENV"
fi
uv pip install -p "$PY" "transformers==${TRANSFORMERS_VERSION}" "jinja2>=3.1" "jmespath>=1.0"

# Sanity: pinned version, no torch.
"$PY" - "$TRANSFORMERS_VERSION" <<'EOF'
import importlib.util, sys
import transformers
want = sys.argv[1]
assert transformers.__version__ == want, f"transformers {transformers.__version__} != {want}"
assert importlib.util.find_spec("torch") is None, "torch must not be installed in .venvs/transformers"
print(f"transformers {transformers.__version__} OK (no torch)")
EOF

if [[ "$prefetch" == 1 ]]; then
  PYTHONPATH="$ROOT/src" "$PY" - <<'EOF'
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from transformers.utils import logging

from canitoolcall.adapters.transformers import PINNED_REPOS

logging.set_verbosity_error()
for repo, revision in sorted(PINNED_REPOS.items()):
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    hf_hub_download(repo, "generation_config.json", revision=revision)  # stop ids for fixture generators
    ships = getattr(tok, "response_template", None) is not None
    print(f"cached {repo}@{revision[:12]} response_template={'yes' if ships else 'NO'}")
EOF
fi

echo "ready: $PY"

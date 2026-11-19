#!/usr/bin/env bash
# Build the isolated vLLM environment used by canitoolcall's vLLM adapter.
#
#   bash scripts/engines/vllm.sh
#
# vLLM ships Linux-only (manylinux) wheels, but its tool/reasoning parsers,
# tokenizer wrappers and v1 detokenizer are pure Python. So instead of
# `pip install vllm` we:
#   1. download the pinned manylinux wheel for this CPU arch from PyPI and
#      verify its sha256 against PyPI's JSON API,
#   2. unzip it under .engines/vllm/vllm-<ver>/ (gitignored),
#   3. create a Python 3.12 venv at .venvs/vllm (gitignored),
#   4. install the minimal runtime deps the parser/tokenizer path imports
#      (incl. CPU torch: vllm/__init__ imports it), pinned,
#   5. put the unpacked wheel on sys.path through a .pth file.
# No model weights are downloaded, nothing is published.
#
# Env overrides: VLLM_VERSION (default 0.34.0), CANITOOLCALL_VLLM_WHEEL (use a
# local copy of the exact wheel instead of downloading; the sha256 is still
# checked against PyPI).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLLM_VERSION="${VLLM_VERSION:-0.34.0}"
ENGINE_DIR="$ROOT/.engines/vllm"
SRC_DIR="$ENGINE_DIR/vllm-$VLLM_VERSION"
VENV="$ROOT/.venvs/vllm"
PY="$VENV/bin/python"

command -v uv >/dev/null || { echo "uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }
command -v unzip >/dev/null || { echo "unzip is required" >&2; exit 1; }

arch="$(uname -m)"
case "$arch" in
  arm64 | aarch64) arch=aarch64 ;;
  x86_64 | amd64) arch=x86_64 ;;
  *) echo "unsupported CPU arch: $arch" >&2; exit 1 ;;
esac

mkdir -p "$ENGINE_DIR" "$ROOT/.venvs"

# --- 1. locate the wheel on PyPI (filename, url, sha256) ---------------------
read -r WHEEL_NAME WHEEL_URL WHEEL_SHA < <(
  curl -fsSL "https://pypi.org/pypi/vllm/$VLLM_VERSION/json" | python3 -c '
import json, sys
arch = sys.argv[1]
files = [f for f in json.load(sys.stdin)["urls"]
         if f["packagetype"] == "bdist_wheel" and "manylinux" in f["filename"] and arch in f["filename"]]
if not files:
    sys.exit(f"no manylinux {arch} wheel on PyPI")
f = files[0]
print(f["filename"], f["url"], f["digests"]["sha256"])
' "$arch"
)
# Digests committed here, so a compromised index cannot swap the wheel (PyPI's
# JSON is only used to find the URL). Other versions fall back to PyPI's digest.
pinned_sha() {
  case "$1" in
    vllm-0.34.0-cp38-abi3-manylinux_2_28_aarch64.whl) echo 7d96231fc141ad4b3a441292b77362b285b335ca7d1b0e2ab441d4ebde82f405 ;;
    vllm-0.34.0-cp38-abi3-manylinux_2_28_x86_64.whl) echo 79ca4b3cd74aec08c4bf476bdf0cb15735b605ae5368526a2faae1e30f044ca2 ;;
    *) echo "" ;;
  esac
}
PINNED_SHA="$(pinned_sha "$WHEEL_NAME")"
if [[ -z "$PINNED_SHA" ]]; then
  echo "warning: no committed sha256 for $WHEEL_NAME; trusting PyPI's ($WHEEL_SHA)" >&2
elif [[ "$PINNED_SHA" != "$WHEEL_SHA" ]]; then
  echo "PyPI reports sha256 $WHEEL_SHA for $WHEEL_NAME, but the committed digest is $PINNED_SHA" >&2
  exit 1
fi
WHEEL="$ENGINE_DIR/$WHEEL_NAME"

sha256_of() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

if [[ ! -f "$WHEEL" ]]; then
  if [[ -n "${CANITOOLCALL_VLLM_WHEEL:-}" ]]; then
    cp "$CANITOOLCALL_VLLM_WHEEL" "$WHEEL.part"
  else
    echo "downloading $WHEEL_NAME"
    curl -fSL --retry 3 -o "$WHEEL.part" "$WHEEL_URL"
  fi
  got="$(sha256_of "$WHEEL.part")"
  if [[ "$got" != "$WHEEL_SHA" ]]; then
    rm -f "$WHEEL.part"
    echo "sha256 mismatch for $WHEEL_NAME: got $got, PyPI says $WHEEL_SHA" >&2
    exit 1
  fi
  mv "$WHEEL.part" "$WHEEL"
fi

# --- 2. unpack ----------------------------------------------------------------
if [[ ! -f "$SRC_DIR/vllm/__init__.py" ]]; then
  rm -rf "$SRC_DIR.tmp"
  unzip -q -o "$WHEEL" -d "$SRC_DIR.tmp"
  rm -rf "$SRC_DIR"
  mv "$SRC_DIR.tmp" "$SRC_DIR"
fi
echo "$WHEEL_SHA  $WHEEL_NAME" >"$SRC_DIR/WHEEL.sha256"

# --- 3. venv ------------------------------------------------------------------
uv python install 3.12 >/dev/null
uv venv -q --allow-existing -p 3.12 "$VENV"

# --- 4. minimal pinned deps ---------------------------------------------------
# The set the parser / tokenizer / detokenizer imports need, pinned to the
# versions the adapter was verified with. They satisfy vLLM 0.34.0's
# requirements/common.txt floors (huggingface_hub>=2, which transformers 5.17
# excludes, tokenizers>=0.23.3, mistral_common>=1.13.0, aiohttp>=3.14.4 and
# oss-harmony>=0.0.11, the `openai_harmony` fork vLLM switched to in #55128);
# the bumped ones match vLLM's v0.34.0 requirements/test/cpu.txt.
DEPS=(
  "transformers==5.22.0" "tokenizers==0.23.3" "huggingface-hub==2.4.0"
  "pydantic==2.13.5" "openai==3.19.2" "partial-json-parser==0.2.1.1.post7" "regex==2026.9.10"
  "oss-harmony==0.0.11" "mistral-common==1.13.0" "fastapi==0.136.3"
  "msgspec==0.21.1" "cachetools==7.2.0" "psutil==7.2.2" "jsonschema==4.26.0"
  "pyyaml==6.0.3" "sentencepiece==0.2.2" "tiktoken==0.14.0" "protobuf==7.36.2"
  "numpy==2.3.5" "pybase64==1.5.0" "cbor2==6.1.4" "blake3==1.0.9"
  "cloudpickle==3.1.2" "pyzmq==27.2.0" "aiohttp==3.14.5"
  "prometheus-client==0.26.0" "uvloop==0.22.1" "py-cpuinfo==9.0.0"
  "xgrammar==0.2.8" "llguidance==1.7.6" "ijson==3.5.1" "einops==0.8.1"
)
TORCH="torch==2.14.0"
if [[ "$(uname -s)" == "Linux" ]]; then
  # CPU-only torch on Linux (the default PyPI wheel pulls CUDA).
  uv pip install -q -p "$PY" --index-url https://download.pytorch.org/whl/cpu "$TORCH"
else
  uv pip install -q -p "$PY" "$TORCH"   # macOS wheels are CPU/MPS only
fi
uv pip install -q -p "$PY" "${DEPS[@]}"

# --- 5. expose the unpacked wheel -------------------------------------------
SITE="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
rm -f "$SITE"/vllm_src.pth "$SITE"/canitoolcall_vllm.pth
echo "$SRC_DIR" >"$SITE/canitoolcall_vllm.pth"

# --- smoke test ---------------------------------------------------------------
"$PY" - <<'EOF'
import vllm
from vllm.parser.parser_manager import ParserManager  # noqa: F401
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager
from vllm.v1.engine.detokenizer import IncrementalDetokenizer  # noqa: F401

print(
    f"vllm {vllm.__version__} from {vllm.__file__}: "
    f"{len(ToolParserManager.list_registered())} tool parsers, "
    f"{len(ReasoningParserManager.list_registered())} reasoning parsers"
)
EOF
echo "ready: $PY"

#!/usr/bin/env bash
# Build the isolated SGLang environment used by the sglang adapter.
#
#   bash scripts/engines/sglang.sh
#
# SGLang only publishes Linux wheels, but its parsers, detokenizer and
# OpenAI serving layer are pure Python. So we:
#   1. download the pinned cp312 manylinux wheel for this CPU architecture
#      (sha256 checked against PyPI) into .engines/sglang/,
#   2. unzip it to .engines/sglang/src,
#   3. create a Python 3.12 venv at .venvs/sglang with the minimal runtime deps
#      (found by importing the parser, detokenizer and serving_chat modules),
#   4. add the unpacked source to the venv through a .pth file.
#
# canitoolcall itself is NOT installed into the venv: the runner starts the
# worker with PYTHONPATH=<repo>/src. Nothing here is published anywhere.
set -euo pipefail

SGLANG_VERSION=0.5.23
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENGINE_DIR="$ROOT/.engines/sglang"
SRC_DIR="$ENGINE_DIR/src"
VENV="$ROOT/.venvs/sglang"
PY="$VENV/bin/python"

# Runtime deps of the modules the adapter imports. Versions that SGLang 0.5.23
# pins in its wheel METADATA (Requires-Dist ==) are used exactly: tokenizer
# behaviour depends on them (e.g. transformers/tokenizers decode). The smoke
# test below fails if an installed version drifts from those pins. The rest
# were validated on 2026-09-25 and re-checked with 0.5.23 on 2026-11-19 (CPU
# torch is only imported transitively).
DEPS=(
  # pinned by sglang-0.5.23 METADATA
  "transformers==5.19.0" "tokenizers==0.23.2" "torch==2.14.1" "xgrammar==0.2.7"
  "openai==2.8.0" "blobfile==3.0.0" "compressed-tensors==0.18.0" "apache-tvm-ffi==0.1.11"
  "sentencepiece==0.2.1"
  # DeepSeek-V4/V4.1 prompts and output parsing (serving_deepseek_recipe.py); pinned by the METADATA
  "deepseek-recipe==0.2.0"
  # unpinned upstream; versions validated here
  "huggingface-hub" "torchvision" "pydantic==2.13.5" "partial-json-parser==0.2.1.1.post7"
  "orjson==3.12.0" "regex==2026.9.10" "psutil==7.2.2" "pybase64==1.5.0"
  "requests==2.34.2" "pillow==12.3.0" "starlette==1.7.0" "pyzmq==27.2.0"
  "ipython==9.17.1" "aiohttp==3.14.3" "tiktoken==0.14.0" "msgspec==0.21.1"
  # needed by sglang.srt.managers.detokenizer_manager / serving_chat imports
  "setproctitle==1.3.7" "dill==0.4.1" "gguf==0.19.0"
  "fastapi==0.141.1" "uvloop==0.22.1" "uvicorn==0.54.0" "einops==0.8.2" "jsonschema==4.26.0"
  # get_tokenizer loads bare-tekken Mistral checkpoints via mistral-common (sglang: >=1.11.5)
  "mistral-common==1.12.0"
)

arch="$(uname -m)"
case "$arch" in
  arm64 | aarch64) arch=aarch64 ;;
  x86_64 | amd64) arch=x86_64 ;;
  *) echo "unsupported architecture: $arch" >&2; exit 1 ;;
esac

command -v uv >/dev/null || { echo "uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }
mkdir -p "$ENGINE_DIR" "$ROOT/.venvs"

# --- 1+2. wheel -> .engines/sglang/src ------------------------------------------
# Re-unpack when the pin changes: the dist-info directory names the version.
if [ ! -d "$SRC_DIR/sglang-$SGLANG_VERSION.dist-info" ]; then
  read -r wheel_name wheel_url wheel_sha < <(
    curl -fsSL "https://pypi.org/pypi/sglang/$SGLANG_VERSION/json" | python3 -c '
import json, sys
arch = sys.argv[1]
for f in json.load(sys.stdin)["urls"]:
    n = f["filename"]
    if "cp312-cp312-manylinux" in n and n.endswith(arch + ".whl"):
        print(n, f["url"], f["digests"]["sha256"])
        break
else:
    sys.exit("no cp312 manylinux wheel for " + arch)
' "$arch"
  )
  # Digests committed here, so a compromised index cannot swap the wheel.
  case "$wheel_name" in
    sglang-0.5.23-cp312-cp312-manylinux_2_34_aarch64.whl) pinned=2d538a89a6e7b1b0707e2ab7f8edef4af6a7beb5da661f9a231cddcdf5308417 ;;
    sglang-0.5.23-cp312-cp312-manylinux_2_34_x86_64.whl) pinned=8d257a319395b322bb1be327d05e5528859358153c8563b04c0a4a3c8af164b6 ;;
    *) echo "no committed sha256 for $wheel_name" >&2; exit 1 ;;
  esac
  if [ "$pinned" != "$wheel_sha" ]; then
    echo "PyPI reports sha256 $wheel_sha for $wheel_name, but the committed digest is $pinned" >&2
    exit 1
  fi
  wheel="$ENGINE_DIR/$wheel_name"
  [ -f "$wheel" ] || curl -fsSL -o "$wheel" "$wheel_url"
  got="$(python3 -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$wheel")"
  if [ "$got" != "$wheel_sha" ]; then
    echo "sha256 mismatch for $wheel_name: $got != $wheel_sha" >&2
    rm -f "$wheel"
    exit 1
  fi
  rm -rf "$SRC_DIR"
  unzip -q -o "$wheel" -d "$SRC_DIR"
  echo "unpacked $wheel_name ($wheel_sha)"
fi

# --- 3. venv + deps ------------------------------------------------------------
uv python install 3.12 >/dev/null
uv venv -q --allow-existing -p 3.12 "$VENV"
uv pip install -q -p "$PY" "${DEPS[@]}"

# --- 4. .pth -> unpacked source --------------------------------------------------
site="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
# Drop any earlier .pth that points at another SGLang tree (e.g. the spike's).
for f in "$site"/*.pth; do
  [ -e "$f" ] || continue
  if grep -q "sglang" "$f" 2>/dev/null && [ "$(basename "$f")" != "canitoolcall_sglang.pth" ]; then
    rm -f "$f"
  fi
done
echo "$SRC_DIR" > "$site/canitoolcall_sglang.pth"

# --- smoke test ------------------------------------------------------------------
# Parsing needs no GPU. With a CUDA GPU visible, torch.cuda.is_available() is true
# and SGLang takes CUDA-only import branches (flashinfer) that this CPU-only env
# lacks, so hide the GPUs. `canitoolcall run` does the same for the worker.
CUDA_VISIBLE_DEVICES=-1 "$PY" -W ignore - "$SRC_DIR" <<'EOF'
import re
import sys
from importlib import metadata
from pathlib import Path

import sglang
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.managers.detokenizer_manager import DetokenizerManager  # noqa: F401
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat  # noqa: F401

meta = next(Path(sys.argv[1]).glob("sglang-*.dist-info")) / "METADATA"
drift = []
for line in meta.read_text().splitlines():
    m = re.match(r"Requires-Dist: ([A-Za-z0-9_.-]+)==([^;\s]+)$", line)
    if not m:
        continue
    try:
        have = metadata.version(m.group(1))
    except metadata.PackageNotFoundError:
        continue  # not needed by the parsing/serving modules
    if have != m.group(2):
        drift.append(f"{m.group(1)}: installed {have}, sglang pins {m.group(2)}")
if drift:
    sys.exit("dependency drift from SGLang's pins:\n  " + "\n  ".join(drift))
print(
    f"sglang {sglang.__version__} from {sglang.__file__}: "
    f"{len(FunctionCallParser.ToolCallParserEnum)} tool parsers, "
    f"{len(ReasoningParser.DetectorMap)} reasoning parsers"
)
EOF
echo "ready: $PY"

#!/usr/bin/env bash
# Build the llama.cpp replay engine for canitoolcall.
#
#   1. Shallow-fetch llama.cpp at the pinned commit into .engines/llamacpp/llama.cpp
#   2. Build the JSON-lines replay harness (harnesses/llamacpp) against it with
#      cmake: ggml (CPU only) + llama + llama-common + our tool. No model, no GPU.
#      -> .engines/llamacpp/build/canitoolcall-llamacpp
#   3. Create the isolated venv .venvs/llamacpp (Python 3.12) with llama.cpp's own
#      requirements-convert_hf_to_gguf.txt, used by scripts/engines/gguf_vocab.sh to
#      make vocab-only GGUFs. The adapter's worker also runs in this venv (it only
#      needs the standard library; the parser runs in the compiled harness).
#      Also .engines/llamacpp/convert-tf5: the same requirements with transformers
#      5.17.0, a recorded fallback for tokenizer files transformers 4.57.6 cannot read.
#
# Usage: scripts/engines/llamacpp.sh [--no-harness] [--no-venv] [--no-vocab]
#   --no-harness  skip step 2 (converter only: what the Ollama job needs to make the
#                 vocab-only GGUFs it shares with this adapter; no cmake needed)
#   --no-venv   skip step 3
#   --no-vocab  do not build the vocab-only GGUFs for the fixture reference models
#               (by default, scripts/engines/gguf_vocab.sh --all runs at the end)
# Env: LLAMACPP_REPO (default https://github.com/ggml-org/llama.cpp), JOBS.
# Override the harness binary with $CANITOOLCALL_LLAMACPP_HARNESS and the worker
# interpreter with $CANITOOLCALL_LLAMACPP_PYTHON.
set -euo pipefail

LLAMACPP_REF="${LLAMACPP_REF:-a25c9865fe03c954c93fd755b5d79ae86ba99750}"  # keep in sync with LlamaCppAdapter.pinned_version

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENGINE="$ROOT/.engines/llamacpp"
SRC="$ENGINE/llama.cpp"
BUILD="$ENGINE/build"
VENV="$ROOT/.venvs/llamacpp"
REPO="${LLAMACPP_REPO:-https://github.com/ggml-org/llama.cpp}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)}"

harness=1
venv=1
vocab=1
for arg in "$@"; do
  case "$arg" in
    --no-harness) harness=0 ;;
    --no-venv) venv=0 ;;
    --no-vocab) vocab=0 ;;
    -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
[[ $harness == 0 ]] || command -v cmake >/dev/null || { echo "cmake is required (e.g. brew install cmake)" >&2; exit 1; }

[[ "$LLAMACPP_REF" =~ ^[0-9a-f]{40}$ ]] || { echo "LLAMACPP_REF must be a full commit SHA" >&2; exit 2; }

# 1. Source at the pin (shallow fetch of exactly one commit).
mkdir -p "$ENGINE"
if [[ ! -d "$SRC/.git" ]]; then
  git init -q "$SRC"
  git -C "$SRC" remote add origin "$REPO"
fi
if [[ "$(git -C "$SRC" rev-parse -q --verify HEAD 2>/dev/null || true)" != "$LLAMACPP_REF" ]]; then
  git -C "$SRC" fetch -q --depth 1 origin "$LLAMACPP_REF"
  git -C "$SRC" checkout -q --detach FETCH_HEAD
fi
head="$(git -C "$SRC" rev-parse HEAD)"
[[ "$head" == "$LLAMACPP_REF" ]] || { echo "llama.cpp is at $head, expected $LLAMACPP_REF" >&2; exit 1; }
if [[ -n "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]]; then
  echo "llama.cpp checkout has local modifications; refusing to build a non-pinned tree" >&2
  exit 1
fi

# 2. Harness build (Release, CPU only; see harnesses/llamacpp/CMakeLists.txt).
if [[ $harness == 1 ]]; then
  cmake -S "$ROOT/harnesses/llamacpp" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CPP_DIR="$SRC" >/dev/null
  cmake --build "$BUILD" --target canitoolcall-llamacpp -j "$JOBS" >/dev/null
  BIN="$BUILD/canitoolcall-llamacpp"
  echo '{"op":"hello"}' | "$BIN" | head -1
  echo "built $BIN"
fi

# 3. Converter / worker venv.
if [[ $venv == 1 ]]; then
  command -v uv >/dev/null || { echo "uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }
  if [[ ! -x "$VENV/bin/python" ]]; then
    uv python install 3.12
    uv venv -q -p 3.12 "$VENV"
  fi
  # llama.cpp's pinned converter requirements (CPU torch comes with them), plus the
  # optional tokenizer backends the converter imports for some repos (Kimi:
  # tiktoken/blobfile; Mistral-format repos: mistral-common, see conversion/base.py),
  # and huggingface_hub for tokenizer-only downloads.
  # Exact pins (the versions the vocab GGUFs were built with); huggingface_hub differs
  # per env because transformers 4.57.6 needs <1.0 and transformers 5.17.0 needs >=1.0.
  EXTRAS=("tiktoken==0.14.0" "blobfile==3.3.0" "mistral-common[image,audio]==1.12.0")
  # unsafe-best-match: the requirements add the PyTorch CPU index, which on Linux also
  # hosts an old `requests` (2.28.1, urllib3<1.27) that conflicts with blobfile>=3.3.
  uv pip install -q -p "$VENV/bin/python" --index-strategy unsafe-best-match \
    -r "$SRC/requirements/requirements-convert_hf_to_gguf.txt" "${EXTRAS[@]}" "huggingface_hub==0.36.2"
  echo "venv ready: $VENV"

  # Fallback converter env for repos whose tokenizer files need transformers 5
  # (used by gguf_vocab.sh only after the pinned env fails; recorded per GGUF).
  TF5="$ENGINE/convert-tf5"
  [[ -x "$TF5/bin/python" ]] || uv venv -q -p 3.12 "$TF5"
  grep -v '^transformers' "$SRC/requirements/requirements-convert_legacy_llama.txt" > "$ENGINE/requirements-convert-tf5.txt"
  uv pip install -q -p "$TF5/bin/python" --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu \
    -r "$ENGINE/requirements-convert-tf5.txt" "$(grep "^torch" "$SRC/requirements/requirements-convert_hf_to_gguf.txt")" \
    "transformers==5.17.0" "${EXTRAS[@]}" "huggingface_hub==1.33.0"
  echo "fallback converter env ready: $TF5"
fi

if [[ $vocab == 1 && $venv == 1 ]]; then
  "$ROOT/scripts/engines/gguf_vocab.sh" --all
fi

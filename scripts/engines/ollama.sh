#!/usr/bin/env bash
# Build the Ollama replay environment (offline afterwards; nothing is published).
#
#   .engines/ollama/ollama/        Ollama clone at OLLAMA_REF
#   .engines/ollama/llama.cpp/     llama.cpp at Ollama's own LLAMA_CPP_VERSION tag
#   .engines/ollama/bin/ctcreplay  Go harness (harnesses/ollama/ctcreplay) linked
#                                  against the pinned model/parsers package
#   .engines/ollama/bin/ctc-detok  llama-server token rendering from vocab GGUFs
#   .venvs/ollama/                 tiny interpreter for the adapter worker
#                                  (the adapter is stdlib-only)
#
# Vocab-only GGUFs are shared with the llama.cpp adapter: they are produced by
# scripts/engines/gguf_vocab.sh into .engines/gguf/<org>--<model>.vocab.gguf
# (override the directory with CANITOOLCALL_GGUF_DIR).
#
# Requires: git, go (GOTOOLCHAIN=auto fetches the go.mod toolchain), cmake, a
# C++17 compiler, uv.
set -euo pipefail

REQUESTED_OLLAMA_REF="${OLLAMA_REF:-}"
OLLAMA_REF=7af393188defd52d370464de0d2064649cab9b41   # keep in sync with OllamaAdapter.pinned_version
OLLAMA_REF="${REQUESTED_OLLAMA_REF:-$OLLAMA_REF}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENG="$ROOT/.engines/ollama"
SRC="$ENG/ollama"
BIN="$ENG/bin"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)}"
venv=1
for arg in "$@"; do
  case "$arg" in
    --no-venv) venv=0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
[[ "$OLLAMA_REF" =~ ^[0-9a-f]{40}$ ]] || { echo "OLLAMA_REF must be a full commit SHA" >&2; exit 2; }
mkdir -p "$ENG" "$BIN"

fetch() {  # $1=repo url  $2=dir  $3=ref (commit or tag)
  if [ ! -d "$2/.git" ]; then
    git init -q "$2"
    git -C "$2" remote add origin "$1"
  fi
  if [ "$(git -C "$2" rev-parse -q --verify HEAD 2>/dev/null)" != "$(git -C "$2" rev-parse -q --verify "$3^{commit}" 2>/dev/null || echo x)" ]; then
    git -C "$2" fetch -q --depth 1 origin "$3"
    git -C "$2" checkout -q --detach FETCH_HEAD
  fi
}

echo "== Ollama @ $OLLAMA_REF"
fetch https://github.com/ollama/ollama "$SRC" "$OLLAMA_REF"
test "$(git -C "$SRC" rev-parse HEAD)" = "$OLLAMA_REF"

echo "== Go harness (cmd/ctcreplay)"
mkdir -p "$SRC/cmd/ctcreplay"
cp "$ROOT/harnesses/ollama/ctcreplay/main.go" "$SRC/cmd/ctcreplay/main.go"
(cd "$SRC" && CGO_ENABLED=0 GOTOOLCHAIN=auto go build -trimpath \
  -ldflags "-X main.ollamaCommit=$OLLAMA_REF" -o "$BIN/ctcreplay" ./cmd/ctcreplay)

LLAMA_TAG="$(tr -d '[:space:]' < "$SRC/LLAMA_CPP_VERSION")"
echo "== llama.cpp @ $LLAMA_TAG (Ollama's LLAMA_CPP_VERSION)"
if [ ! -d "$ENG/llama.cpp/.git" ] || [ "$(git -C "$ENG/llama.cpp" describe --tags --exact-match 2>/dev/null || true)" != "$LLAMA_TAG" ]; then
  if [ ! -d "$ENG/llama.cpp/.git" ]; then
    git init -q "$ENG/llama.cpp"
    git -C "$ENG/llama.cpp" remote add origin https://github.com/ggml-org/llama.cpp
  fi
  git -C "$ENG/llama.cpp" fetch -q --depth 1 origin tag "$LLAMA_TAG"
  git -C "$ENG/llama.cpp" checkout -q --detach "$LLAMA_TAG"
fi

echo "== detokenizer (ctc-detok)"
cmake -S "$ROOT/harnesses/ollama/detok" -B "$ENG/detok-build" -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CPP_SOURCE="$ENG/llama.cpp" -DLLAMA_BUILD_NUMBER="${LLAMA_TAG#b}" >/dev/null
cmake --build "$ENG/detok-build" --target ctc-detok -j "$JOBS" >/dev/null
cp "$ENG/detok-build/ctc-detok" "$BIN/ctc-detok"

echo "== worker venv (.venvs/ollama)"
if [[ $venv == 1 ]]; then
  uv venv -q --allow-existing -p 3.12 "$ROOT/.venvs/ollama"
fi

echo "== smoke test"
echo '{"op":"describe","parser":"qwen3-coder"}' | "$BIN/ctcreplay"
echo '{"op":"hello"}' | "$BIN/ctc-detok"
echo "built $BIN/ctcreplay and $BIN/ctc-detok"
if [ ! -d "${CANITOOLCALL_GGUF_DIR:-$ROOT/.engines/gguf}" ]; then
  echo "note: no vocab GGUFs yet; run scripts/engines/llamacpp.sh --no-harness (converter + gguf_vocab.sh --all) before replaying"
fi

# Verify compiled engine fixes

The compiled route of `scripts/verify_upstream_pr.py` builds Ollama or llama.cpp
from the complete, immutable base/head SHAs returned for the selected pull
request. The Python-engine overlay route is unchanged.

```sh
uv sync
uv run python scripts/verify_upstream_pr.py --engine ollama --pr 18663 --family glm --build-jobs 2 --json report.json
```

A POSIX build environment with GitHub CLI, git, bash, Go (for Ollama), CMake
and a C++ compiler is needed. Build the normal vocabulary-only GGUF assets
for the selected families first, as described by `scripts/engines/gguf_vocab.sh`.
No weights, inference server or live probe is required. Missing assets or an
incompatible build do not establish an improvement.

## Isolation and source identity

Each side receives a new temporary project containing the trusted build
recipes and harness source from this checkout. Its `.engines` tree lives
inside that project. Engine source files are fetched by full SHA, and the
resolved checkout is checked before replay. Base and head are not overlays
onto the normal pinned engine.

A small, temporary adapter subclass declares that side's expected commit.
The existing adapter's version check still runs. This avoids relaxing the
normal adapter's pin, modifying its installed source, or changing the default
engine for other work. Parser provenance uses the side's declared revision.

The usual vocabulary directory is read by both sides, or use the existing
`CANITOOLCALL_GGUF_DIR` override. No worker venv is created by either isolated
build. Compilation defaults to two parallel jobs; `--build-jobs` is explicit.
The builders accept full SHA overrides through `OLLAMA_REF` and `LLAMACPP_REF`,
while their defaults stay at the original pins.

## Results and failure handling

The optional JSON report must use a new file outside `.engines`, `.venvs` and
`.git`. Existing files are not overwritten. The destination is resolved and
checked before builds start, and report publication never replaces a target. This
also avoids following an existing report alias into a pinned binary. It is
not a guarantee against an adversary changing parent directories concurrently.

The command prints the per-fixture differences and writes source SHAs,
binary SHA-256 values and the work directory to its JSON report. Build and
replay logs are retained in that directory, even after a failed build.
The normal pinned harness files are hashed before and after the operation.
The hashes distinguish an absent pre-existing build from a preserved one.

Exit 0 means no regression in a complete, comparable replay; exit 1 means a
previously passing fixture failed; exit 3 means no reliable comparison was
obtained. Missing, malformed or duplicate case records, changed coverage,
harness errors and build failures return 3 instead of a misleading success.
The same statuses and changed-fixture shape as the existing verifier are used.

Reports are completely written in a temporary file in the destination directory
before publication. The optional export uses a no-clobber hard link, requiring
filesystem hard-link support; lack of support is an explicit report error.
Partial writes are not published as the requested export. This is visibility
and error-handling behavior, not a power-loss durability guarantee.

Report I/O failure returns 3 with `status=report_error` and preserves the prior
comparison exit as `operation_exit_code`. The internal receipt is corrected
when possible. If that write also fails, stderr emits a complete JSON record
prefixed `VERIFICATION_REPORT_JSON `. That terminal record and process exit
are authoritative: an older internal receipt may remain if correction failed.
No saved-receipt guarantee is made if both storage and stderr are unavailable.

This compares the chosen fixture families, not every model, tokenizer, server
configuration or arbitrary inference behavior. A compiling harness and a
successful parser comparison are separate milestones. Do not describe
mocked orchestration tests as real upstream compilation or fixture replay.

The Windows host should run compilation inside its Linux/WSL environment;
only the Python orchestration tests are intended to be host-independent.

## Cancellation and process lifetime

Build and replay commands run in a private POSIX session whose small Python
supervisor retains the process-group ID until cleanup is complete. Only that
group is signalled; unrelated processes are not enumerated or terminated.
On timeout, SIGINT or SIGTERM, cleanup sends TERM, allows a short grace period,
then sends KILL to remaining group members. The direct supervisor is reaped.
Normal completion also cleans descendants left behind by the foreground tool.
The same mechanism supports Linux and macOS without waitid/WNOWAIT.

Timeout is an explicit incomplete comparison (exit 3). Handled SIGINT and
SIGTERM are recorded as interruption (exit 130 or 143); a report-write failure
can still yield exit 3 with the original operation code retained. These
semantics apply while a managed build or replay is running. The subprocess
runner requires the POSIX main thread; use WSL from Windows.

Tests use expiring, owned local child/grandchild processes, including a child
that ignores TERM, and confirm that an unrelated test process survives.
Descendants that deliberately detach into another session, an uncatchable
kill of the supervising Python process, uninterruptible kernel waits and
OS failure are outside this process-group contract. This is not a general
service supervisor or a guarantee of cleanup after power loss.

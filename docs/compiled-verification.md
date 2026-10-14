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

## Running upstream code

This command **compiles and executes upstream pull-request code**, including
build-time smoke tests. Read the source first and use a disposable, low-privilege
container or VM without personal credentials for code you do not trust.
Temporary build directories and process-group cleanup are not a security sandbox.

The parent uses GitHub CLI authentication only to read PR metadata. Build, smoke
and replay children receive an allowlisted environment: PATH/HOME, locale and
temporary-directory settings, named compiler/CMake/macOS/Go settings, UV_CACHE_DIR,
and the known offline CANITOOLCALL fixture, vocabulary and compiled-adapter paths.
The exact list is `CHILD_ENVIRONMENT_KEYS` in `verify_compiled_pr.py`. It excludes
GH_TOKEN, GITHUB_TOKEN, HF_TOKEN, CANITOOLCALL_API_KEY and unrelated variables;
it does not permit every variable merely because it starts with CANITOOLCALL_.
The per-side revision and binary paths override inherited values.

HOME, PATH, allowed toolchain settings and files on disk remain accessible.
An allowed value can itself contain sensitive data. Filtering the environment
does not block malicious code from reading files or using the network. Arbitrary
`--env` forwarding is Python-engine-only and is rejected on the compiled route,
rather than silently ignored or used to bypass the allowlist.

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
`--build-timeout SECONDS` sets a positive time limit for each side's complete
build command, including its smoke test; the default is 3600 seconds (one hour).
Replay retains its separate 900-second limit. A timed-out build reports its
engine, elapsed budget and build-log path, and returns exit 3 after cleanup.
`--build-jobs` and `--build-timeout` are rejected for Python engines.
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
obtained. Missing, malformed or duplicate case records and build failures return
3 instead of a misleading success. A valid per-fixture `error` (including a
recorded `harness_error`) is retained, just as on the Python path: error-to-pass
is a fix, pass-to-error a regression, and a pre-existing error alone does not
invalidate the comparison. The summary includes the same `other changes` count.

Two stricter coverage checks are intentional: an all-unsupported compiled
replay, or a fixture switching between supported and unsupported, returns 3.
No compiled parser behavior was exercised in the former case; changed support
cannot establish a like-for-like revision comparison in the latter. The Python
path retains its existing policy. Likewise, compiled `--json` refuses to replace
an existing export; the Python path's existing overwrite behavior is unchanged.

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

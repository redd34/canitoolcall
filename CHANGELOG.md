# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- `canitoolcall lookup NAME [--engine E] [--matrix PATH_OR_URL] [--json]` answers "does family or model X parse on engine Y?" from a `matrix.json` (stdlib HTTP, with a timeout and a 20 MB cap; unknown names, unreachable sources, unsupported schema versions and malformed files exit 2). Everything read from the file is validated (status values, page paths that stay under the site) and control characters are escaped before printing. A model name reports that model's own results where the matrix has them, and otherwise says the row covers the whole family. Each engine's newest version is the one with the latest `run_at` (versions may be commit hashes); `newest_cell()` exposes that choice. Page links resolve against the matrix's URL or, for a local file, its directory. The exit code is 0 even when the status is `fail`; scripts should use `--json`. The default source (the published matrix) works once the site is rebuilt by a release containing this change: the live file predates `schema_version` and is rejected with an explanation until then; use `--matrix` with a newer file meanwhile.
- `matrix.json` now has a versioned schema (`spec/matrix.schema.json`, also published at its `$id` by the site build, `schema_version` "1.0", rules in `spec/README.md`) an optional `family_info` map of family names and model repos, and optional per-model results in each cell (`models`, attributed by each fixture's `models`). Existing fields are unchanged.
- `canitoolcall matrix --site-url URL`: canonical and Open Graph links (with a 1200×630 preview image) on every page and a `sitemap.xml`. Without the flag the site stays location-independent. The nightly build passes the Pages URL.
- `canitoolcall mcp` (issue #29): a stdio MCP server behind the optional extra `canitoolcall[mcp]` (official `mcp` SDK, pinned `==2.3.0`; the base install is unchanged). Tools: `lookup`, `list_families`, `list_engines` (read the `matrix.json` chosen with `--matrix`, never by the caller) and `probe`. The operator decides where `probe` may go: loopback only by default with no API key, `--allow-base-url URL` for more targets and `--allow-api-key-env VAR=URL` to bind a key to one exact URL. Other hosts are resolved and vetted at connect time (private, link-local, CGNAT, multicast, metadata addresses refused), redirects are not followed, and refusals are generic. The key is never logged or returned.
- Two Mistral v3 derived-cut fixtures that stop after the last call's closing brace, before the list's closing `]` (`fixtures/mistral/derived-truncated.jsonl`). Every call is complete, so a correct parser keeps them (recovery contract section 2, #43). The notes of derived cuts with no unfinished call now state that rule instead of "the unfinished call must not be returned".

### Changed

- Matrix site: every page has a meta description and preview tags, the index page title names what the site is, and the index says it is not a ranking (unsupported cells and policy fails).
- README: the hero image is the 2026-11-15 matrix, and there are lines on who the project is for and what it does not cover yet.
- The verifiers (`scripts/verify_upstream_pr.py`, `scripts/verify_compiled_pr.py`) now print the `unsupported` count on the `fixtures replayed` line, e.g. `fixtures replayed: 48 (N unsupported)`, and their JSON reports gain an `unsupported` field.

## [0.2.0] - 2026-11-15

Fixture spec 0.2, a recovery contract for malformed and truncated output, 31 new fixtures (500 in total) and a
fresh results snapshot. Release notes: [`docs/release-notes/v0.2.0.md`](docs/release-notes/v0.2.0.md).

### Added

- Recovery contract for malformed and truncated output (`docs/formats/recovery.md`, issue #36): recover complete calls, never return unfinished ones, compare modes on calls and exceptions for `expected_error` fixtures, count parser exceptions as outcomes.
- Fixture fields `assumes` and `finish` (#9, PR #38): record optional replay assumptions and generation finish metadata; scope stop-token-ignored fixtures before parsing and keep unsupported out of scoring checks.
- 31 derived-cut fixtures (`fixtures/<family>/derived-truncated.jsonl`, all nine families): token prefixes of complete fixtures, cut at structural boundaries and labelled by section 2 of the recovery contract ([`docs/formats/recovery.md`](docs/formats/recovery.md#2-truncated-before-the-outer-close-max_tokens), issue #36). `scripts/fixtures/derived_truncation.py` regenerates them.
- Compiled upstream-PR verification for Ollama and llama.cpp: isolated full-source builds, configurable build timeouts, allowlisted child environments, revision-checked adapters, owned-process cleanup and complete-file reports. Existing pins and exports are preserved; fixture error outcomes and summary counts follow the Python verifier.
- `AGENTS.md` and `llms.txt` for AI coding agents, an "Agent check-in" issue form for sharing goals and needs, and an AI-assistance field in the PR template.
- The Pages site now also serves `llms.txt` and `AGENTS.md` at its root.
- `scripts/verify_upstream_pr.py`: verify an upstream parser fix PR (vLLM, SGLang, transformers) by swapping its changed files into the pinned engine, replaying the fixtures at the PR's base and head, and diffing per fixture.
- The matrix site footer now links to the source repository, a "report a problem"
  issue and `CONTRIBUTING.md`, so a visitor landing on the published site has a
  one-click way to the code and to report a wrong cell (issue #1).
- Results snapshot `results/2026-11-15/` (nightly run 43690158071, linux-x86_64, 500 fixtures) with triage regenerated under the recovery contract (#36): two #13 findings retired, two refiled as `engine_bug`, and new findings for missing inner closes, raises on fixtures that do not accept `exception`, and derived cuts that drop a complete call or return its tail as content. 15 of the 16 predicted `soft_pass` cells confirmed. README and `llms.txt` now point at this snapshot.

### Changed

- Fixture spec 0.2: `spec_version` is now `"0.2"` for the whole corpus, because #38 added the optional `assumes` and `finish` fields (a minor bump under the spec's versioning rule). The schemas accept any `0.x`, so 0.1 fixtures still validate. All 500 fixtures and 9 `family.json` files were regenerated by their generators; nothing else in them changed.
- `stream_equals_nonstream` and `split_invariance` give `soft_pass`, not `fail`, when two parses of an `expected_error` fixture differ only in recovered text (same tool calls and exception type), per the recovery contract (#36). Calls or raises that differ between modes still fail.
- `llama/l4-truncated-missing-close-bracket` now expects its first call: a call closed by its own `)` is complete even when the list never closes (recovery contract section 2). It previously accepted no calls.
- `WISHLIST.md`: a list of wanted work with sizes and acceptance criteria that humans and agents can pick up, linked from the README, `AGENTS.md` and `llms.txt`.
- Design records for main-branch engine columns (`docs/DESIGN-main-branch-columns.md`, issue #25) and a provider-observation pilot (`docs/DESIGN-provider-matrix.md`, issue #27). Both are proposals; neither is approved or built.
- Triage: the nine fixtures of `vllm-stream-nonstream-other` are split into five findings (issue #10, PR #13).
- `AGENTS.md`: notes on using WSL2 for the engine scripts on Windows, GPU hosts, CI approval on a first PR, and reporting the `unsupported` count with a verify-upstream-fix result.

### Fixed

- `scripts/verify_upstream_pr.py` no longer reports "nothing regressed" for an unusable comparison: a replay that exits with anything but 0 or 1, a missing, unreadable, empty or malformed report, duplicate fixture IDs, or base and head reports with different fixture inventories now exit 3 with a diagnostic (issue #30, PR #31).
- SGLang replay on a host with a visible CUDA GPU: `sglang.sh` failed its import smoke test with `ModuleNotFoundError: No module named 'flashinfer'`, because SGLang takes CUDA-only imports whenever torch sees a GPU. `canitoolcall run` now starts SGLang workers with `CUDA_VISIBLE_DEVICES=-1` (an explicit `--env CUDA_VISIBLE_DEVICES=...` overrides it), and `sglang.sh` hides the GPUs for its smoke test. `scripts/verify_upstream_pr.py` gains `--env KEY=VALUE`. Not yet exercised on a real GPU host (issue #19).
- Tests on native Windows: `test_worker_env` split `PYTHONPATH` on `:` instead of `os.pathsep`; the file-mode assertion and the four tests that run a `#!`-script fake llama.cpp harness are skipped on Windows, as they are POSIX-only (PR #32).

## [0.1.1] - 2026-09-26

### Fixed

- `canitoolcall probe` no longer fails a scenario because `reasoning_content` contains tool-call
  markers such as `<tool_call>`. Models often draft their call while thinking, and the probe
  cannot know the expected reasoning, so this is now a warning (`pass*`) that does not change
  the exit code. Reasoning delimiters and channel markup (`<think>`, `</think>`, `[THINK]`,
  Harmony's `<|channel|>`, `<|message|>`, `<|start|>`, `<|end|>`, `<|return|>`, …) in
  `reasoning_content` still fail, because the reasoning parser must strip them. Markers in
  `content`, tool names or tool arguments still fail, and a missing or wrong call still fails,
  with the reasoning markers mentioned as a hint. Found with qwen3:4b on Ollama, where
  `parallel-calls` returned both calls correctly but was reported as a failure. The offline
  suite's `no_leakage` check is unchanged.

### Changed

- README: a one-line `uvx` quickstart, badges, a per-engine results table, links to the issues
  reported upstream, three Mermaid diagrams (PyPI shows a link to the rendered versions), and
  an FAQ.

## [0.1.0] - 2026-09-26

### Added

- Fixture spec v0.1 (`spec/`), 469 fixtures across nine model families, each with provenance.
- Offline replay through the pinned parsers of vLLM 0.30.0, SGLang 0.5.20, llama.cpp `a25c9865`,
  Ollama `7af39318` and HF transformers 5.17.0, with eight checks and seeded token-level chunking.
- Per-engine streaming granularity: multi-token chunkings are reported as synthetic for engines
  whose servers stream one token per event (`run.synthetic_strategies`).
- `canitoolcall probe` for live OpenAI-compatible endpoints, a pytest plugin, and the static matrix site.
- A committed results snapshot (`results/2026-09-25/`) with every failure triaged.

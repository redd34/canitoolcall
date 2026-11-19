# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Changed

- HF transformers pin bumped from 5.17.0 to 5.23.0 (`scripts/engines/transformers.sh`, the `transformers` extra, which now needs Python >= 3.11, adapter, docs). `parse_response` and the response-template format are unchanged apart from small argument-coercion fixes (`oneOf`, whole-number floats for integers, already-decoded arrays, copied defaults: transformers#48719, #49547, #49062, #49370); the Gemma 4 replays are byte-identical to 5.17.0. `serve` now passes `tools` to the parser (transformers#50385), as the adapter already did, and reads reasoning from `reasoning_content`, `thinking` or `reasoning` (transformers#49535); the adapter now does the same in non-streaming. New coverage: `Qwen/Qwen3.8-27B` ships a `response_template` since Hub commit `3756181` (2026-11-07), so its 24 `qwen3-xml` fixtures are replayed (local run: 20 `soft_pass`, 4 `fail`; they were `unsupported`). The template is read at `3756181` and the tokenizer at the fixtures' pin `1d4bf0f`, whose tokenizer files are identical. No other repo the fixtures list ships a template. The committed snapshot is unchanged; the nightly regenerates it.

## [0.3.0] - 2026-11-19

Ask the matrix without a checkout (`lookup`, a versioned `matrix.json`, an MCP server, badges), probe the Responses API
and run the probe as a GitHub Action, two new families (MiniMax M2.x, Seed-OSS; 555 fixtures in total), an mlx-lm adapter
and the vLLM pin at 0.34.0. The committed results snapshot is unchanged. Release notes: [`docs/release-notes/v0.3.0.md`](docs/release-notes/v0.3.0.md).

### Added

- `canitoolcall lookup NAME [--engine E] [--matrix PATH_OR_URL] [--json]` answers "does family or model X parse on engine Y?" from a `matrix.json` (stdlib HTTP, with a timeout and a 20 MB cap; unknown names, unreachable sources, unsupported schema versions and malformed files exit 2). Everything read from the file is validated (status values, page paths that stay under the site) and control characters are escaped before printing. A model name reports that model's own results where the matrix has them, and otherwise says the row covers the whole family. Each engine's newest version is the one with the latest `run_at` (versions may be commit hashes); `newest_cell()` exposes that choice. Page links resolve against the matrix's URL or, for a local file, its directory. The exit code is 0 even when the status is `fail`; scripts should use `--json`. The default source (the published matrix) works once the site is rebuilt by a release containing this change: the live file predates `schema_version` and is rejected with an explanation until then; use `--matrix` with a newer file meanwhile.
- `matrix.json` now has a versioned schema (`spec/matrix.schema.json`, also published at its `$id` by the site build, `schema_version` "1.0", rules in `spec/README.md`) an optional `family_info` map of family names and model repos, and optional per-model results in each cell (`models`, attributed by each fixture's `models`). Existing fields are unchanged.
- `canitoolcall mcp` (issue #29): a stdio MCP server behind the optional extra `canitoolcall[mcp]` (official `mcp` SDK, pinned `==2.3.0`; the base install is unchanged). Tools: `lookup`, `list_families`, `list_engines` (read the `matrix.json` chosen with `--matrix`, never by the caller) and `probe`. The operator decides where `probe` may go: loopback only by default with no API key, `--allow-base-url URL` for more targets and `--allow-api-key-env VAR=URL` to bind a key to one exact URL. Other hosts are resolved and vetted at connect time (private, link-local, CGNAT, multicast, metadata addresses refused), redirects are not followed, and refusals are generic. The key is never logged or returned.
- `canitoolcall matrix` now also writes shields.io endpoint badges (`badges/<engine>/<family>.json` per cell and `badges/<engine>.json` per engine, newest version of each engine (latest `run_at`, as in `lookup`), derived from the same data as `matrix.json`; messages read like `fail · 23/54 strict pass`, stale badge files are cleared on each build, layout documented in `spec/README.md`), and cell pages show the Markdown snippets in a "Badges" section (badge URLs use `--site-url` when given, else the published site).
- `canitoolcall matrix --site-url URL`: canonical and Open Graph links (with a 1200×630 preview image) on every page and a `sitemap.xml`. Without the flag the site stays location-independent. The nightly build passes the Pages URL.
- `probe --api {chat,responses,both}`: probe the OpenAI Responses API (`/v1/responses`) with the same scenarios and judgments, non-streaming and streaming (`response.*` SSE events, with `response.completed`, `.done` and `output_item.done` checked against the deltas). The default stays `chat`, so existing runs and exit codes are unchanged. Report outcomes gain an `api` field. Sources: [`docs/probe-responses-api.md`](docs/probe-responses-api.md).
- GitHub Action (`action.yml`, related to #28): `uses: redd34/canitoolcall@v0` runs `canitoolcall probe` with the pinned release via `uvx`, writes a results table to the job summary (`scripts/probe_summary.py`), uploads the JSON report as an artifact and fails the job with the probe's exit code (also shown in the summary heading). A file already at the report path is removed first, so a run that writes no report never shows an old one. The key is passed by variable name only. Docs and an Ollama example (untested) in [`docs/github-action.md`](docs/github-action.md); `.github/workflows/action-selftest.yml` runs it against the probe tests' mock server, which `scripts/probe_mock_server.py` now starts on its own. From 0.3.0 on, the moving `v0` tag points at the latest 0.x release, and the action's `version` input defaults to that release.
- New model family `minimax` (MiniMax M2, M2.1, M2.5, M2.7; issue #4): format notes (`docs/formats/minimax.md`), a generator that renders through the pinned MiniMax-M2.1 chat template (modified MIT; M2, M2.5 and M2.7 are checked to render identically) (`scripts/fixtures/minimax/build.py`), 23 fixtures, and parser mappings for vLLM (`minimax_m2`) and SGLang (`minimax-m2`). llama.cpp, Ollama and transformers report the family `unsupported` until it is replayed there. MiniMax-M3 is not included.
- New family `seed-oss` (ByteDance Seed-OSS-36B-Instruct): format notes `docs/formats/seed-oss.md`, a generator `scripts/fixtures/seed-oss/build.py` and 30 fixtures (20 template renders at rev `497f1dc`, 6 token-prefix truncations, 4 llama.cpp `test-chat.cpp` imports). Parser mappings for vLLM (`seed_oss` tool and reasoning parsers) and llama.cpp (autoparser); SGLang, Ollama and transformers have no Seed-OSS parser at their pins and report `unsupported`.
- Two Mistral v3 derived-cut fixtures that stop after the last call's closing brace, before the list's closing `]` (`fixtures/mistral/derived-truncated.jsonl`). Every call is complete, so a correct parser keeps them (recovery contract section 2, #43). (PR #47).
- New engine adapter `mlx_lm` for mlx-lm 0.32.2 (`scripts/engines/mlx_lm.sh`, `.venvs/mlx_lm`). It runs `mlx_lm.server`'s own `/v1/chat/completions` handler with the tool parser `load_tokenizer` selects for each repo: `json_tools` (qwen3-hermes), `qwen3_coder` (qwen3-xml), `glm47` (glm), `gemma4`, `kimi_k2`/`kimi_k3` (kimi) and `mistral` (Mistral repos that ship an HF chat template). gpt-oss, DeepSeek and Llama are `unsupported`. The server streams one token per event, so only `token` counts. The env needs mlx: macOS arm64, or Linux through the `mlx-cpu` wheels. Where mlx cannot be installed or imported, the setup script exits 3 and the nightly skips the engine. It is not in the committed results snapshot.

### Changed

- vLLM pin bumped from 0.30.0 to 0.34.0 (`scripts/engines/vllm.sh`, adapter, docs). The parser names for all nine families are unchanged; `llama3_json`/`llama4_json` now resolve to the parser-engine `LlamaJsonEngineToolParser` (vllm#63569). The adapter follows 0.34's serving path: `get_parser(tokenizer=...)`, `set_prompt_token_ids` before `adjust_request` and before the non-streaming `parse`, `always_adjust_request`, the `DeepseekV41Renderer`, and Kimi K3 thinking efforts. The vLLM env moves to `oss-harmony` (vllm#55128), transformers 5.22.0 and huggingface-hub 2.4.0 to meet 0.34's floors, and adds `einops`. The committed snapshot is unchanged; the nightly regenerates it.
- Matrix site: every page has a meta description and preview tags, the index page title names what the site is, and the index says it is not a ranking (unsupported cells and policy fails).
- README: the hero image is the 2026-11-15 matrix, and there are lines on who the project is for and what it does not cover yet.
- Both upstream-PR verifiers now validate replay reports with one shared module, `scripts/replay_report.py`: exactly one complete results file from a replay that exited 0 or 1, valid unique fixture IDs and statuses, and matching base/head inventories, otherwise exit 3. Exit codes are unchanged; the compiled route keeps its stricter coverage checks.
- Triage (#23): the last `transformers-other` case, `gemma4/sglang-text-around-call`, is now `transformers-gemma4-content-overwritten` (`engine_bug`): Gemma 4's `response_template` declares `content` without `repeats`/`join`, so in non-streaming the text after a call replaces the text before it. No transformers case is left untriaged.
- The verifiers (`scripts/verify_upstream_pr.py`, `scripts/verify_compiled_pr.py`) now print the `unsupported` count on the `fixtures replayed` line, e.g. `fixtures replayed: 48 (N unsupported)`, and their JSON reports gain an `unsupported` field.

### Fixed

- The notes of three derived cuts from 0.2.0 that keep every call (`deepseek/vllm-v3-single-call-cut-before-section-close`, `kimi/k2i-single-call-cut-before-section-close`, `kimi/k3-vllm-call-thinking-disabled-cut-before-tools-close`) said "the unfinished call must not be returned", though they have no unfinished call. The generator now writes the recovery contract section 2 rule for any cut that keeps every call; only the `notes` field changed (PR #47).

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

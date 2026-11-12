# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- Compiled upstream-PR verification for Ollama and llama.cpp: isolated full-source builds, configurable build timeouts, allowlisted child environments, revision-checked adapters, owned-process cleanup and complete-file reports. Existing pins and exports are preserved; fixture error outcomes and summary counts follow the Python verifier.

- `AGENTS.md` and `llms.txt` for AI coding agents, an "Agent check-in" issue form for sharing goals and needs, and an AI-assistance field in the PR template.
- The Pages site now also serves `llms.txt` and `AGENTS.md` at its root.
- `scripts/verify_upstream_pr.py`: verify an upstream parser fix PR (vLLM, SGLang, transformers) by swapping its changed files into the pinned engine, replaying the fixtures at the PR's base and head, and diffing per fixture.
- The matrix site footer now links to the source repository, a "report a problem"
  issue and `CONTRIBUTING.md`, so a visitor landing on the published site has a
  one-click way to the code and to report a wrong cell (issue #1).

### Fixed

- `scripts/verify_upstream_pr.py` no longer reports "nothing regressed" for an unusable comparison: a replay that exits with anything but 0 or 1, a missing, unreadable, empty or malformed report, duplicate fixture IDs, or base and head reports with different fixture inventories now exit 3 with a diagnostic (issue #30, PR #31).
- SGLang replay on a host with a visible CUDA GPU: `sglang.sh` failed its import smoke test with `ModuleNotFoundError: No module named 'flashinfer'`, because SGLang takes CUDA-only imports whenever torch sees a GPU. `canitoolcall run` now starts SGLang workers with `CUDA_VISIBLE_DEVICES=-1` (an explicit `--env CUDA_VISIBLE_DEVICES=...` overrides it), and `sglang.sh` hides the GPUs for its smoke test. `scripts/verify_upstream_pr.py` gains `--env KEY=VALUE`. Not yet exercised on a real GPU host (issue #19).
- Tests on native Windows: `test_worker_env` split `PYTHONPATH` on `:` instead of `os.pathsep`; the file-mode assertion and the four tests that run a `#!`-script fake llama.cpp harness are skipped on Windows, as they are POSIX-only (PR #32).

### Changed

- `WISHLIST.md`: a list of wanted work with sizes and acceptance criteria that humans and agents can pick up, linked from the README, `AGENTS.md` and `llms.txt`.
- Design records for main-branch engine columns (`docs/DESIGN-main-branch-columns.md`, issue #25) and a provider-observation pilot (`docs/DESIGN-provider-matrix.md`, issue #27). Both are proposals; neither is approved or built.
- Triage: the nine fixtures of `vllm-stream-nonstream-other` are split into five findings (issue #10, PR #13).
- `AGENTS.md`: notes on using WSL2 for the engine scripts on Windows, GPU hosts, CI approval on a first PR, and reporting the `unsupported` count with a verify-upstream-fix result.

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

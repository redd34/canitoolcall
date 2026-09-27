# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- `AGENTS.md` and `llms.txt` for AI coding agents, an "Agent check-in" issue form for sharing goals and needs, and an AI-assistance field in the PR template.
- `scripts/verify_upstream_pr.py`: verify an upstream parser fix PR (vLLM, SGLang, transformers) by swapping its changed files into the pinned engine, replaying the fixtures at the PR's base and head, and diffing per fixture.
- The matrix site footer now links to the source repository, a "report a problem"
  issue and `CONTRIBUTING.md`, so a visitor landing on the published site has a
  one-click way to the code and to report a wrong cell (issue #1).

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

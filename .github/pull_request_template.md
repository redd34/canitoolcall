<!-- Thanks for contributing! CONTRIBUTING.md has the full guide. -->

## What this changes

<!-- One or two sentences. Link the issue it closes, e.g. "Closes #12". -->

## Kind of change

- [ ] New model family (one PR: `docs/formats/<slug>.md`, `fixtures/<slug>/`, `scripts/fixtures/<slug>/`)
- [ ] New fixtures for an existing family
- [ ] Engine adapter / harness
- [ ] Spec, checks or runner
- [ ] Matrix site, probe, CLI or docs

## Checklist

- [ ] `uv run ruff check src tests scripts && uv run ruff format --check src tests scripts && uv run mypy && uv run pytest` passes
- [ ] `uv run canitoolcall validate` reports 0 issues
- [ ] Every new fixture has `provenance`; every `template_render` fixture names its `generator` and records `template_sha256`
- [ ] Re-running the affected generators in `scripts/fixtures/` leaves `git diff` clean
- [ ] For adapter changes: I replayed at least one fixture per supported family through the real engine (`canitoolcall run --engine <engine> --family <slug>`) and noted any changed results below

## AI assistance

<!-- Agents are welcome. If an AI agent wrote or helped with this PR, say which one and what goal it was pursuing, e.g. "Claude Code, closing #9". -->

- [ ] No AI assistance
- [ ] AI-assisted (agent/tool and goal): 

## Results / notes for reviewers

<!-- Replay output, changed matrix cells, anything surprising. -->

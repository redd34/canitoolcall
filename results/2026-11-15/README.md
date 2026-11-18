# Results snapshot, 2026-11-15

Real offline replays of the whole fixture corpus (500 fixtures, 9 families, `fixtures_digest` `fcdf55ce…`) through five engines' own parser code, at commit `aa13524` (after #42 and #43). Nothing here is hand-written except the triage selectors in `triage.py`.

These are the results artifacts of the `nightly` workflow, [run 43690158071](https://github.com/redd34/canitoolcall/actions/runs/43690158071) (manual dispatch, 2026-11-15), gzipped unchanged (`gzip -9 -n`). The runs were on GitHub's `linux-x86_64` runners, with engine workers on Python 3.12.16 (Ollama's harness on 3.12.3). The previous snapshot, [`2026-09-25`](../2026-09-25/README.md), was run on macOS arm64. Pass counts can differ between platforms (see `ollama-qwen3coder-int64-clamp`).

| File | Engine (pin) | pass | soft_pass | fail | error | unsupported |
|---|---|---|---|---|---|---|
| `vllm-0.30.0.json.gz` | vLLM 0.30.0 | 270 | 93 | 135 | 0 | 2 |
| `sglang-0.5.20.json.gz` | SGLang 0.5.20 | 200 | 70 | 206 | 0 | 24 |
| `llamacpp-a25c9865.json.gz` | llama.cpp `a25c9865` | 287 | 53 | 124 | 0 | 36 |
| `ollama-7af39318.json.gz` | Ollama `7af39318` | 234 | 13 | 26 | 0 | 227 |
| `transformers-5.17.0.json.gz` | transformers 5.17.0 | 34 | 7 | 10 | 0 | 449 |

The nightly workflow runs `uv run canitoolcall run --engine <engine> --out results --env HF_HUB_OFFLINE=1`, which uses the default strategies (`one`, `special`, `token`, `rand:1..5:8`). It leaves out the `char:0` and `char:1` strategies the 2026-09-25 snapshot ran; those never count toward a status. llama-server, Ollama and transformers `serve` stream one token per event, so for those three only `token` (plus the non-streaming parse) counts. The others are recorded in `run.synthetic_strategies`.

`canitoolcall matrix --results results/2026-11-15` reads the files directly.

## Changes since 2026-09-25

- **Recovery contract** ([`docs/formats/recovery.md`](../../docs/formats/recovery.md), #36). Section 3 (#41) moved the 16 predicted cells from `fail`; 15 are now `soft_pass`. The 16th, SGLang `llama/l4-truncated-missing-close-bracket`, was relabelled by #43 and now fails as `sglang-truncated-drops-complete-call`. The #13 findings `vllm-harmony-recovery-newline` and `vllm-malformed-content-recovery-contract` are retired. `vllm-harmony-malformed-header-rejection` and `vllm-deepseek-v3-malformed-call-fragments` are refiled as `engine_bug` under sections 4 and 1. The missing-inner-close cases left in the catch-alls get their own findings (`sglang-` and `llamacpp-missing-close-corrupts-argument`, `ollama-missing-close-rejected`), as do raises on fixtures that do not accept `exception` (`ollama-malformed-json-raises`, `transformers-gemma4-malformed-call-raises`).
- **Derived-cut fixtures** (#42, 31 fixtures). Cuts that return the unfinished call are `*-truncated-call-returned` (`truncation_policy`), which now also covers `expected` fixtures where some mode returns more calls than expected. Cuts after a complete call where non-streaming drops that call are `*-truncated-drops-complete-call`, and a kept call with the tail returned as content is `sglang-truncated-tail-as-content`, both `engine_bug` under section 2. Each `*-truncated-drops-complete-call` finding sits just before its engine's catch-all, so a more specific root cause (for example `llamacpp-deepseek-v3-preserved-tokens`) wins.
- No vLLM case is left untriaged. 28 cases remained `untriaged_discrepancy` at release (SGLang 14, llama.cpp 7, Ollama 6, transformers 1). None are from #42 or #43.
- Since release (#23): the transformers case, `gemma4/sglang-text-around-call`, is now `transformers-gemma4-content-overwritten` (`engine_bug`). 27 cases remain `untriaged_discrepancy` (SGLang 14, llama.cpp 7, Ollama 6).

## Triage

`triage.jsonl` assigns every failing case, per engine, to exactly one finding. `tests/core/test_results_snapshot.py` checks this against the results files. `triage.py` regenerates it. The classifications are:

| Classification | Meaning |
|---|---|
| `engine_bug` | The engine's parser returns a wrong parse for output the model can produce. One finding can cover many fixtures: count findings, not fixtures (for example `sglang-gpt-oss-role-header-recipient` is one gap across every `x-recipient-in-role` fixture). |
| `engine_bug_known_upstream` | The same, and already reported upstream (see `upstream`). |
| `engine_bug_candidate` | Reproduced through the engine's own code path, but not yet confirmed against a live server. |
| `truncation_policy` | Output cut by `max_tokens` comes back as a (partial) tool call, which the spec treats as a failure (spec/README.md, "Expected results"). |
| `strict_grammar_variant` | The engine rejects an engine-test variant that the official template never renders. |
| `not_generated_by_model` | The input is not what the model generates, for example `<tool_call>` without the newline Qwen3 emits. |
| `format_unverified` | Whether the model generates this form is unverified, for example Mistral v11 `[CALL_ID]` (mistral-common's generation grammar omits it, llama.cpp's tests assume it). Not counted as a bug until a recorded generation settles it. |
| `not_applicable_stop_applied` | The fixture contains an interior stop token that a server applying the stop token never passes on. |
| `intended_engine_behaviour` | The engine's own test asserts this behaviour; the fixture disagrees. |
| `uncertain_history_render` / `contested_fixture` | The expected parse depends on a history render or on a disputed format. Not counted as a bug. |
| `untriaged_discrepancy` | A failing case that is not yet assigned to a root cause. |

Every finding has a `repro` command that replays one of its fixtures through the harness. Four findings also have `direct_repro` scripts in `repro/`, which call the engine's own API with no canitoolcall code. Run each one with that engine's interpreter, for example `HF_HUB_OFFLINE=1 .venvs/sglang/bin/python results/2026-11-15/repro/sglang_qwen25_one_delta.py`.

Upstream reports filed by this project are listed in the main [README](../../README.md#reported-upstream); since then it has also posted a patch on ollama/ollama#18663 (#37). Findings classified `engine_bug_known_upstream` match an issue someone else already filed, linked in `upstream`. Draft reports for the other findings are in `RELEASE_CHECKLIST.md` §7. Re-check each finding against the engine's latest release before filing it.

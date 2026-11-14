# CanIToolCall design

This document is the contract between the people (and agents) building CanIToolCall in parallel. It covers the architecture, the interfaces, the faithfulness rules found by the engine spike, and **exactly which files each builder owns**. The plan is in `../docs/plans/01-canitoolcall.md` (outside this repo); the fixture format is in [`spec/README.md`](../spec/README.md); per-family format notes are in [`docs/formats/`](formats/README.md).

## 1. What the tool does

CanIToolCall replays recorded raw model outputs (**fixtures**) through each inference engine's **own parser code**, offline, with no GPU and no model weights. Each fixture is parsed once without streaming and once per chunking strategy. The results are checked against the expected parse and against each other. Results are written as JSON and rendered into a static family × engine matrix.

An adapter counts only if it runs the engine's genuine parser code at a pinned version. If something cannot run, it is reported as `unsupported`, never guessed.

## 2. Architecture

```
fixtures/<family>/*.jsonl ──► fixtures.py (load + validate)
                                   │
canitoolcall run --engine E ──► runner.py ── WorkerClient ──(JSON lines over stdin/stdout)──┐
   (main dev env, .venv)           │                                                        │
                                   │                   .venvs/E/bin/python -m canitoolcall.adapters.worker E
                                   │                          │   (PYTHONPATH=<repo>/src, stdlib-only core)
                                   │                          ▼
                                   │                   adapters/E.py (Adapter) ──► engine parser code
                                   │                          │     (vLLM/SGLang/transformers: in-process Python;
                                   │                          │      llama.cpp/Ollama: compiled harness subprocess)
                                   ▼                          │
                          checks.py ◄── Observation (nonstream + one ParseResult per strategy)
                                   │
                          results.py ── results/E-<version>.json ──► matrix.py ──► site/_build/index.html
```

### Process model and import rules

- **The main env** (`uv sync`, Python 3.12, `.venv/`) runs the CLI, the runner, the checks, validation, probe and matrix. It never imports an engine.
- **Each engine runs in its own isolated interpreter**, `.venvs/<engine>/bin/python`, which is built by `scripts/engines/<engine>.sh`. Override it with `$CANITOOLCALL_<ENGINE>_PYTHON`. The runner starts `python -m canitoolcall.adapters.worker <engine>` there with `PYTHONPATH=<repo>/src`, so canitoolcall is **not installed** into engine venvs.
- Therefore these modules **must import with the standard library only**: `canitoolcall/__init__.py`, `fixtures.py` (jsonschema is imported lazily inside `validate`), `chunking.py`, `results.py`, `adapters/__init__.py`, `adapters/base.py`, `adapters/worker.py` and every `adapters/<engine>.py` at module level. Engine imports go inside methods. `tests/core/test_adapters_registry.py::test_worker_starts_in_engine_env` enforces this.
- **Worker protocol v1** (see `adapters/worker.py` for the full text): `hello`, `replay` (fixture record + family + strategy ids → `nonstream` + `streams{id: ParseResult}` + `parser_config`), and `shutdown`. One worker process serves a whole run. A harness failure answers `{"ok": false, "error": traceback}`, and the worker keeps serving.
- The compiled harnesses (llama.cpp C++, Ollama Go) are separate long-lived processes, owned by their Python adapter, and speak **JSON lines** as well: one process per run, never one per fixture.

### Why fixtures are token ids first

Engines stream **tokens**, and special tokens are atomic. The spike showed that per-character splits inside a special token give failures no real server can produce, and that re-encoding text is lossy (`mistral_common` never maps the text `[TOOL_CALLS]` to its control token). So:

1. Fixtures carry `output_token_ids` plus a `tokenizer` pin whenever the source allows it (spec/README.md, "Token ids come first").
2. `Adapter.units(raw)` returns those ids (or the engine tokenizer's encoding of `raw_output` as a fallback).
3. `chunking.split(units, strategy, special=...)` groups the ids. The default strategies are `one`, `special` (split at special-token boundaries, using `Adapter.special_token_ids`), `token` and `rand:1..5:8`, all seeded and identical on every machine. If an adapter returns `None` from `special_token_ids`, the worker records `special` in the case's `skipped_strategies` instead of guessing.
4. The adapter turns each id group into a text delta with **the engine's own detokenizer**, then calls the engine's streaming parser.

`char:<seed>` exists as an opt-in stress strategy. It cannot be expressed as token groups, so the worker runs it only through an adapter's optional text path (`supports_text_deltas = True` plus `parse_stream_text`), and otherwise records it as skipped. Its rows are reported but never count toward the case or matrix status (`checks.case_status`).

## 3. Interfaces (core-owned)

The signatures below are frozen for the parallel build. If you need a change, don't edit core files: describe the change in your final report, and the core owner applies it.

| Module | Key types / functions | State in skeleton |
|---|---|---|
| `fixtures.py` | `Fixture`, `Family`, `ReferenceModel`, `Provenance`, `Expected`, `ExpectedError`, `load_fixtures`, `load_families`, `validate`, `fixtures_digest`, `default_fixtures_dir` | **implemented** + tested |
| `chunking.py` | `ChunkStrategy` (`id`, `parse`, `realistic`), `DEFAULT_STRATEGIES`, `group_sizes`, `special_group_sizes`, `split`, `split_text`, `parse_strategies` | **implemented** + tested (golden grouping pinned) |
| `results.py` | `Status`, `worst_status`, `ParsedToolCall`, `ParseResult`, `Observation`, `CheckResult`, `CaseResult` (+ optional `reason`, `skipped_strategies`), `EngineInfo`, `RunInfo`, `RunResults` (`to_dict`/`from_dict`/atomic `write`/`load`) | **implemented** + tested against `spec/results.schema.json` |
| `adapters/base.py` | `Adapter` ABC (`name`, `pinned_version`, `version`, `supports`, `parser_config`, `units`, `parse`, `parse_stream`, `engine_details`, `commit`, `close`; optional `special_token_ids`, `supports_text_deltas` + `parse_stream_text`), `ReplayInput`, `Support`, `AdapterUnavailable` | **implemented** |
| `adapters/__init__.py` | `ENGINES` registry, `adapter_class` (registry name or `package.module:Class` spec), `is_adapter_spec`, `load_adapter`, `engine_python` | **implemented** |
| `adapters/worker.py` | protocol v1: `hello` (+ `pinned_version`, `python`), `replay` (+ `skipped`), `handle`, `serve`, `main` (protocol on a private dup of fd 1) | **implemented** + tested with fake and reference adapters |
| `checks.py` | `normalize_text`, `canonical`, `canonical_arguments`, `compare`, `describe_difference`, `expected_as_result`, `is_synthetic`, `case_status`, `check_*` (8 checks), `ALL_CHECKS`, `run_checks` | **implemented** + tested |
| `runner.py` | `RunConfig` (+ `tags`, `ids`, `jobs`, `validate`), `WorkerClient`, `WorkerError` (`fatal`), `FixtureValidationError`, `evaluate`, `replay_case`, `run`, `run_and_write` | **implemented** + tested end to end with the reference adapter |
| `cli.py` | `run` (`--tag`, `--id`, `--jobs`, `--env`, `--no-validate`), `probe`, `matrix`, `validate`, `engines` | **implemented** |
| `probe.py` | `Scenario`, `ProbeExpectation`, `ProbeOutcome`, `ProbeReport`, `BUILTIN_SCENARIOS`, `chat`, `evaluate`, `probe` | stubs |
| `pytest_plugin.py` | options, `pytest_generate_tests`, `assert_conforms` | options done; rest stubs |
| `matrix.py` | `Cell`, `Matrix`, `load_results`, `build_matrix`, `render_site` | stubs |

### Adapter contract (every adapter MUST honour these)

Each rule was found by the spike **breaking without it**:

1. **Token ids first**: see §2.
2. **Engine detokenization:**
   - **vLLM:** call `parser.adjust_request(request)` first, since many parsers force `skip_special_tokens=False` there. Then detokenize with the v1 `IncrementalDetokenizer` (`from_new_request(tok, SimpleNamespace(request_id, sampling_params=req.to_sampling_params(4096, {}), prompt_token_ids, prompt_embeds=None))`, then `.update(ids, False)` and `.get_next_output_text(finished, delta=True)`). Never use `tok.decode`: vLLM's `MistralTokenizer` keeps `[TOOL_CALLS]`/`[ARGS]` even when `skip_special_tokens=True`.
   - **SGLang:** when tools are present, detokenize with `skip_special_tokens=False`.
   - **llama.cpp and Ollama:** render only the harness's `preserved_tokens` as text, from a **vocab-only GGUF** made by `convert_hf_to_gguf.py --vocab-only`, using `common_token_to_piece(vocab, id, special = id ∈ preserved_tokens)`. **Never** approximate GGUF CONTROL tokens with HF `special` flags: that is wrong for Gemma 4's `<|"|>`.
3. **Serving-layer entry points:**
   - **vLLM:** `ParserManager.get_parser(...)` gives the parser class. Non-streaming: `p.parse(text, request, enable_auto_tools=True, model_output_token_ids=ids)`. Streaming: `p.parse_delta(delta_text, delta_ids, request, prompt_token_ids=prompt_ids, finished=…)`. **`prompt_token_ids` is required**; without it `reasoning_ended` is never set and calls come out as content.
   - **SGLang:** run `ReasoningParser` first, then `FunctionCallParser` on the normal text, and call `parse_stream_end` at finish.
   - **llama.cpp:** re-parse the accumulated text with `is_partial=true` after each chunk, then `compute_diffs`, exactly like `llama-server`.
   - **Ollama:** `ParserForName`, `Init` once, then `Add(chunk, done)`, as `server/routes.go` does.
4. **Prompt context.** Build `prompt_token_ids` from the reference model's chat template with the fixture's tools and `add_generation_prompt=True`. `ReplayInput.generation_prompt` covers pre-filled reasoning openers such as `<think>\n`.
5. **Stop tokens.** `raw_output` ends **before** the stop token (`ReplayInput.stop_tokens`, from `generation_config.json`). If an engine needs the stop token in the text (SGLang's gpt-oss detector needs `<|call|>`), the adapter re-appends exactly what that engine's server would keep, and records it in `parser_config`. Fixtures tagged `truncated` were cut by `max_tokens`: the model emitted no stop token, so adapters never append one for them (`ReplayInput.truncated`) and replay the finish as `finish_reason: "length"`.
6. **Engine exceptions are outcomes.** Catch them and return `ParseResult(exception="Type: msg")`. Raise only for harness problems; the runner records those as `error`.
7. **Pin everything.** `parser_config(raw)` records the tool/reasoning parser names, template source + sha256, `enable_thinking`/`reasoning_format`, tokenizer mode and repo@revision. For llama.cpp, the template **source** matters: the current HF DeepSeek-V3.1 template fails where llama.cpp's own `models/templates/` copy works.
8. **Stream accumulation** is OpenAI-client style. Concatenate content and reasoning deltas, and merge tool-call deltas by `index`: argument fragments are concatenated into `arguments_raw`. The name comes from the first delta that carries it. If an engine sends the name again in later deltas, adapters follow openai-python's `accumulate_delta`, which concatenates the fragments, so a repeated name shows up in results the way a real client would see it.
9. **`supports(family, model)` is honest and cheap.** If the engine has no parser for a model version (for example, SGLang 0.5.20 has none for DeepSeek-V4.1), return `Support(False, reason)`. The runner reports `unsupported`.
10. **Offline by default.** Tokenizers and templates are fetched once into the HF cache at pinned revisions (config and tokenizer files only, never weights). Tests set `HF_HUB_OFFLINE=1` when the cache is warm. Full runs should pass `--env HF_HUB_OFFLINE=1` too: some tokenizer paths (vLLM's Mistral mode) list repo files over the network even at a pinned revision, and bulk runs then hit Hub rate limits.

### Per-replay conditions (issue #9)

The additive nonabstract `supports_replay(raw: ReplayInput) -> Support` refines
applicability without changing the existing two-argument family/model hook.
The worker preserves a family refusal, then checks per-replay support before
requesting units, parsing, or parser configuration. Applicability never consumes
expected answers and never becomes a scored check. See the normative
[replay conditions](../spec/README.md#replay-conditions-and-generation-termination)
for optional `assumes` / `finish` fields, their validation and serialization.

In rule 5 above, `stop_tokens` is the family's **candidate** list. Explicit
`finish` identifies the actual terminal token independently; explicit `length`
forbids appending any stop. Without metadata, existing engine-specific defaults
stay unchanged. SGLang and llama.cpp currently implement explicit finish;
other adapters decline it instead of silently guessing. No adapter currently
opts into `stop_token_ignored`. Historical matrix archives are not regenerated
or reclassified by this change.

### Checks and normalization

The semantics are in spec/README.md ("Checks" and "Normalization policy soft-v1"). A case's status is the worst over its checks: `fail` > `error` > `soft_pass` > `pass`. `unsupported` is used when the adapter declines. Whitespace-only differences in content/reasoning, and leading or trailing newlines, are `soft_pass`, never `pass`, and the matrix shows them separately. The spike saw these in several engines (glm45 leading `\n`, Qwen3 hermes `"\n\n"` content, Gemma 4 trailing `\n`).

## 4. Working in the repo

- **Vocab-only GGUFs** are generated by `scripts/engines/gguf_vocab.sh` into `.engines/gguf/<org>--<model>.vocab.gguf`. The llama.cpp adapter uses them, and the Ollama adapter reads them read-only. The script needs a llama.cpp checkout for `convert_hf_to_gguf.py` (made by `scripts/engines/llamacpp.sh`).
- **Engine checkouts, unpacked wheels and harness builds** go under `.engines/<engine>/`, and engine venvs under `.venvs/<engine>/`. Both are gitignored.
- **The fixture shape** is defined by `spec/`. A new field or enum tag is a spec change; project-specific tags use `x-*` and need no spec change.
- **Family slugs** (directory names) are fixed: `qwen3-hermes`, `qwen3-xml`, `gpt-oss`, `deepseek`, `kimi`, `glm`, `llama`, `mistral`, `gemma4`. Adapters key their parser maps on these.
- **Fixture sources**, in order of preference: generator scripts in `scripts/fixtures/<slug>/` that render through the official template or encoder (`apply_chat_template(tokenize=True)`, storing `output_token_ids`); engine test suites, with the license and a line-anchored URL (vLLM/SGLang/transformers Apache-2.0, llama.cpp `tests/test-chat.cpp` and Ollama `model/parsers/*_test.go` MIT); bug reports, with the issue URL. **History render ≠ generation** (spec/README.md). **Never type a format by hand.**
- **Adapter tests** contain a fast unit test of `supports`/`parser_config` that doesn't need the engine, and `@pytest.mark.engine("<engine>")` tests that replay the core sample and at least one fixture per supported family through the real worker.
- Before sending a change, run `uv run ruff check src tests scripts && uv run ruff format --check src tests scripts && uv run mypy && uv run pytest`.

## 5. Engine setup notes

| Engine | Pin | Env | Notes |
|---|---|---|---|
| vllm | 0.30.0 | `.venvs/vllm` (py3.12; manylinux aarch64 wheel unzipped under `.engines/vllm/`, added via `.pth`; CPU torch) | `ParserManager` path; the Harmony path uses `is_harmony`; `tokenizer_mode='mistral'` for Mistral |
| sglang | 0.5.20 | `.venvs/sglang` (same unzip + `.pth` approach) | text-only parsers; the gpt-oss detector needs `<|call|>` |
| llamacpp | `a25c9865` | `.engines/llamacpp/` clone + harness build (cmake, ~16 s); `.venvs/llamacpp` for `convert_hf_to_gguf.py`, plus a fallback converter env `.engines/llamacpp/convert-tf5` (llama.cpp's requirements with transformers 5.17.0) for repos whose tokenizer files the pinned transformers 4.57.6 cannot read; each GGUF's `.vocab.json` sidecar records which env built it | batch JSON-lines mode; exact vocab-GGUF detokenization; pin the template source; one token per streamed event |
| transformers | 5.17.0 | `.venvs/transformers` | supported only where the Hub repo ships `response_template` (Gemma 4 today); one token per streamed event |
| ollama | `7af39318` | `.engines/ollama/` clone + `go build` (Go version from Ollama's `go.mod`, via `GOTOOLCHAIN`) | built-in parsers only (the legacy `tools.NewParser` path is not covered); reuses the vocab GGUFs; one token per streamed event |

## 6. Discrepancies found (see `results/2026-09-25/`)

The spike's candidate discrepancies were turned into fixtures and re-verified by full runs. `results/2026-09-25/triage.jsonl` assigns every failing case to one finding, with a repro command. `tests/core/test_results_snapshot.py` keeps that record consistent with the committed results. What happened to the spike's candidates:

- vLLM `deepseek_v3`/`deepseek_v31`, SGLang `qwen25`/`deepseekv3`/`deepseekv31`: **confirmed**. A call (or its arguments) is lost when a delta carries a whole call. The SGLang case is also reproduced directly against `Qwen25Detector` (`results/2026-09-25/repro/`).
- SGLang `glm45`: **confirmed**. `\n<think>` leaks into non-streaming `reasoning_content`.
- llama.cpp DeepSeek-V3.1 HF template: **confirmed**, and it also affects DeepSeek-V3-0324. The autoparser derives malformed `preserved_tokens`, so `<｜tool▁sep｜>` is never rendered and every call fails to parse. This happens only with the HF template (which some popular GGUFs embed); llama.cpp's own tests use its rewritten `models/templates/` copy, which works.
- Ollama `qwen3-thinking` leading newline: a whitespace-only split variance (`soft_pass`), not a failure.
- Mistral `[CALL_ID]`: **unverified format**. mistral-common's generation grammar has no `[CALL_ID]`, but it only constrains guided decoding, and llama.cpp's tests assume the model emits it. Until a recorded Mistral-Small-3.2 generation settles it, these cases are triaged as `format_unverified`, never as bugs.
- Streaming granularity: llama-server, Ollama and transformers `serve` emit one event per token, so multi-token chunkings are synthetic for those engines (`run.synthetic_strategies`) and never count. Three Ollama "failures" of the first snapshot existed only under such chunkings and are gone.
- Truncated fixtures: an earlier run appended a stop token to output cut by `max_tokens`. That fabricated three failures (SGLang gpt-oss, llama.cpp Kimi K3); adapters now replay those fixtures with `finish_reason: "length"`.

This project has reported nothing upstream. Ten findings match issues that others had already filed (`engine_bug_known_upstream`, with the issue in `upstream`). One of them, llama.cpp#27720, was closed by the maintainer as not feasible to handle. `RELEASE_CHECKLIST.md` §7 has draft reports for the rest. Re-check each finding against the engine's latest release before filing it.

## 7. Definition of done (v0.1)

- At least 2 real engine adapters run their genuine parser code against ≥150 fixtures. Target: vLLM, SGLang and llama.cpp.
- The matrix site is generated only from real `results/*.json`.
- `uv build` and `twine check` pass.
- README, CONTRIBUTING, the spec, LICENSE and the CI workflows are present.
- Nothing is published.

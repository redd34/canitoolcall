# Mistral / Magistral / Devstral / Ministral, slug `mistral`

**Current models (HF, 2026-09-25):**
- Mistral-Medium-3.5-128B (2026-03)
- Mistral-Small-4-119B-2603 (2026-01…03)
- Ministral-3-3B/8B/14B-Instruct-2512 and Ministral-3-*-Reasoning-2512 (2025-10)
- Devstral-2-123B and Devstral-Small-2-24B-Instruct-2512 (2025-11)
- Magistral-Small-2509 (2025-09)
- Mistral-Small-3.2-24B-Instruct-2506 (2025-06)
- Still heavily used: Mistral-7B-Instruct-v0.3 (2.3M downloads) and Mistral-Nemo

The Mistral format is defined by the **tokenizer version** in `mistral-common`, not by a Jinja template. Several repos (Mistral-Small-3.2, Magistral-Small-2509) ship **only** `tekken.json` and `params.json`, with no `tokenizer_config.json` or chat template. Newer repos (Ministral-3, Devstral-2, Medium-3.5, Small-4) also ship `chat_template.jinja`. The reference renderer is `mistral-common`; all examples below come from actually running `mistral-common==1.12.0` or the HF templates.

## Formats by tokenizer version

In the SentencePiece (non-tekken) renderings, `▁` is SentencePiece's visible space marker.

### v3 (Mistral-7B-Instruct-v0.3, Mixtral) and v3-tekken (Mistral-Nemo): JSON array

```
[/INST][TOOL_CALLS]▁[{"name":▁"get_weather",▁"arguments":▁{"city":▁"Zürich"},▁"id":▁"abcDEF123"},▁{"name":▁"get_weather",▁"arguments":▁{"city":▁"Bern"},▁"id":▁"xyzXYZ789"}]</s>[TOOL_RESULTS]▁{"content":▁{"temp":▁20},▁"call_id":▁"abcDEF123"}[/TOOL_RESULTS]…
```

The tekken (Nemo) version is the same without `▁`: `[TOOL_CALLS][{"name": "get_weather", "arguments": {...}, "id": "abcDEF123"}, …]`.

### v7 (rendered with `MistralTokenizer.v7()`): JSON array, new tool-result encoding

```
[TOOL_CALLS]▁[{"name":▁"get_weather",▁"arguments":▁{"city":▁"Zürich"},▁"id":▁"abcDEF123"}, …]</s>[TOOL_RESULTS]▁abcDEF123[TOOL_CONTENT]▁{"temp":20}[/TOOL_RESULTS]
```

### v11 (Mistral-Small-3.2-24B-Instruct-2506): one `[TOOL_CALLS]` per call, with `[CALL_ID]`

Rendered with `MistralTokenizer.from_hf_hub("mistralai/Mistral-Small-3.2-24B-Instruct-2506")`, which reports `TokenizerVersion.v11`:

```
[/INST][TOOL_CALLS]get_weather[CALL_ID]abcDEF123[ARGS]{"city": "Zürich"}[TOOL_CALLS]get_weather[CALL_ID]xyzXYZ789[ARGS]{"city": "Bern"}</s>[TOOL_RESULTS]abcDEF123[TOOL_CONTENT]{"temp":20}[/TOOL_RESULTS]…
```

### v13+ (Magistral-Small-2509, Ministral-3, Devstral-2, Medium-3.5, Small-4): `[TOOL_CALLS]NAME[ARGS]{json}`, no call ID

For Magistral-Small-2509, `mistral-common` reports `TokenizerVersion.v13`:

```
[/INST][TOOL_CALLS]get_weather[ARGS]{"city": "Zürich"}</s>
```

The HF Jinja templates give the same shape. Here is our two-call render from Mistral-Medium-3.5-128B `chat_template.jinja` (rev `22b2b86`); Mistral-Small-4 (rev `a11f36b`) is identical:

```
[/INST][THINK]I should call the tools.[/THINK][TOOL_CALLS]get_weather[ARGS]{"city": "Zürich", "unit": "c"}[TOOL_CALLS]search[ARGS]{"query": "café \"best\"", "filters": {"tags": ["a", "b"], "max": 3}}</s>[TOOL_RESULTS]{"temp": 20}[/TOOL_RESULTS][TOOL_RESULTS][][/TOOL_RESULTS][THINK]Done.[/THINK]It is 20C.</s>
```

- **Separators:** there are none between calls. Each call restarts with `[TOOL_CALLS]`.
- **Arguments:** JSON, where a dict goes through `tojson` and a string is passed through verbatim. Our first run passed ASCII-escaped strings and got `Zürich` back, which shows that string arguments are not re-serialized.
- **Tool results** in the HF templates are `[TOOL_RESULTS]…[/TOOL_RESULTS]` with no ID.

### Reasoning
- **v13+ reasoning models** (Magistral-2509, Medium-3.5, Small-4, Ministral-3-Reasoning) use `[THINK]…[/THINK]` **special tokens**. Medium-3.5 has ids 34/35.
- **Older v11 Magistral** (2506/2507) used plain-text `<think>…</think>`. vLLM's `vllm/parser/mistral.py` documents both modes: `"special_token"` for v13+ `[THINK]`/`[/THINK]` and `"text"` for v11 `<think>`/`</think>`.
- **Instruct-only** models (Ministral-3-Instruct, Devstral-2) have no reasoning.

### Tool-call ID constraint
The HF templates **raise** `Tool call IDs should be alphanumeric strings with length 9!`. We observed this with llama.cpp's copy of the Mistral-Small-3.2 template when passing `call_1`. OpenAI-style IDs such as `call_abc…` must be rewritten by clients or engines.

## Special tokens (Mistral-Medium-3.5 `tokenizer.json`)

| id | token |
|---|---|
| 2 | `</s>` (EOS) |
| 3 / 4 | `[INST]` / `[/INST]` |
| 5 / 6 | `[AVAILABLE_TOOLS]` / `[/AVAILABLE_TOOLS]` |
| 7 / 8 | `[TOOL_RESULTS]` / `[/TOOL_RESULTS]` |
| 9 | `[TOOL_CALLS]` |
| 17 / 18 | `[SYSTEM_PROMPT]` / `[/SYSTEM_PROMPT]` |
| 19 | `[TOOL_CONTENT]` |
| 32 | `[ARGS]` |
| 33 | `[CALL_ID]` |
| 34 / 35 | `[THINK]` / `[/THINK]` |

All of these are control tokens. With `skip_special_tokens=True` they disappear from detokenized text, which destroys the call structure. Engines must parse on token IDs or keep special tokens.

## Sources

| Model | Source | URL |
|---|---|---|
| Mistral-Medium-3.5-128B | `chat_template.jinja` | https://huggingface.co/mistralai/Mistral-Medium-3.5-128B/blob/22b2b868a15677cfa6061277ed2f653d1349a9ab/chat_template.jinja |
| Mistral-Small-4-119B-2603 | `chat_template.jinja` | https://huggingface.co/mistralai/Mistral-Small-4-119B-2603/blob/a11f36bebf709121056b1dbcc943d1c6afbe494d/chat_template.jinja |
| Ministral-3-14B-Instruct-2512 | `chat_template.jinja` | https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512/blob/29439f81c2be264d8d393273f99e7db9c0961120/chat_template.jinja |
| Devstral-Small-2-24B-Instruct-2512 | `chat_template.jinja` | https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512/blob/55c5b41e98c2dbd21b0c8afffc540dcfc9eb5128/chat_template.jinja |
| Magistral-Small-2509 | `tekken.json` (v13), no template | https://huggingface.co/mistralai/Magistral-Small-2509/tree/a31cc96ab10cf19bc42c628fedf1e359e0853c49 |
| Mistral-Small-3.2-24B-Instruct-2506 | `tekken.json` (v11), no template | https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506/tree/95a6d26c4bfb886c58daf9d3f7332c857cb27b43 |
| Reference renderer | `mistral-common` 1.12.0 (Apache-2.0) | https://github.com/mistralai/mistral-common |

## Engine parser mapping

| Engine | Tool parser | Reasoning parser | Notes |
|---|---|---|---|
| vLLM v0.34.0 | `mistral` (`MistralToolParser`; grammar `vllm/parser/mistral.py`, which handles "both pre-v11 JSON-array and v11+ `funcname{args}` formats") | `mistral` (`MistralParserReasoningAdapter`) | The Medium-3.5, Small-4 and Devstral-2 model cards use `--tool-call-parser mistral --enable-auto-tool-choice`, and Medium-3.5 and Small-4 add `--reasoning-parser mistral`. `supports_required_and_named = False`, because `tool_choice` is enforced through the mistral-common grammar. |
| SGLang v0.5.20 | `mistral` (`MistralDetector`, canonical bot `"[TOOL_CALLS] ["`; also parses compact `[TOOL_CALLS]name[ARGS]{…}` and tolerates missing `]`/`[ARGS` delimiters while streaming) | `mistral` | |
| llama.cpp v0.5.0 | PEG, from template. Test templates: `Mistral-Small-3.2-24B-Instruct-2506.jinja`, `mistralai-Ministral-3-14B-Reasoning-2512.jinja`, `mistralai-Mistral-Nemo-Instruct-2407.jinja`, `unsloth-mistral-Devstral-Small-2507.jinja` | same | |
| Ollama v0.34.4 | `ministral`, plus generic template-driven detection for older models | | |

**`[CALL_ID]` (v11) is not in the generation grammar.** mistral-common 1.12.0 constrains generation with `guidance/grammar_factory.py` `_TOOL_CALL_GRAMMAR = "{tool_calls_token} SAFE_WS? {tool_name} {args_token} SAFE_WS? %json {args_json} SAFE_WS?"`, which has no call id, and `InstructTokenizerV11` only writes `[CALL_ID]id` when the history message carries an id. llama.cpp's tests nevertheless parse `[TOOL_CALLS]name[CALL_ID]id[ARGS]{…}` (`tests/test-chat.cpp`, Mistral Small 3.2), so the fixtures cover both shapes: `[CALL_ID]` cases come from those engine tests, never from history renders.

**Fixture sources (Apache-2.0):**
- vLLM: `tests/tool_use/mistral/test_mistral_tool_calls.py`, `tests/parser/mistral/`
- SGLang: `test/registered/unit/function_call/test_mistral_detector.py`
- `mistral-common`'s own tests

## Known edge cases (real reports)

| Edge case | URL |
|---|---|
| Mistral v11+ parser includes trailing text in function arguments | https://github.com/vllm-project/vllm/issues/48975 |
| v11 streaming merges parallel calls into one when content precedes `[TOOL_CALLS]` in a single delta | https://github.com/vllm-project/vllm/issues/48318 |
| Ministral 3 streaming tool calls not working | https://github.com/vllm-project/vllm/issues/29968 |
| Devstral-Small-2507 streaming tool parsing issue | https://github.com/vllm-project/vllm/issues/23180 |
| The "always adjust_request" grammar path rebuilds the parser and compiles regex over the full vocab on every request | https://github.com/vllm-project/vllm/issues/58145 |
| Tool call silently dropped when a tool parameter is named `name` (devstral-small-2) | https://github.com/ollama/ollama/issues/16932 |
| Tool call tag may have an extraneous `[` | https://github.com/ollama/ollama/issues/11470 |
| 8 SGLang parsers (incl. mistral) wrong in token-by-token streaming | https://github.com/sgl-project/sglang/issues/35564 |
| 13 SGLang parsers delete the message when generation stops at a tool-call open marker | https://github.com/sgl-project/sglang/issues/35565 |

**Fixture tags to cover:**
- `v3-json-array`
- `v7-json-array`
- `v11-call-id`
- `v13-compact`
- `think-special-token`
- `think-text` (v11 Magistral)
- `content-then-calls-single-delta`
- `trailing-text-after-args`
- `param-named-name`
- `id-format-9-alnum`
- `skip-special-tokens`
- `stop-at-open-marker`

## Fixtures (`fixtures/mistral/`)

| File | Source | Generator |
|---|---|---|
| `rendered.jsonl` | `mistral-common` 1.12.0 renders for v13 + `[THINK]` (Magistral-Small-2509, reference), v13 instruct (Ministral-3-14B-Instruct-2512), v11 (Mistral-Small-3.2) and v3 (Mistral-7B-Instruct-v0.3) | `scripts/fixtures/mistral/render_mistral.py` |
| `imported.jsonl` | vLLM v0.30.0 `tests/parser/mistral/`, llama.cpp v0.5.0 `tests/test-chat.cpp` (Ministral-3 Reasoning, Mistral-Small-3.2 `[CALL_ID]`, Devstral), Ollama #11470 | `scripts/fixtures/mistral/import_mistral.py` |

Conventions:
- Every fixture carries `output_token_ids`. Tekken models are pinned with `mode: mistral`. Mistral-7B-v0.3 is pinned with `mode: hf`: its `tokenizer.json` has the same ids, and the HF decoder turns SentencePiece `▁` into spaces.
- Renders use no call ids, matching the generation grammar. v13+ validation requires ids but never tokenizes them; the script asserts that `[CALL_ID]` is absent.
- Extra models are listed in `models` only when their own tokenizer renders byte-identical ids.
- For copied strings, control markers become control-token ids and the text between them is encoded normally (vLLM's `encode_mistral_output`).

# GLM (4.5, 4.6, 4.7, 5.x), slug `glm`

**Models (Z.ai / zai-org):**
- GLM-4.5 and GLM-4.5-Air (2025-07)
- GLM-4.6 (2025-09)
- GLM-4.7 and GLM-4.7-Flash (2025-12, 2026-01)
- GLM-5 (2026-02), GLM-5.1 (2026-04) and GLM-5.2 (2026-06)
- **GLM-5.3 and GLM-5.3-Flash (2026-08-25)**, which lead the zai-org download chart

## Format

Each call is:
- `<tool_call>` + function name
- then, for each argument, `<arg_key>K</arg_key><arg_value>V</arg_value>`
- then `</tool_call>`

The name is **bare text** right after `<tool_call>`, with no attribute or separator token. Values follow the template rule `v | tojson(ensure_ascii=False) if v is not string else v`: strings are raw and unescaped, and everything else is JSON. A parser must therefore use the tool schema to tell the string `"3"` from the number `3`.

There are **two whitespace variants**.

### Variant A: GLM-4.5 and GLM-4.6 (newline-separated)

The two templates are byte-identical (GLM-4.5 rev `cbb2c7c`, GLM-4.6 rev `be72194`). Rendered:

```
<|assistant|>
<think>I should call the tools.</think>
<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Zürich</arg_value>
<arg_key>unit</arg_key>
<arg_value>c</arg_value>
</tool_call>
<tool_call>search
<arg_key>query</arg_key>
<arg_value>café "best"</arg_value>
<arg_key>filters</arg_key>
<arg_value>{"tags": ["a", "b"], "max": 3}</arg_value>
</tool_call><|observation|>
<tool_response>
{"temp": 20}
</tool_response>
<tool_response>
[]
</tool_response><|assistant|>
```

### Variant B: GLM-4.7, 5.x (compact, no newlines)

Rendered from GLM-4.7 (rev `602d01e`). GLM-5.3 (rev `aca966e`) and GLM-5.3-Flash (rev `eb9eb20`) give the same output:

```
<|assistant|><think>I should call the tools.</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Zürich</arg_value><arg_key>unit</arg_key><arg_value>c</arg_value></tool_call><tool_call>search<arg_key>query</arg_key><arg_value>café "best"</arg_value><arg_key>filters</arg_key><arg_value>{"tags": ["a", "b"], "max": 3}</arg_value></tool_call><|observation|><tool_response>{"temp": 20}</tool_response><tool_response>[]</tool_response><|assistant|>
```

**Scalars** (verified by rendering `{"flag": true, "n": 3, "x": null, "s": "  two  spaces\n"}`) come out as:

```
<arg_value>true</arg_value>
<arg_value>3</arg_value>
<arg_value>null</arg_value>
<arg_value>  two  spaces\n</arg_value>
```

Unlike Qwen XML, the value has no wrapping newlines, so all whitespace is significant.

### Reasoning and generation prompt

| Version | Generation prompt | Raw completion starts with |
|---|---|---|
| 4.5 / 4.6 | `<\|assistant\|>`, or `<\|assistant\|>\n<think></think>` when `enable_thinking=false` | the model emits `\n<think>` itself |
| 4.7 | `<\|assistant\|><think>` (thinking) or `<\|assistant\|></think>` (non-thinking) | reasoning text, **no opening tag**; or content after a lone `</think>` |
| 5.3 | always `<\|assistant\|><think>`. The system prompt is prefixed with `Reasoning Effort: Low\|High\|Max` (default `max`). | reasoning text, no opening tag |

- **Reasoning history:** GLM-5.3 has `clear_thinking` (default `false`), so it **keeps** reasoning in history. Older versions keep reasoning only after the last user turn.
- **Tool results:** these come back after `<|observation|>`, wrapped in `<tool_response>`.
- **`/nothink`:** this is a token (id 151360 in GLM-4.5, 154851 in GLM-5.3) that users append to disable thinking in 4.5/4.6.

## Special tokens

| token | GLM-4.5 id | GLM-5.3 id | special? |
|---|---|---|---|
| `<\|system\|>` / `<\|user\|>` / `<\|assistant\|>` / `<\|observation\|>` | 151335–151338 | 154826–154829 | yes |
| `<think>` / `</think>` | 151350 / 151351 | 154841 / 154842 | no |
| `<tool_call>` / `</tool_call>` | 151352 / 151353 | 154843 / 154844 | no |
| `<tool_response>` / `</tool_response>` | 151354 / 151355 | 154845 / 154846 | no |
| `<arg_key>` / `</arg_key>` | 151356 / 151357 | 154847 / 154848 | no |
| `<arg_value>` / `</arg_value>` | 151358 / 151359 | 154849 / 154850 | no |
| `/nothink` | 151360 | 154851 | no in 5.3 |
| `generation_config.eos_token_id` | [151329, 151336, 151338] | [154820, 154827, 154829] | |

The EOS list is `<|endoftext|>`, `<|user|>` and `<|observation|>`. After tool calls the model stops on `<|observation|>`, so raw text normally ends right after `</tool_call>` once the stop token is stripped.

## Template sources

| Model | URL |
|---|---|
| GLM-5.3 | https://huggingface.co/zai-org/GLM-5.3/blob/aca966e4e02791568aa6a4ced368624b3d897f42/chat_template.jinja |
| GLM-5.3-Flash | https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/chat_template.jinja |
| GLM-4.7 | https://huggingface.co/zai-org/GLM-4.7/blob/602d01efcdd332c5238ca4bcede555defbe83eb7/chat_template.jinja |
| GLM-4.6 | https://huggingface.co/zai-org/GLM-4.6/blob/be72194883d968d7923a07e2f61681ea9a2826d1/chat_template.jinja |
| GLM-4.5 | https://huggingface.co/zai-org/GLM-4.5/blob/cbb2c7cfb52fa128a9660cb1a7a78e017899e115/chat_template.jinja |

## Engine parser mapping

| Engine | Tool parser | Reasoning parser |
|---|---|---|
| vLLM v0.34.0 | `glm45` and `glm47` **both** map to `Glm47MoeModelToolParser` (grammar `vllm/parser/glm47_moe.py`). Its arg regex allows `\s*` between `</arg_key>` and `<arg_value>` and strips keys, but keeps values verbatim. `supports_required_and_named = False`. | `glm45` / `glm47` (`Glm47MoeParserReasoningAdapter`) |
| SGLang v0.5.20 | `glm` / `glm45` → `Glm4MoeDetector` (4.5/4.6); `glm47` → `Glm47MoeDetector` (4.7/5.x) | `glm45` |
| llama.cpp v0.5.0 | PEG, from template. Test templates: `GLM-4.6.jinja`, `GLM-4.7-Flash.jinja` | same |
| Ollama v0.34.4 | `glm-4.7` (also `glm46.go`) | built in |

**Official recommendation:** the GLM-4.7 model card uses `--tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice`.

**Fixture sources (Apache-2.0):**
- vLLM: `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `test_glm47_moe_tool_parser.py`, `tests/reasoning/test_glm4_moe_reasoning_parser.py`
- SGLang: `test/registered/unit/function_call/test_glm47_schema_types.py`

## Known edge cases (real reports)

| Edge case | URL |
|---|---|
| GLM `tool_choice=required` + streaming repeats JSON | https://github.com/vllm-project/vllm/issues/47504 |
| GLM streaming final chunks repeat metadata and combine arguments with `finish_reason` | https://github.com/vllm-project/vllm/issues/44098 |
| GLM-5 FP8 function-call error with Claude Code | https://github.com/vllm-project/vllm/issues/44843 |
| `glm47_moe` (and `deepseek_v4`, `minimax_m2`) drop the last parameter when the closing tag is omitted | https://github.com/vllm-project/vllm/issues/57826 |
| SGLang GLM parser corrupts number-looking strings with underscores (`"123_456"` → `123456`) | https://github.com/sgl-project/sglang/issues/30644 |
| SGLang GLM parser leaks raw `<tool_call>` markup into content with `tool_choice="none"` | https://github.com/sgl-project/sglang/issues/30925 |
| SGLang `Glm47MoeDetector` sets `tool_index` to the tool's position in `tools`, not the call index | https://github.com/sgl-project/sglang/issues/33324 |
| 8 SGLang parsers give wrong or missing calls in token-by-token streaming | https://github.com/sgl-project/sglang/issues/35564 |
| 9 SGLang parsers drop arguments when one streaming increment contains more than one complete call | https://github.com/sgl-project/sglang/issues/37634 |

**Fixture tags to cover:**
- `whitespace-variant-4.5`
- `whitespace-variant-4.7`
- `number-looking-string`
- `schema-coercion`
- `marker-in-string` (a value containing `</arg_value>`)
- `tool-choice-none`
- `call-index-vs-tool-index`
- `multi-call-single-delta`
- `think-no-open-tag`
- `missing-close-arg`

## Fixtures in this repo

`fixtures/glm/` (51 fixtures) is built by `uv run --script scripts/fixtures/glm/build.py`. The build is deterministic and downloads only tokenizer and template files at pinned revisions.

| File | Source | `models` |
|---|---|---|
| `glm45-render.jsonl` | GLM-4.5 `chat_template.jinja` @ `cbb2c7c` (newline variant) | GLM-4.5, GLM-4.5-Air, GLM-4.6 (renders checked identical) |
| `glm47-render.jsonl` | GLM-4.7 `chat_template.jinja` @ `602d01e` (compact variant) | GLM-4.7, GLM-4.7-Flash (identical text; Flash uses the 154xxx GLM-5 vocabulary) |
| `glm53-render.jsonl` | GLM-5.3 `chat_template.jinja` @ `aca966e` | GLM-5.3 |
| `truncated-malformed.jsonl` | token prefixes of renders; one render with its last `</arg_value>` token deleted (vLLM #57826) | |
| `imported.jsonl` | vLLM v0.30.0 tests (Apache-2.0), Ollama `7af39318` `glm46_test.go` (MIT) and SGLang issue #33324 | |

Notes:
- **Expected content and reasoning** are the message as authored. The separators that the template adds (GLM-4.5's `\n` before `<think>`, content and `<tool_call>`) are format glue, not content. Engines that keep them get `soft_pass`.
- **Typed values.** `glm/*-schema-coercion` renders `seconds=3` (integer) and `label="3"` (string) as the same text `3`; only the schema tells them apart. `glm/glm47-number-looking-string` keeps `123_456` and `0042` as strings (SGLang #30644).
- **Marker inside a value.** `glm/*-marker-in-arguments` contains `<tool_call>`, `</tool_call>` and `<arg_key>` inside a string value, but not `</arg_value>`, so the value is still delimited unambiguously.
- **Thinking disabled.** Engine-test strings that start directly with content or a call are recorded with `thinking=false` and the generation prompt that makes them a complete completion (`<|assistant|>\n<think></think>` for 4.5, `<|assistant|></think>` for 4.7).
- **models[0] and parsers.** GLM-5.3 fixtures use `zai-org/GLM-5.3` as `models[0]`. Parser maps keyed on `GLM-4.7` must also map `GLM-5` to the glm47 parsers.


# Qwen XML tool calls (Qwen3-Coder, Qwen3.5, 3.6, 3.8), slug `qwen3-xml`

**Models:**
- Qwen3-Coder-30B-A3B-Instruct and Qwen3-Coder-480B-A35B (2025-07)
- Qwen3-Coder-Next (2026-02)
- Qwen3.5-* (2026-02)
- Qwen3.6-* (2026-04)
- Qwen3.8-27B / Qwen3.8-Flash-Next / Qwen3.8-2.4T-A95B (2026-08)

These are today's most-downloaded Qwen LLMs. Starting with **Qwen3.5, general Qwen models switched from Hermes JSON to the Coder XML format**, and added default-on thinking. Treating Qwen3.5+ as `hermes` is a common misconfiguration.

## Format

Each call is:
- `<tool_call>` + `\n`
- `<function=NAME>` + `\n`
- zero or more `<parameter=KEY>` + `\n` + VALUE + `\n` + `</parameter>` + `\n`
- `</function>` + `\n` + `</tool_call>`

**Parallel calls** are separate `<tool_call>` blocks joined by `\n`. If there is text content, the first call is preceded by `\n\n` of markup in both Qwen3.5+ and Qwen3-Coder (the Coder template writes `'\n' + content + '\n'` and then `'\n<tool_call>'`; verified by rendering with transformers 5.17).

**Values are raw text, not JSON:**
- String values are inserted verbatim, with no escaping at all. Quotes, `<`, `&` and newlines all appear literally.
- Objects and arrays are `tojson`.
- Parsers must coerce scalar values back to JSON **using the tool's JSON schema**. For example, `3` becomes an integer only if the schema says `integer`.

Rendered from Qwen3.8-27B (rev `1d4bf0f`). Qwen3.5-9B and Qwen3.6 are identical for this input:

```
<|im_start|>assistant
<think>
I should call the tools.
</think>

<tool_call>
<function=get_weather>
<parameter=city>
Zürich
</parameter>
<parameter=unit>
c
</parameter>
</function>
</tool_call>
<tool_call>
<function=search>
<parameter=query>
café "best"
</parameter>
<parameter=filters>
{"tags": ["a", "b"], "max": 3}
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"temp": 20}
</tool_response>
<tool_response>
[]
</tool_response><|im_end|>
```

Qwen3-Coder-30B-A3B (rev `b2cff64`) produces the same call block, with three differences:
- no `<think>`
- the `\n` after `<|im_start|>assistant` is emitted by the tool-call prefix (`'\n<tool_call>'`) rather than the role header. The generation prompt `<|im_start|>assistant\n` already contains it, so a completion without content starts directly with `<tool_call>`
- `</tool_response>\n` before `<|im_end|>`

### Scalar serialization differs between template revisions (verified by rendering)

The input was `{"flag": true, "n": 3, "x": null, "s": "  two  spaces\n"}`:

| Template | `flag` | `x` | `s` |
|---|---|---|---|
| Qwen3-Coder-30B-A3B (`b2cff64`), Qwen3.5-9B (`c202236`) | `True` | `None` | `  two  spaces\n` (raw, with the trailing newline kept, so the value ends `\n\n</parameter>`) |
| Qwen3-Coder-Next (`a7fbcb5`), Qwen3.6-35B-A3B (`995ad96`), Qwen3.8-27B (`1d4bf0f`) | `true` | `null` | same |

The older templates use Jinja `| string` on Python values, which yields `True`/`None`. So a model trained on those templates may emit Python literals, and a parser must decide whether `True` is the boolean or the string `"True"`. The wrapping newlines are template markup. A parser should strip exactly one leading and one trailing `\n`; vLLM's `_trim_wrapping_newlines` in `vllm/parser/qwen3.py` does this. It must not strip other whitespace (see vLLM #48753 below).

### Reasoning
- **Qwen3.5/3.6/3.8 generation prompt:** `<|im_start|>assistant\n<think>\n`. The completion starts inside reasoning with **no opening `<think>`** and contains `</think>\n\n` before the answer or calls.
- **`enable_thinking=false`:** the prompt pre-fills `<think>\n\n</think>\n\n`.
- **Qwen3.8 effort control:** it adds a `reasoning_effort` kwarg (`xhigh` default, `medium` or `low`) that injects a sentence into the system prompt.
- **History:** Qwen3.8 has `preserve_thinking` **default true**, which keeps reasoning in history. In Qwen3.6 it is opt-in. In Qwen3.5, reasoning is kept only after the last user query.
- **Qwen3-Coder** has no reasoning.
- **The model may emit `<tool_call>` inside `<think>`** (see issues below). This is the single most-reported failure for this family.

## Special tokens (Qwen3.5/3.6/3.8 share one vocab)

| id (3.5+) | id (Coder) | token | special? |
|---|---|---|---|
| 248045 | 151644 | `<\|im_start\|>` | yes |
| 248046 | 151645 | `<\|im_end\|>` (EOS) | yes |
| 248058 / 248059 | 151657 / 151658 | `<tool_call>` / `</tool_call>` | no |
| 248066 / 248067 | 151665 / 151666 | `<tool_response>` / `</tool_response>` | no |
| 248068 / 248069 | 151667 / 151668 | `<think>` / `</think>` | no |

`<function=`, `<parameter=`, `</parameter>` and `</function>` are **not** single tokens. They are ordinary BPE text, so they can be split across stream deltas even with per-token streaming.

## Template sources

| Model | URL |
|---|---|
| Qwen3.8-27B | https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/chat_template.jinja |
| Qwen3.6-35B-A3B | https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/chat_template.jinja |
| Qwen3.5-9B | https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/chat_template.jinja |
| Qwen3-Coder-Next | https://huggingface.co/Qwen/Qwen3-Coder-Next/blob/a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb/chat_template.jinja |
| Qwen3-Coder-30B-A3B-Instruct | https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/blob/b2cff646eb4bb1d68355c01b18ae02e7cf42d120/chat_template.jinja |

## Engine parser mapping

| Engine | Tool parser | Reasoning parser |
|---|---|---|
| vLLM v0.34.0 | `qwen3_coder` or `qwen3_xml`: both map to `Qwen3EngineToolParser` (`vllm/tool_parsers/qwen3_engine_tool_parser.py`), with the grammar in `vllm/parser/qwen3.py`. (`mimo` reused it up to v0.30.0; v0.34.0 gives it its own `MiMoToolParser`.) | `qwen3` |
| SGLang v0.5.20 | `qwen3_coder` (`Qwen3CoderDetector`); also reused by `step3p5` and `nanbeige` | `qwen3` |
| llama.cpp v0.5.0 | Automatic (PEG), from template. Test templates: `models/templates/Qwen3-Coder.jinja`, `Qwen3.5-4B.jinja` | same |
| Ollama v0.34.4 | `qwen3-coder`, `qwen3.5` | built in |

**Official recommendation:** the Qwen3.5-9B model card says `--tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice`, and the Qwen3-Coder-Next card says `--tool-call-parser qwen3_coder`.

**Fixture sources (Apache-2.0):**
- `vllm/tests/tool_parsers/test_qwen3coder_tool_parser.py`
- `vllm/tests/reasoning/test_qwen3_reasoning_parser.py`

## Known edge cases (real reports)

| Edge case | URL |
|---|---|
| Tool calls printed in XML *inside the thinking block*, then the model stops (60 comments) | https://github.com/ggml-org/llama.cpp/issues/20837 |
| vLLM may lose tool calls for Qwen3.5-35B-A3B when XML `<tool_call>` is emitted inside `<think>` | https://github.com/vllm-project/vllm/issues/39056 |
| `qwen3_xml` parser consumes `</think>`, merging reasoning into content | https://github.com/vllm-project/vllm/issues/51679 |
| Last parameter dropped when the model omits `</parameter>` before `</function>` (non-streaming `{}`, streaming unterminated JSON) | https://github.com/vllm-project/vllm/issues/57699 |
| Parser strips meaningful whitespace from string values (breaks exact-match edit tools) | https://github.com/vllm-project/vllm/issues/48753 |
| Misalignment between the Qwen3.5 chat template and the recommended parser | https://github.com/vllm-project/vllm/issues/38885 |
| Behaviour change between vLLM 0.22 and 0.23 for `qwen3_coder` | https://github.com/vllm-project/vllm/issues/46493 |
| SGLang `qwen3_coder` non-streaming drops visible text after a valid tool call | https://github.com/sgl-project/sglang/issues/40739 |
| `common_chat_parse` silently drops Qwen3-Coder tool calls | https://github.com/ggml-org/llama.cpp/issues/27363 |
| Qwen3-Coder-Next produces invalid JSON tool calls | https://github.com/ggml-org/llama.cpp/issues/19382 |
| Tool call lost when the model omits the `<tool_call>` wrapper (bare `<function=…>`) | https://github.com/ollama/ollama/issues/17353 |
| Tool call lost when reasoning precedes it and the opener is missing | https://github.com/ollama/ollama/issues/18530 |
| Number arguments outside the int64 range changed | https://github.com/ollama/ollama/issues/18421 |
| Qwen3.6 violates its own template; the qwen3.5 parser returns 500 instead of tolerating it | https://github.com/ollama/ollama/issues/16383 |

**Fixture tags to cover:**
- `call-inside-think`
- `missing-close-param`
- `bare-function` (no `<tool_call>`)
- `python-literals` (`True`/`None`)
- `whitespace-significant-string`
- `multiline-string`
- `marker-in-string` (a value containing `</parameter>`)
- `schema-coercion` (`"3"` vs `3`)
- `think-no-open-tag`
- `parallel`
- `text-after-call`

## Fixtures (`fixtures/qwen3-xml/`)

| File | Source | Generator |
|---|---|---|
| `rendered.jsonl` | HF chat templates of Qwen3.8-27B (reference; Qwen3.6-35B-A3B listed when byte-identical), Qwen3.5-9B, Qwen3-Coder-30B-A3B (+480B when identical) and Qwen3-Coder-Next | `scripts/fixtures/qwen3-xml/render_qwen3_xml.py` |
| `imported.jsonl` | llama.cpp v0.5.0 `tests/test-chat.cpp` (Qwen3.5-4B and Qwen3-Coder templates), vLLM v0.30.0 `tests/tool_parsers/test_qwen3coder_tool_parser.py`, bug reports (vLLM #57699, Ollama #18530, #18421, SGLang #40739) | `scripts/fixtures/qwen3-xml/import_qwen3_xml.py` |

Conventions:
- Qwen3.5+ fixtures set `generation_prompt`: `<|im_start|>assistant\n<think>\n` by default, or the pre-filled empty think block when `thinking` is false. `raw_output` therefore starts inside the reasoning.
- If the prompt/completion boundary falls inside a merged BPE token (an empty reasoning renders `<think>\n\n</think>`), the render is sliced as text and the remainder re-encoded. The fixture's notes say so.
- Expected `content` omits the markup newlines between text and `<tool_call>`, so engines that keep them get `soft_pass`.
- Scalar arguments are expected in their JSON-schema type (`"02139"` stays a string, `True` with a boolean schema becomes `true`).
- `q38-marker-in-arguments` and `coder-marker-in-arguments` contain marker text that is not at a line start. Only one parse is consistent with the template (a value always ends with `\n</parameter>`). The fully line-aligned case is ambiguous and deliberately not included.

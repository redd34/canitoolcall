# DeepSeek (V3/R1 → V3.1 → V3.2 → V4 → V4.1), slug `deepseek`

**Models, newest first (HF, 2026-09-25):**
- DeepSeek-V4.1-Flash (2026-09-10)
- DeepSeek-V4-Flash-0731, V4-Flash-DSpark, V4-Flash and V4-Pro (2026-04…07)
- DeepSeek-V3.2 (2025-12)
- DeepSeek-V3.1 (2025-08)
- DeepSeek-R1-0528 and DeepSeek-V3-0324 (2025)

DeepSeek has shipped **five mutually incompatible tool-call syntaxes** in about 18 months. A parser for one generation silently fails on the next. The fixture spec should treat these as sub-formats: `deepseek-v3`, `deepseek-v31`, `deepseek-v32`, `deepseek-v4`, `deepseek-v41`.

From V3.2 onward **there is no Jinja chat template** in the HF repo: `tokenizer_config.json` has no `chat_template`. The official reference is a Python encoder in `encoding/`. Engines have to ship their own templates or renderers.

## Sub-format A: V3 / V3-0324 / R1 / R1-0528 (`deepseek-v3`)

Rendered from DeepSeek-R1-0528 `tokenizer_config.json` (rev `4236a6a`):

```
<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"city": "Zürich", "unit": "c"}
```<｜tool▁call▁end｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>search
```json
{"query": "café \"best\"", "filters": {"tags": ["a", "b"], "max": 3}}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>{"temp": 20}<｜tool▁output▁end｜>
<｜tool▁output▁begin｜>[]<｜tool▁output▁end｜><｜tool▁outputs▁end｜>It is 20C.<｜end▁of▁sentence｜>
```

- Each call has a literal type word `function`, then `<｜tool▁sep｜>` and the name.
- The arguments follow in a ```` ```json ```` fenced block.
- Calls are separated by `\n`.
- The template inserts `arguments` **as a string**. Passing a dict raises `TypeError: can only concatenate str (not "dict") to str`, which we observed. So the JSON text is whatever the caller serialized.
- R1 reasoning is `<think>\n…\n</think>`. The generation prompt differs by revision:
  - The original DeepSeek-R1 (rev `56d4cbb`) pre-fills `<｜Assistant｜><think>\n`, so the raw completion has **no opening tag**.
  - R1-0528 (rev `4236a6a`) emits only `<｜Assistant｜>`, and the model writes `<think>` itself.

## Sub-format B: V3.1 (`deepseek-v31`)

Rendered from DeepSeek-V3.1 `tokenizer_config.json` (rev `c0781d0`):

```
<｜Assistant｜></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"city": "Zürich", "unit": "c"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>search<｜tool▁sep｜>{"query": "café \"best\"", "filters": {"tags": ["a", "b"], "max": 3}}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁output▁begin｜>{"temp": 20}<｜tool▁output▁end｜><｜tool▁output▁begin｜>[]<｜tool▁output▁end｜>It is 20C.<｜end▁of▁sentence｜>
```

- **Changes from V3:**
  - The `function` type word is gone.
  - There is no code fence.
  - The arguments are bare JSON right after `<｜tool▁sep｜>`.
  - Calls are chained with **no separator**. The README says "chain them directly without separators or spaces".
- **Thinking:** the model card says "Toolcall is supported in non-thinking mode". Non-thinking is signalled by a pre-filled `</think>` right after `<｜Assistant｜>`, and thinking mode pre-fills `<think>`. The raw completion therefore contains **only `</think>` or nothing**, never a matched pair.
- **Tools input:** the HF template does **not** consume `tools` (0 references). Tool descriptions must be written into the system prompt in the README's `## Tools` format. vLLM ships `examples/tool_chat_template_deepseekv31.jinja` for this.

## Sub-format C: V3.2 DSML (`deepseek-v32`)

Rendered by running `encoding/encoding_dsv32.py` from DeepSeek-V3.2 (rev `a7e62ac`) with `thinking_mode="thinking"`:

```
<｜Assistant｜><think>I should call the tools.</think>

<｜DSML｜function_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="city" string="true">Zürich</｜DSML｜parameter>
<｜DSML｜parameter name="unit" string="true">c</｜DSML｜parameter>
</｜DSML｜invoke>
<｜DSML｜invoke name="search">
<｜DSML｜parameter name="query" string="true">café "best"</｜DSML｜parameter>
<｜DSML｜parameter name="filters" string="false">{"tags": ["a", "b"], "max": 3}</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls><｜end▁of▁sentence｜>

<function_results>
<result>{"temp": 20}</result>
<result>[]</result>
</function_results>

<think>Done.</think>It is 20C.<｜end▁of▁sentence｜>
```

- This is an XML-ish "DSML" syntax. `string="true"` means the value is a raw string. `string="false"` means the value is JSON: a number, bool, object or array.
- String values are **not escaped**, so `"` and `<` appear literally.
- `\n\n` separates the reasoning/content from the `<｜DSML｜function_calls>` block.
- Tool results come back as `<function_results><result>…</result></function_results>`.

## Sub-format D: V4 / V4-Flash / V4-Pro (`deepseek-v4`)

Rendered by running `encoding/encoding_dsv4.py` from DeepSeek-V4-Flash (rev `60d8d70`). The only change to the call block is the outer tag, `function_calls` → **`tool_calls`**:

```
<｜Assistant｜><think>I should call the tools.</think>

<｜DSML｜tool_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="city" string="true">Zürich</｜DSML｜parameter>
…
</｜DSML｜invoke>
</｜DSML｜tool_calls><｜end▁of▁sentence｜><｜User｜><tool_result>{"temp": 20}</tool_result>

<tool_result>[]</tool_result><｜Assistant｜><think>Done.</think>It is 20C.<｜end▁of▁sentence｜>
```

Tool results moved into a `<｜User｜>` turn as `<tool_result>` blocks. They are sorted by call order.

## Sub-format E: V4.1 (`deepseek-v41`)

Rendered by running `encoding/encoding.py` from DeepSeek-V4.1-Flash (rev `dba1be0`). The DSML tag names now have a **leading space** and the outer tag is **` calls`**:

```
<｜Assistant｜><think>I should call the tools.</think>

<｜DSML｜ calls>
<｜DSML｜ invoke name="get_weather">
<｜DSML｜ parameter name="city" string="true">Zürich</｜DSML｜ parameter>
<｜DSML｜ parameter name="unit" string="true">c</｜DSML｜ parameter>
</｜DSML｜ invoke>
<｜DSML｜ invoke name="search">
<｜DSML｜ parameter name="query" string="true">café "best"</｜DSML｜ parameter>
<｜DSML｜ parameter name="filters" string="false">{"tags": ["a", "b"], "max": 3}</｜DSML｜ parameter>
</｜DSML｜ invoke>
</｜DSML｜ calls><｜end▁of▁sentence｜>
```

The official `encoding/README.md` says: "DSML tag names use a leading space… The V4 format used `<｜DSML｜tool_calls>` without a space."

**Other V4.1 changes:**
- `Reasoning Effort: N (range 1-100 …)` prefix in thinking mode
- mid-conversation `<｜System｜>`
- a `<｜latest_reminder｜>` role
- namespaced tool names

## Special tokens

These are from `tokenizer.json` `added_tokens`. "ns" means `special: false`, so the token survives `skip_special_tokens`.

| token | V3.1 id | V4.1 id |
|---|---|---|
| `<｜begin▁of▁sentence｜>` | 0 | 0 |
| `<｜end▁of▁sentence｜>` (EOS) | 1 | 1 |
| `<｜User｜>` / `<｜Assistant｜>` | 128803 / 128804 (ns) | 128803 / 128804 (ns) |
| `<｜System｜>` | none | 128799 (ns) |
| `<think>` / `</think>` | 128798 / 128799 (ns) | 128821 / 128822 (ns): **ids moved** |
| `<｜tool▁calls▁begin｜>` / `…end｜>` | 128806 / 128807 (ns) | same, still present |
| `<｜tool▁call▁begin｜>` / `…end｜>` | 128808 / 128809 (ns) | same |
| `<｜tool▁sep｜>` | 128814 (ns) | same |
| `<｜tool▁output(s)▁begin/end｜>` | 128810–128813 (ns) | same |
| `｜DSML｜` | none | **128825 (ns)**, a single token. The surrounding `<`, `</`, ` calls>`, ` invoke name="` are ordinary BPE text. |
| `<｜latest_reminder｜>` | none | 128828 (ns) |

The characters are U+FF5C `｜` (full-width vertical bar) and U+2581 `▁` (lower one-eighth block), **not** ASCII `|` and `_`. Fixtures must preserve them byte-exactly.

## Sources

| Version | Source | URL |
|---|---|---|
| R1-0528 | `tokenizer_config.json` | https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/blob/4236a6af538feda4548eca9ab308586007567f52/tokenizer_config.json |
| V3.1 | `tokenizer_config.json` + README §ToolCall | https://huggingface.co/deepseek-ai/DeepSeek-V3.1/blob/c0781d039fb7a1ba2abc4add0bdc293e92d2b8db/tokenizer_config.json |
| V3.2 | `encoding/encoding_dsv32.py` (+ `test_output.txt`) | https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6/encoding/encoding_dsv32.py |
| V4 | `encoding/encoding_dsv4.py` (+ `tests/test_output_*.txt`) | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/encoding/encoding_dsv4.py |
| V4.1 | `encoding/encoding.py` + `encoding/README.md` | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/encoding/encoding.py |

The V3.2, V4 and V4.1 encoders also include `parse_message_from_completion_text`, DeepSeek's own reference parser. It is ideal as an oracle for expected results. Its README warns it "is designed to handle well-formatted model output only".

## Engine parser mapping

| Sub-format | vLLM v0.34.0 tool / reasoning | SGLang v0.5.20 tool / reasoning |
|---|---|---|
| V3 / R1 | `deepseek_v3` / `deepseek_r1` | `deepseekv3` / `deepseek-r1` |
| V3.1 | `deepseek_v31` / `deepseek_v3` | `deepseekv31` / `deepseek-v3` |
| V3.2 | `deepseek_v32` (`DeepSeekV32EngineToolParser`, grammar `vllm/parser/deepseek_v32.py`) / `deepseek_v3` (not stated in vLLM docs; this assumes the same `<think>` convention as V3.1) | `deepseekv32` / `deepseek-v3`, as stated in `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` |
| V4 | `deepseek_v4` / `deepseek_v4` | `deepseekv4` / `deepseek-v4` |
| V4.1 | `deepseek_v41` / `deepseek_v41` (grammar `vllm/parser/deepseek_v41.py`: `DSML_TOOL_START = "<｜DSML｜ calls>"`) | **No V4.1 support found.** No `DSML｜ ` (spaced) string anywhere in `python/sglang/srt` at v0.5.20. The `deepseekv4` detector expects `<｜DSML｜tool_calls>`. |

Other engines:
- **llama.cpp v0.5.0:** PEG, from template. Test templates: `deepseek-ai-DeepSeek-V3.1.jinja`, `-V3.2.jinja`, `-V4.jinja`, `-V4-Flash-0731.jinja`. These are llama.cpp-authored, since there is no upstream Jinja for V3.2+.
- **Ollama v0.34.4:** `deepseek3`.

**Fixture sources (Apache-2.0):**
- `vllm/tests/tool_parsers/test_deepseekv3_tool_parser.py`, `test_deepseekv31_tool_parser.py`, `test_deepseekv32_tool_parser.py`, `test_deepseekv4_tool_parser.py`
- `sglang/test/registered/unit/function_call/test_deepseekv4_detector.py`

**Fixture sources (DeepSeek):** the official `encoding/tests/test_output_*.txt` files. The repo `LICENSE` is MIT (checked for V3.2 and V4.1 at the pinned revisions).

## Known edge cases (real reports)

| Edge case | URL |
|---|---|
| Using the V4 parser on V4.1 output: the leading space breaks state-machine matching and the param regex, and no streamed args are returned | https://github.com/vllm-project/vllm/issues/58640 |
| V4-Flash-0731 intermittently emits a malformed DSML start wrapper | https://github.com/vllm-project/vllm/issues/51914 |
| DSML recovery emits ghost empty-argument calls on truncation and narration, and recovers undeclared tools | https://github.com/vllm-project/vllm/issues/56482 |
| V4 buffers long string arguments until `</parameter>`, so there is no incremental streaming | https://github.com/vllm-project/vllm/issues/52846 |
| `deepseek_v4` (and `glm47_moe`, `minimax_m2`) drop the last parameter when the closing tag is omitted | https://github.com/vllm-project/vllm/issues/57826 |
| V3.1: leading whitespace accumulates across multi-turn tool calling | https://github.com/vllm-project/vllm/issues/28804 |
| V3.2 `tool_calls` failure | https://github.com/vllm-project/vllm/issues/26897 |
| SGLang DSML parser wraps arguments in a spurious `"arguments"`/`"input"` key | https://github.com/sgl-project/sglang/issues/38924 |
| SGLang: DSML calls returned as content (bare invoke, unterminated section, malformed sibling) | https://github.com/sgl-project/sglang/issues/40236 |
| SGLang `deepseekv32`/`deepseekv4` drop calls when streamed output arrives in two chunks | https://github.com/sgl-project/sglang/issues/35563 |
| llama.cpp stops mid-tag when the reasoning stream contains DSML closing fragments (V4-Flash) | https://github.com/ggml-org/llama.cpp/issues/27613 |
| llama.cpp: V4 Flash errors on tool calls with similar parameter names | https://github.com/ggml-org/llama.cpp/issues/25796 |

**Fixture tags to cover:**
- `version-mismatch`: feed V4.1 output to the V4 parser
- `string-false-json`
- `unescaped-quote-in-string`
- `dsml-in-reasoning`
- `bare-invoke`
- `unterminated-section`
- `missing-close-param`
- `two-chunk-stream`
- `fullwidth-chars`
- `code-fence-args` (V3/R1)
- `no-separator-chain` (V3.1)
- `think-close-only`

## Fixture corpus (`fixtures/deepseek/`, 85 fixtures)

Regenerate with `.venvs/transformers/bin/python scripts/fixtures/deepseek/build.py`. Add `--check` to confirm that the committed files are up to date. `models[0]` selects the sub-format, and every fixture also carries an `x-deepseek-v3|v31|v32|v4|v41` tag.

| Sub-format | Reference (pinned) | Official renderer used | Fixtures |
|---|---|---|---|
| V3 / R1 | DeepSeek-V3-0324 `e9b33ad` | `tokenizer_config.json` Jinja. Its tool-call rendering is identical to R1-0528's, but it has no think stripping | 10 rendered + 7 vLLM tests |
| V3.1 | DeepSeek-V3.1 `c0781d0` (= V3.1-Terminus template) | Jinja, `thinking=false` | 11 rendered + 1 vLLM test |
| V3.2 | DeepSeek-V3.2 `a7e62ac` | `encoding/encoding_dsv32.py`, run as-is | 16 rendered + 1 SGLang test |
| V4 | DeepSeek-V4-Flash `60d8d70` (= V4-Pro encoder) | `encoding/encoding_dsv4.py` | 17 rendered + 1 SGLang test |
| V4.1 | DeepSeek-V4.1-Flash `dba1be0` | `encoding/encoding.py` | 19 rendered + 2 vLLM tests |

**The expected values are oracle-checked.** For V3.2, V4 and V4.1, every well-formed rendered fixture, and every imported one marked `oracle`, is fed back through DeepSeek's own `parse_message_from_completion_text`. The build fails unless that parse equals `expected` exactly. The expected values are therefore the model vendor's own parse, not any engine's.

**Coverage, per sub-format:**
- single, parallel, no call, text before the call, empty arguments
- numeric arguments, with a `"007"` string next to real numbers
- unicode
- marker-like text inside a string argument
- truncation

**Additional coverage:**
- **DSML only:**
  - thinking mode (the prompt ends with `<think>`, so the output holds only `</think>`)
  - `string="false"` JSON objects and arrays
  - unescaped `<`, `>`, `"` and `\n` in `string="true"` values
  - truncation inside reasoning, and in the second of two invokes
- **V4.1 only:** long arguments and whitespace-padded string values.
- **Multi-turn:** V3.1 (the prompt after tool output is empty), V4 and V4.1.

**Limitations:**
- **R1-0528 reasoning cannot be rendered.** Its template drops everything before `</think>`, so there are no rendered R1-0528 fixtures. V3/R1 is covered through V3-0324, which renders the same tool-call bytes.
- **V3.1 thinking mode with tools is not supported** (per the model card) and is not rendered.
- **`sglang-v4-self-closing-invoke`** follows SGLang's claim that V4 emits `<｜DSML｜invoke name="x"/>` for zero-argument tools. The official encoder never renders that form, and the reference parser rejects it. See the fixture's notes.

# Gemma 4 (and a note on Gemma 3), slug `gemma4`

**Models:**
- gemma-4-26B-A4B-it and gemma-4-31B-it (2026-03). These are Google's two most-downloaded LLMs on HF: 11.1M and 9.3M.
- gemma-4-E4B-it and E2B-it (2026-03)
- gemma-4-12B-it (2026-05)

The 31B and 12B templates are byte-identical.

**Gemma 3** (2025-03) repos are gated (`gated: manual`), so its official template was **not fetched or verified** here. Engines have no dedicated Gemma 3 tool parser. vLLM ships only a *community* `examples/tool_chat_template_gemma3_pythonic.jinja` that prompts for pythonic calls, parsed with `pythonic`. Gemma 3 should be marked **unverified** until its template is checked with HF access. If it is covered, map it to the pythonic format with provenance "vLLM example template". FunctionGemma is a separate model with its own parser, `functiongemma`.

## Format

Gemma 4 uses a **custom, non-JSON object notation** inside dedicated tokens:

```
<|tool_call>call:NAME{key:VALUE,key:VALUE}<tool_call|>
```

- **Strings** are delimited by the special token `<|"|>` on both sides. Nothing inside is escaped: `"`, `,` and `}` all appear raw.
- **Keys** in tool calls are **unquoted**, including in nested objects (`escape_keys=False`). A key containing a space or colon is emitted as-is.
- **Numbers** use Python formatting, e.g. `1.5e-05`. Booleans are `true`/`false` and null is `null`. Arrays are `[…]` and objects are `{…}`.
- **Key order** comes from Jinja `dictsort`, which is **alphabetical and case-insensitive**, not insertion order. Example: `{a:…,B:…}`.
- **Parallel calls** are consecutive `<|tool_call>…<tool_call|>` blocks with no separator.
- The template **rejects string arguments**: `tool_calls[].function.arguments must be a JSON object (mapping), not a string`.

Rendered from gemma-4-31B-it `chat_template.jinja` (rev `842da37`) with our standard two-call conversation:

```
<|turn>model
<|channel>thought
I should call the tools.
<channel|><|tool_call>call:get_weather{city:<|"|>Zürich<|"|>,unit:<|"|>c<|"|>}<tool_call|><|tool_call>call:search{filters:{max:3,tags:[<|"|>a<|"|>,<|"|>b<|"|>]},query:<|"|>café "best"<|"|>}<tool_call|><|tool_response>response:get_weather{value:<|"|>{"temp": 20}<|"|>}<tool_response|><|tool_response>response:search{value:<|"|>[]<|"|>}<tool_response|><|channel>thought
Done.
<channel|>It is 20C.<turn|>
```

Edge-value render. The input was `{"key with space": 1.5e-5, "s": "x,y} \"q\"", "flag": false, "none": null, "obj": {"B": 1, "a": [1, "t"]}}`, and the output is:

```
<|tool_call>call:f{flag:false,key with space:1.5e-05,none:null,obj:{a:[1,<|"|>t<|"|>],B:1},s:<|"|>x,y} "q"<|"|>}<tool_call|>
```

### Turn structure and stop behaviour
- **Tool results:** these stay **inside the same model turn** as `<|tool_response>response:NAME{value:…}<tool_response|>`. They do not start a new user or tool turn.
- **Stop tokens:** `generation_config.eos_token_id` is `[1, 106, 50]`, which is `<eos>`, `<turn|>` and **`<|tool_response>`**. After emitting its calls, the model generates `<|tool_response>` and stops. The raw completion therefore ends with `<tool_call|>`, plus `<|tool_response>` if stop tokens are kept.

### Reasoning
- **Enabling it:** reasoning is switched on with `enable_thinking=true`, which puts `<|think|>` in the system turn.
- **Channel syntax:** reasoning is `<|channel>thought\n…<channel|>`.
- **Generation prompt:**
  - Thinking **off** pre-fills an empty channel: `<|turn>model\n<|channel>thought\n<channel|>`.
  - Thinking **on** gives just `<|turn>model\n`, and the model opens the channel itself.
  - After a tool response with thinking on, the prompt continues **in the same turn** with `<|channel>thought\n`. The raw continuation then starts inside reasoning with no channel opener.
- **History:** reasoning is re-rendered only after the last user turn, or on tool-call turns with `preserve_thinking`.

## Special tokens (gemma-4-31B-it `tokenizer.json`, all `special: true`)

| id | token |
|---|---|
| 1 / 2 | `<eos>` / `<bos>` |
| 105 / 106 | `<\|turn>` / `<turn\|>` |
| 46 / 47 | `<\|tool>` / `<tool\|>` (tool declarations) |
| 48 / 49 | `<\|tool_call>` / `<tool_call\|>` |
| 50 / 51 | `<\|tool_response>` / `<tool_response\|>` |
| 52 | `<\|"\|>` (string delimiter) |
| 98 | `<\|think\|>` |
| 100 / 101 | `<\|channel>` / `<channel\|>` |

The asymmetric `<|x>` / `<x|>` open/close style is unique to Gemma 4. **The string delimiter `<|"|>` is a special token**, so `skip_special_tokens=True` deletes it and turns `city:<|"|>Zürich<|"|>` into `city:Zürich`. Parsers must run on text decoded with special tokens kept, or on token IDs.

## Sources

| Model | URL |
|---|---|
| gemma-4-31B-it | https://huggingface.co/google/gemma-4-31B-it/blob/842da3794eaa0b77d5f08bae87a17459d91ff475/chat_template.jinja |
| gemma-4-12B-it (identical) | https://huggingface.co/google/gemma-4-12B-it/blob/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7/chat_template.jinja |
| generation_config (stop tokens) | https://huggingface.co/google/gemma-4-31B-it/blob/842da3794eaa0b77d5f08bae87a17459d91ff475/generation_config.json |

## Engine parser mapping

| Engine | Tool parser | Reasoning parser |
|---|---|---|
| vLLM v0.34.0 | `gemma4` (`Gemma4EngineToolParser`, grammar `vllm/parser/gemma4.py`: `TOOL_CALL_START = "<\|tool_call>"`, `STRING_DELIM = '<\|"\|>'`) | `gemma4` (`Gemma4ParserReasoningAdapter`) |
| SGLang v0.5.23 | `gemma4` (`Gemma4Detector`) | `gemma4` |
| llama.cpp v0.5.0 | dedicated `COMMON_CHAT_FORMAT_PEG_GEMMA4`. Test templates: `google-gemma-4-31B-it.jinja`, `google-gemma-4-31B-it-interleaved.jinja` | same |
| Ollama v0.34.4 | `gemma4`, `gemma4-no-thinking` | built in |

**Fixture sources (Apache-2.0):**
- vLLM: `tests/tool_parsers/test_gemma4_tool_parser.py`, `tests/reasoning/test_gemma4_reasoning_parser.py`

## Known edge cases (real reports)

| Edge case | URL |
|---|---|
| Parser drops tool calls when transitioning out of the reasoning channel (`<channel\|>call:…`) | https://github.com/vllm-project/vllm/issues/54256 |
| With no reasoning parser, a synthetic `<\|channel>` leaks into content | https://github.com/vllm-project/vllm/issues/57231 |
| Streamed content ends with stripped `<\|channel>`/`<channel\|>` | https://github.com/vllm-project/vllm/issues/57232 |
| `finish_reason=tool_calls` with an empty `tool_calls` array under concurrent load | https://github.com/vllm-project/vllm/issues/50889 |
| `tool_choice="required"` not enforced: prose returned with `finish_reason="tool_calls"` | https://github.com/vllm-project/vllm/issues/53363 |
| Streaming content empty while reasoning holds everything when the channel is left open | https://github.com/vllm-project/vllm/issues/49717 |
| `<pad>` tokens under concurrent requests | https://github.com/vllm-project/vllm/issues/39392 |
| SGLang Responses API: a gemma4 tool call leaks as text under `tool_choice="auto"` | https://github.com/sgl-project/sglang/issues/28475 |
| llama.cpp: an array parameter is serialized as a JSON string when values contain `{` or `}` | https://github.com/ggml-org/llama.cpp/issues/21384 |
| llama.cpp: a Gemma 4 tool call is returned as content | https://github.com/ggml-org/llama.cpp/issues/22786 |
| llama.cpp: PEG→GBNF `until()` stops mid-delimiter, leaving calls unconstrained under `required` | https://github.com/ggml-org/llama.cpp/issues/29089 |
| Ollama: calls using `key=value` instead of `key:value` fail to parse | https://github.com/ollama/ollama/issues/17882 |
| Ollama: object keys containing spaces are left unquoted and the call is dropped | https://github.com/ollama/ollama/issues/18390 |
| Ollama: string-placeholder collision silently drops valid tool calls | https://github.com/ollama/ollama/issues/18354 |
| Ollama: a missing brace drops the call | https://github.com/ollama/ollama/issues/17562 |

**Fixture tags to cover:**
- `string-delim-token`
- `unquoted-keys`
- `key-with-space`
- `brace-in-string`
- `quote-in-string`
- `dictsort-order`
- `python-float`
- `nested-object`
- `array-of-strings`
- `call-after-channel-close`
- `stop-on-tool-response`
- `skip-special-tokens`
- `key-equals-drift`
- `unclosed-channel`

## Fixture corpus (`fixtures/gemma4/`, 48 fixtures)

Regenerate with `.venvs/transformers/bin/python scripts/fixtures/gemma4/build.py`. Add `--check` to confirm that the committed files are up to date. The reference is gemma-4-31B-it `842da37` (tokenizer files and `chat_template.jinja` only). The 26B-A4B-it and 12B-it templates are byte-identical. E2B/E4B ship a *different* template and are not listed in `models`.

| Source | Count | What |
|---|---|---|
| `template_render` | 30 | The official template, with thinking on and off. There is a history-vs-generation difference here: with thinking off, the generation prompt pre-fills `<\|channel>thought\n<channel\|>`, which a history render lacks. The output is then everything after `<\|turn>model\n`. Generation is cut at the first stop id (`<eos>`, `<turn\|>`, `<\|tool_response>`). |
| `engine_test` | 17 | vLLM `test_gemma4_tool_parser.py` and `test_gemma4_reasoning_parser.py`, and SGLang `TestGemma4Detector`, line-anchored. |
| `bug_report` | 1 | ollama#18390: an unquoted object key containing spaces. |

**Other notable renders:**
- The case-insensitive `dictsort` key order.
- Python-formatted floats (`1e-05`, `1e+21`).
- Raw newlines, quotes, braces and backslashes inside `<\|"\|>` strings.
- `<\|tool_call>…<tool_call\|>` and `<channel\|>` text inside a string argument. These become special-token ids.
- Array values whose strings contain `{`/`}` (llama.cpp#21384).
- The exact 45-strings-plus-array input from ollama#18354.

**Notes:**
- **Text position.** The template renders an assistant message's text **after** its tool calls (`text-after-call`). Models also emit text before calls; the engine tests cover that.
- **Official response template.** The `response_template` that Google ships in `tokenizer_config.json` matches names with `(?P<name>\w+)`. Hyphenated or dotted tool names (`get-weather`, `weather.get`), which are valid OpenAI names that the template renders verbatim, are therefore outside it.
- **Not included: a thinking-on continuation after a tool response.** Its generation prompt is `<\|channel>thought\n`, inside the same model turn. Fixtures carry no conversation history, so adapters cannot rebuild that prompt, and the case was dropped until the spec has a history field.

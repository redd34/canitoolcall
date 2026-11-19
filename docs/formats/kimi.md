# Kimi (K2.x and K3), slug `kimi`

**Models:**
- Kimi-K2-Instruct (2025-07) and K2-Instruct-0905 (2025-09)
- Kimi-K2-Thinking (2025-11)
- Kimi-K2.5 (2026-01) and Kimi-K2.6 (2026-04)
- Kimi-K2.7-Code (2026-06)
- **Kimi-K3 (2026-06)**, the most-downloaded Moonshot model on HF as of 2026-09-25

K2.x and K3 use **completely different** encodings, so they are two sub-formats: `kimi-k2` and `kimi-k3`.

## Sub-format A: K2.x section tokens (`kimi-k2`)

Rendered from Kimi-K2.6 `chat_template.jinja` (rev `7eb5002`), with `preserve_thinking=true` and the official tool-ID style:

```
<|im_assistant|>assistant<|im_middle|><think>I should call the tools.</think><|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "Zürich", "unit": "c"}<|tool_call_end|><|tool_call_begin|>functions.search:1<|tool_call_argument_begin|>{"query": "café \"best\"", "filters": {"tags": ["a", "b"], "max": 3}}<|tool_call_end|><|tool_calls_section_end|><|im_end|><|im_system|>get_weather<|im_middle|>## Return of functions.get_weather:0
{"temp": 20}<|im_end|><|im_system|>search<|im_middle|>## Return of functions.search:1
[]<|im_end|>
```

**The tool name is not a separate field.** It is encoded inside the **tool-call ID**, which has the form `functions.{name}:{idx}`. The official guidance, `docs/tool_call_guidance.md` in Kimi-K2-Instruct-0905, says:

> "The format of the tool ID is `functions.{func_name}:{idx}`, from which we can parse the function name."
>
> "`idx` is a global counter that starts at 0 and increments with each function invocation."

The template writes whatever `tool_call['id']` the client sends. If a client sends back an OpenAI-style `call_abc123` ID, the model sees history in the wrong format. The FAQ in the same doc says this is the usual cause of "special tokens like `<|tool_call_begin|>` in the `content` field".

**Other details:**
- **Arguments:** the JSON follows `<|tool_call_argument_begin|>` directly. Strings are passed through; dicts go through `tojson`.
- **Separators:** there are none between calls.
- **Reasoning:** the K2.5, K2.6 and K2.7-Code generation prompt is `<|im_assistant|>assistant<|im_middle|><think>`. The raw completion therefore starts **inside the reasoning** and contains only `</think>`. With `thinking=false` the prompt pre-fills `<think></think>`.
  - **Kimi-K2-Thinking is different:** its generation prompt (rev `a51ccc0`) is just `<|im_assistant|>assistant<|im_middle|>`, so the model emits `<think>` itself (verified by rendering all four templates, 2026-09-25).
  - By default (`preserve_thinking=false`), reasoning in history is replaced with `<think></think>` up to the last non-tool-call assistant message.
  - K2-Instruct/0905 have no thinking.
- **Tool declarations:** `<|im_system|>tool_declare<|im_middle|>` + `tools|tojson(separators=(',', ':'))`.

### K2 special tokens (Kimi-K2.6 `tokenizer_config.json`)

| id | token | special? |
|---|---|---|
| 163586 | `<\|im_end\|>` | yes |
| 163587 / 163588 / 163594 | `<\|im_user\|>` / `<\|im_assistant\|>` / `<\|im_system\|>` | yes |
| 163601 | `<\|im_middle\|>` | yes |
| 163595 / 163596 | `<\|tool_calls_section_begin\|>` / `<\|tool_calls_section_end\|>` | **no** |
| 163597 / 163599 | `<\|tool_call_begin\|>` / `<\|tool_call_end\|>` | no |
| 163598 | `<\|tool_call_argument_begin\|>` | no |
| 163606 / 163607 | `<think>` / `</think>` (K2.5+ only) | no |

vLLM's `KimiK2ToolParser.adjust_request` forces `skip_special_tokens = False` whenever tools are present.

## Sub-format B: K3 "XTML" (`kimi-k3`)

Kimi-K3 ships **no Jinja template**. The reference is the Python encoder `encoding_k3.py`, which renders "XTML" from three control tokens:
- `<|open|>TAG attrs<|sep|>` opens an element
- `<|close|>TAG<|sep|>` closes it
- `<|end_of_msg|>` ends a message

Rendered by running `encoding_k3.build_chat_segments` from Kimi-K3 (rev `f831ab6`):

```
<|open|>message role="assistant"<|sep|><|open|>think<|sep|>I should call the tools.<|close|>think<|sep|><|open|>response<|sep|><|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool="get_weather" index="1"<|sep|><|open|>argument key="city" type="string"<|sep|>Zürich<|close|>argument<|sep|><|open|>argument key="unit" type="string"<|sep|>c<|close|>argument<|sep|><|close|>call<|sep|><|open|>call tool="search" index="2"<|sep|><|open|>argument key="query" type="string"<|sep|>café "best"<|close|>argument<|sep|><|open|>argument key="filters" type="object"<|sep|>{"tags": ["a", "b"], "max": 3}<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|><|end_of_msg|><|open|>message role="tool" tool="get_weather" index="1"<|sep|>{"temp": 20}<|close|>message<|sep|><|end_of_msg|><|open|>message role="tool" tool="search" index="2"<|sep|>[]<|close|>message<|sep|><|end_of_msg|>
```

**Details:**
- **Element order:** every assistant message is `think` → `response` → optional `tools`, in that fixed order. In thinking mode the `think` element is always present, even when empty.
- **Generation prompt:** `<|open|>message role="assistant"<|sep|><|open|>think<|sep|>`, or `…<|open|>response<|sep|>` when `thinking=False`. The raw completion starts inside `think`.
- **Argument types:** each argument carries a `type` attribute (`string|number|boolean|null|object|array`).
  - String values are raw, with no escaping.
  - Other values are `json.dumps(…, ensure_ascii=False)`.
  - Attribute values escape `&` → `&amp;` and `"` → `&quot;` (`_escape_attr_value`).
- **Call index:** `index` is **1-based** and per message.
- **Non-object arguments:** if the arguments string is not a JSON object, the encoder falls back to `<|open|>json type="object"<|sep|>RAW<|close|>json<|sep|>` (`_xtml_json_block`).
- **Control tokens:** the encoder uses `EncodeSegment(allow_special=True)` so that structural markers become special tokens while user text cannot inject them.

### K3 special tokens (Kimi-K3 `tokenizer_config.json`)

| id | token |
|---|---|
| 163586 | `<\|end_of_msg\|>` (`generation_config.eos_token_id`) |
| 163587 | `<\|open\|>` |
| 163588 | `<\|close\|>` |
| 163589 | `<\|sep\|>` (marked `special: false`) |

Note that K3 **re-used ids 163586–163588**. In K2 they were `<|im_end|>`, `<|im_user|>` and `<|im_assistant|>`.

## Sources

| Model | URL |
|---|---|
| Kimi-K2.6 template | https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/chat_template.jinja |
| Kimi-K2-Instruct-0905 template | https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905/blob/ac6c49f04883bd0a0598b790693a72061c676629/chat_template.jinja |
| K2 tool-call guidance (ID format, manual parser) | https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905/blob/ac6c49f04883bd0a0598b790693a72061c676629/docs/tool_call_guidance.md |
| Kimi-K3 encoder | https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/encoding_k3.py |

## Engine parser mapping

| Sub-format | vLLM v0.34.0 tool / reasoning | SGLang v0.5.23 tool / reasoning |
|---|---|---|
| K2.x | `kimi_k2` (`KimiK2ToolParser` → `vllm/parser/kimi_k2.py`, `_TOOL_ID_RE = r"(?P<id>.+:\d+)"`) / `kimi_k2` | `kimi_k2` (`KimiK2Detector`) / `kimi_k2` (also `kimi` for the older Kimi-VL/1.5 `◁think▷` style) |
| K3 | `kimi_k3` (`KimiK3ToolParser`) / `kimi_k3` | `kimi_k3` (`KimiK3Detector`, with structural-tag support in `kimik3_structural_tag.py`) / `kimi_k3` |
| (related) | `k2_horizon` | `k2_horizon` (`K2V3Detector`) |

Other engines:
- **llama.cpp v0.5.0:** PEG, from template. Test templates: `moonshotai-Kimi-K2.jinja`, `Kimi-K2-Instruct.jinja`, `Kimi-K2-Thinking.jinja`, `Kimi-K3.jinja`. The K3 one is llama.cpp-authored.
- **Ollama v0.34.4:** no dedicated Kimi parser in `model/parsers/`. Kimi is used mainly via `:cloud` models.

**Fixture sources:**
- vLLM (Apache-2.0): `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `test_kimi_k3_named_tool_choice.py`, `tests/reasoning/test_kimi_k2_reasoning_parser.py`, `test_kimi_k3_reasoning_parser.py`
- SGLang (Apache-2.0): `test/registered/function_call/test_kimik2_detector.py`, `test_kimik3_detector.py`, `test/registered/unit/parser/test_kimik3_reasoning_parser.py`
- Moonshot's K2-Vendor-Verifier (594★) is a related external conformance effort for this family. Check its license before borrowing.

## Known edge cases (real reports)

| Edge case | URL |
|---|---|
| Kimi 2.6 + `kimi_k2` parser passes malformed JSON arguments to the client without validation | https://github.com/vllm-project/vllm/issues/41739 |
| `stream_interval 4` + `kimi_k2` reasoning parser misbehaves, so results depend on chunking | https://github.com/vllm-project/vllm/issues/45866 |
| Responses API returns a raw malformed Kimi tool marker as assistant text | https://github.com/vllm-project/vllm/issues/50768 |
| `kimi_k3` streaming classifies a response-only completion as reasoning, diverging from non-streaming | https://github.com/vllm-project/vllm/issues/57688 |
| `kimi_k3` non-streaming classifies truncated reasoning as content | https://github.com/vllm-project/vllm/issues/57353 |
| `kimi_k3`: structured output never engages when the completion skips the think channel | https://github.com/vllm-project/vllm/issues/57714 |
| SGLang `KimiK2Detector` fails on `callNNNNN` IDs in multi-turn | https://github.com/sgl-project/sglang/issues/25358 |
| SGLang `KimiK2Detector` streaming drops or hangs on long multi-turn arguments | https://github.com/sgl-project/sglang/issues/23363 |
| K2-Thinking drops optional parameters after the reasoning phase | https://github.com/sgl-project/sglang/issues/12932 |
| Kimi-K3 strict tool-call grammar: `additionalProperties` dilutes a property's type | https://github.com/sgl-project/sglang/issues/38587 |
| Kimi-K3 cross-prompt reasoning leakage | https://github.com/sgl-project/sglang/issues/34259 |
| llama.cpp tool-call grammar generator emits GBNF that fails its own parser (Kimi K2.7-Code, 22 tools) | https://github.com/ggml-org/llama.cpp/issues/24658 |
| Kimi K2.5 cannot call tools via Ollama in OpenClaw | https://github.com/ollama/ollama/issues/14592 |

**Fixture tags to cover:**
- `id-encodes-name`
- `non-kimi-call-id`
- `global-idx`
- `think-no-open-tag`
- `empty-think`
- `xtml-attr-escaping`
- `xtml-json-fallback`
- `typed-args`
- `skip-think-channel`
- `stream-interval`
- `parallel`

## Fixtures in this repo

`fixtures/kimi/` (49 fixtures) is built by `uv run --script scripts/fixtures/kimi/build.py`. The build is deterministic and downloads only tokenizer and template files at pinned revisions.

| File | Source | What |
|---|---|---|
| `k26-render.jsonl` | Kimi-K2.6 `chat_template.jinja` @ `7eb5002` | K2 section tokens with thinking (`preserve_thinking=true`, so the rendered turn keeps its reasoning) |
| `k2i-render.jsonl` | Kimi-K2-Instruct-0905 `chat_template.jinja` @ `ac6c49f` | K2 without thinking (also emitted by Kimi-K2-Instruct: the render is identical) |
| `k3-render.jsonl` | Kimi-K3 `encoding_k3.py` @ `f831ab6`, called by the K3 tokenizer | XTML; control tokens vs. plain-text tokens exactly as the encoder produces them |
| `truncated.jsonl` | token prefixes of the renders above | what `max_tokens` produces |
| `imported.jsonl` | vLLM v0.30.0 tests (Apache-2.0) and vLLM issues #57353 and #57688 | line-anchored quotes |

Notes:
- **Siblings.** Every render is compared against the sibling templates (K2.5, K2-Thinking, K2.7-Code), and a sibling is listed in `models` only if all its renders are identical. None of them qualified. K2.5 has no `preserve_thinking`, so its history render drops the reasoning of a plain-text turn. K2-Thinking emits `<think>` itself (see above).
- **Tool-call ids** follow Moonshot's guidance: `functions.{name}:{idx}`, with a global counter. The multi-turn fixture's second call is therefore `:1`.
- **Markers inside arguments.** K2 tokenizes a marker written inside a JSON string as the marker token (`kimi/k26-marker-terminator-in-string` contains the `<|tool_call_end|>` id inside the string). K3 encodes argument text with `allow_special=False`, so `kimi/k3-control-marker-text-in-value` has the *text* `<|close|>` as ordinary tokens. Only `output_token_ids` can tell it apart from the real marker.
- **The K3 no-tools request** (`kimi/k3-bug-truncated-reasoning-recorded`) has `tools: []`, as in the issue. vLLM rejects an empty `tools` array, so adapters should omit the field when it is empty.


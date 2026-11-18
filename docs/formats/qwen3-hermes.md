# Qwen3 (Hermes-style JSON), slug `qwen3-hermes`

**Models:**
- Qwen3-0.6B/1.7B/4B/8B/14B/32B/30B-A3B/235B-A22B (2025-04, hybrid thinking)
- Qwen3-*-Instruct-2507 (non-thinking)
- Qwen3-*-Thinking-2507 (thinking only)
- Qwen2.5-*-Instruct, which uses the same tool format without `<think>`

**Not in this family:** Qwen3-Coder and Qwen3.5 and later switched to the XML format. See [`qwen3-xml`](qwen3-xml.md).

## Format

Tool calls are JSON objects wrapped in `<tool_call>`…`</tool_call>`:
- one wrapper per call
- consecutive calls separated by `\n`
- JSON on its own line
- key order `name`, then `arguments`

The template inserts `arguments` as-is if it is a string, and as `tojson` if it is a mapping. `tojson` in HF transformers uses `ensure_ascii=False`, so `ü` stays literal. Reasoning comes first, in `<think>\n…\n</think>\n\n`.

Rendered from the official template (Qwen3-8B rev `b968826`):

```
<|im_start|>assistant
<think>
I should call the tools.
</think>

<tool_call>
{"name": "get_weather", "arguments": {"city": "Zürich", "unit": "c"}}
</tool_call>
<tool_call>
{"name": "search", "arguments": {"query": "café \"best\"", "filters": {"tags": ["a", "b"], "max": 3}}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"temp": 20}
</tool_response>
<tool_response>
[]
</tool_response><|im_end|>
```

**Details that matter for parsers:**
- If the assistant message has text content, the template emits `content` and then `\n<tool_call>…`. Text *before* the first call is normal; the system prompt forbids nothing after.
- Tool results come back in a **`user`** turn wrapped in `<tool_response>`, not in a `tool` role.
- The system prompt tells the model the format: `For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>`.

**Thinking modes. The generation prompt differs by variant, and reasoning parsers must handle both:**
- **Qwen3 hybrid (2504):** the generation prompt is `<|im_start|>assistant\n`. The model itself emits `<think>\n`. With `enable_thinking=false` the template pre-fills `<think>\n\n</think>\n\n`.
- **Qwen3-*-Thinking-2507:** the generation prompt is `<|im_start|>assistant\n<think>\n`. The raw completion therefore has **no opening `<think>`**: it starts inside the reasoning and contains only `</think>`.
- **Qwen3-*-Instruct-2507:** the template has no `<think>` handling (0 occurrences).
- **History:** the template strips `<think>` from assistant turns before the last user query. It keeps reasoning only for turns after the last user message, which is multi-step tool use.

## Special tokens (Qwen3-8B `tokenizer_config.json`)

| id | token | special? |
|---|---|---|
| 151644 | `<\|im_start\|>` | yes |
| 151645 | `<\|im_end\|>` (EOS) | yes |
| 151657 / 151658 | `<tool_call>` / `</tool_call>` | **no**: survives `skip_special_tokens=True` |
| 151665 / 151666 | `<tool_response>` / `</tool_response>` | no |
| 151667 / 151668 | `<think>` / `</think>` | no |

Because the tool and think markers are non-special added tokens, they are single token IDs but detokenize as plain text. A parser sees them whether or not special tokens are skipped. A per-token stream never splits them, but a per-character or re-chunked stream can.

## Template sources

| Model | File | URL |
|---|---|---|
| Qwen3-8B (hybrid) | `tokenizer_config.json` → `chat_template` | https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/tokenizer_config.json |
| Qwen3-4B-Thinking-2507 | `tokenizer_config.json` | https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507/blob/768f209d9ea81521153ed38c47d515654e938aea/tokenizer_config.json |
| Qwen3-4B-Instruct-2507 | `tokenizer_config.json` | https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/tokenizer_config.json |
| Qwen2.5-7B-Instruct | `tokenizer_config.json` | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/a09a35458c702b33eeacc393d103063234e8bc28/tokenizer_config.json |

## Engine parser mapping

| Engine | Tool parser | Reasoning parser | Notes |
|---|---|---|---|
| vLLM v0.34.0 | `hermes` (`vllm/tool_parsers/hermes_tool_parser.py`, `Hermes2ProToolParser`); the new name `qwen3_json` is an alias of the same class (its registry comment marks a dedicated parser as TODO) | `qwen3` (`Qwen3ParserReasoningAdapter`); `deepseek_r1` also works for Thinking-2507, where there is no opening tag | The Qwen3-8B model card mentions `--reasoning-parser qwen3` and `deepseek_r1` |
| SGLang v0.5.20 | `qwen25` / `qwen` (`Qwen25Detector`, bot token `"<tool_call>\n"`, eot `"\n</tool_call>"`); `hermes` (`HermesDetector`, bot `"<tool_call>"`) | `qwen3`, `qwen3-thinking` (`Qwen3Detector`) | `qwen25` hard-codes the newlines from the template |
| llama.cpp v0.5.0 | Automatic, from template (PEG). Test template: `models/templates/Qwen-Qwen3-0.6B.jinja` | same | |
| Ollama v0.34.4 | `qwen3`, `qwen3-thinking` | built in | |

**Fixture sources in engine test suites (Apache-2.0):**
- `vllm/tests/tool_parsers/test_hermes_tool_parser.py`
- `sglang/test/registered/unit/function_call/test_hermes_detector.py`

## Known edge cases (real reports)

| Edge case | URL |
|---|---|
| Hermes streaming drops `{}` arguments for parameterless tools | https://github.com/vllm-project/vllm/issues/28806 |
| Hermes streaming output error in the Qwen3 case | https://github.com/vllm-project/vllm/issues/19056 |
| Streaming with `hermes` returns raw text instead of `tool_calls` | https://github.com/vllm-project/vllm/issues/31871 |
| Non-streaming leaks raw `<tool_call>` markup into content when a call is truncated by `max_tokens` (hermes, qwen25) | https://github.com/sgl-project/sglang/issues/30480 |
| Parallel multi-call streaming fails with `tool_choice="auto"` on Qwen3-Thinking | https://github.com/sgl-project/sglang/issues/9654 |
| `array<object>` parameter values fail `p.json()`; a partial tool_call leaks to the client and poisons history | https://github.com/ggml-org/llama.cpp/issues/21771 |
| qwen3 parser returns HTTP 500 when output is truncated | https://github.com/ollama/ollama/issues/14570 |

**Fixture tags to cover:**
- `parallel`
- `text-before-call`
- `empty-args` (`"arguments": {}`)
- `unicode`
- `nested-json`
- `truncated-in-json`
- `think-no-open-tag` (Thinking-2507)
- `marker-in-string` (an argument containing the literal text `</tool_call>`)

## Fixture corpus (`fixtures/qwen3-hermes/`, 50 fixtures)

Regenerate with `.venvs/transformers/bin/python scripts/fixtures/qwen3-hermes/build.py`. Add `--check` to confirm that the committed files are up to date. The script downloads only tokenizer and template files, at these pinned revisions:
- Qwen3-0.6B `c1899de` (hybrid)
- Qwen3-4B-Thinking-2507 `768f209`
- Qwen3-4B-Instruct-2507 `cdbee75`

Their templates are byte-identical to the 8B/32B/30B-A3B/235B siblings, which each fixture lists in `models`.

| Source | Count | What |
|---|---|---|
| `template_render` | 36 | Official template, `apply_chat_template(tokenize=True)`. Output ids are cut at the first `generation_config` stop id. Covers thinking on and off (`enable_thinking=false` pre-fills an empty think block), Thinking-2507 (the prompt pre-fills `<think>\n`), and Instruct-2507. Truncated fixtures are a token prefix of a render. |
| `engine_test` | 12 | vLLM `test_hermes_tool_parser.py` and `test_qwen3_reasoning_parser.py`, and SGLang `test_hermes_detector.py` and `TestQwen25Detector` (Apache-2.0, line-anchored). This includes the no-newline `<tool_call>{…}</tool_call>` form and two malformed outputs. |
| `bug_report` | 2 | sglang#30480: truncation mid-arguments, and truncation at the `<tool_call>` opener. |

**Conventions:**
- **`expected`** holds the fields the template was *given*. The `\n` the template adds inside `<think>` and before `<tool_call>` is not part of them, so engines that keep it get `soft_pass`.
- **`marker-in-arguments`** puts literal `<tool_call>…</tool_call>` and `<think>` text inside a JSON string. The tokenizer maps these to their added-token ids. The JSON is still open at the inner `</tool_call>`, so a correct parser keeps the whole string.
- **Truncation.** A call cut by `max_tokens` must yield no call (`expected_error`, with accept `no_tool_calls`/`content_passthrough`). `truncated-second-parallel-call` keeps the first, complete call (the fix direction of sglang#30480). A cut inside `<think>` is reasoning, not content (this matches vLLM's own qwen3 reasoning tests).
- **Multi-turn fixtures** carry only the generation prompt, not the conversation history. The spec has no history field yet, and for Qwen3 the generation prompt after a tool response is the same as after a user turn.

# ByteDance Seed-OSS, slug `seed-oss`

**Models (ByteDance-Seed):**
- **Seed-OSS-36B-Instruct** (2025-08), the reference model: rev `497f1dca95ebdec98e41d517b9f060ee753c902f`, Apache-2.0
- Seed-OSS-36B-Base and Seed-OSS-36B-Base-woSyn are base models with no tool-call turns, so they are not listed in `models`

## Format

Each call is wrapped in its own `<seed:tool_call>` … `</seed:tool_call>` pair:

```
<seed:tool_call>
<function=NAME>
<parameter=KEY>VALUE</parameter>
…
</function>
</seed:tool_call>
```

This is the Qwen3-Coder XML grammar (see [`qwen3-xml.md`](qwen3-xml.md)) with three differences that matter to a parser:

1. **The wrapper and reasoning tags are `seed:` tokens:** `<seed:tool_call>`, `</seed:tool_call>`, `<seed:think>`, `</seed:think>`. They are added tokens (ids 7, 8, 3, 4). `<function=`, `<parameter=` and their closes are plain text.
2. **Values have no wrapping newlines.** The template writes `"<parameter=" + name + ">"` and then `value + "</parameter>\n"`, and the system prompt's format example shows a multi-line value with no newline after `>` or before `</parameter>`. A newline right before `</parameter>` is therefore part of the value (a file that ends in a newline). Qwen3-Coder writes `<parameter=K>\nV\n</parameter>`, and parsers built for it strip one newline on each side.
3. **Non-string values are Python `str()`, not JSON.** The template converts every non-string value with Jinja's `| string` filter: `7`, `0.8`, `True`, `False`, `{'theme': 'dark', 'notify': True}`, `['a', 'b']`. Strings are raw and unescaped. A parser must use the tool schema to type a value, and must accept Python literals for booleans, objects and arrays. The vLLM parser that ByteDance contributed with the model ([vllm#23241](https://github.com/vllm-project/vllm/pull/23241)) did exactly that: it lower-cased booleans and fell back to `ast.literal_eval` for objects and arrays ([`seed_oss_tool_parser.py` L180-L206 at v0.10.2](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/entrypoints/openai/tool_parsers/seed_oss_tool_parser.py#L180-L206)). llama.cpp's test suite also feeds a JSON array (`[{"item": …, "selected": false}]`), so both forms are covered by fixtures.

Parallel calls are separate `<seed:tool_call>` blocks joined by `\n`. Content (if any) comes before the first call. The template has no way to write text after a call.

### Rendered example

The standard conversation from [`README.md`](README.md#3-how-the-examples-were-produced-reproducible) (system message, user, assistant with reasoning and two parallel calls, two tool results, final answer), rendered with `transformers==5.17.0` `apply_chat_template(tokenize=False)` at rev `497f1dc`. The tool definitions are abbreviated in the prompt but not in the render:

```
<seed:bos>system
You are helpful.

Function:
def get_weather(city: str,unit: str):
    """
    Get the weather.

    Args:
    - city (str) [必填]: 
    - unit (str) [选填]: 

    """

Function:
def search(query: str,filters: Any):
    """
    Search.

    Args:
    - query (str) [必填]: 
    - filters (Any) [选填]: 

    """
工具调用请遵循如下格式:
<seed:tool_call>
<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>This is the value for the second parameter
that can span
multiple lines</parameter>
</function>
</seed:tool_call>
<seed:eos><seed:bos>user
Weather in Zürich and best cafés?<seed:eos><seed:bos>assistant
<seed:think>I should call the tools.</seed:think>
<seed:tool_call>
<function=get_weather>
<parameter=city>Zürich</parameter>
<parameter=unit>c</parameter>
</function>
</seed:tool_call>
<seed:tool_call>
<function=search>
<parameter=query>café "best"</parameter>
<parameter=filters>{'tags': ['a', 'b'], 'max': 3}</parameter>
</function>
</seed:tool_call><seed:eos><seed:bos>tool
sunny<seed:eos><seed:bos>tool
[...]<seed:eos><seed:bos>assistant
<seed:think>Summarize.</seed:think>
It is sunny.<seed:eos>
```

Note `{'tags': ['a', 'b'], 'max': 3}` for the object argument (Python `str()` of the dict).

### Reasoning, the generation prompt and the thinking budget

- **Default (`thinking_budget` unset, i.e. -1):** the generation prompt is `<seed:bos>assistant\n`. It pre-fills nothing; the model opens `<seed:think>` itself. The model card says "If no thinking budget is set (default mode), Seed-OSS will initiate thinking with unlimited length", so a default-mode generation always starts with `<seed:think>`, and every rendered fixture in default mode has reasoning.
- **`thinking_budget=N` (N > 0):** the template adds a second system message with the budget, and the model writes progress notes inside its reasoning: `<seed:cot_budget_reflect>I have used 129 tokens, and there are 383 tokens remaining for use.</seed:cot_budget_reflect>` (model card, "Thinking Budget"). These notes are **part of `reasoning_content`**; vLLM's own test asserts that they stay in the reasoning ([`tests/parser/engine/test_seed_oss.py` L207-L235](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/tests/parser/engine/test_seed_oss.py#L207-L235)).
- **`thinking_budget=0`:** the generation prompt pre-fills a closed reasoning block, so the completion is the answer:

  ```
  <seed:bos>assistant
  <seed:think>
  <seed:cot_budget_reflect>The current thinking budget is 0, so I will directly start answering the question.</seed:cot_budget_reflect>
  </seed:think>
  ```

  The template has no `enable_thinking` switch. There are no `thinking_budget=0` fixtures yet: the fixture spec's `thinking` flag is a boolean, and the adapters translate it into `enable_thinking`/`thinking`, which this template ignores, so a replay would not reproduce what a client that sends `chat_template_kwargs: {"thinking_budget": 0}` gets.

In a history render, the template writes `"\n" + <seed:think> + reasoning|trim + </seed:think>`, then `"\n" + content|trim + "\n"` before the calls (or `"\n" + content|trim + <seed:eos>` without calls). Fixtures expect the message's content without those separator newlines, as for `glm` and `qwen3-xml`; a parser that keeps them gets `soft_pass`.

## Special tokens

| Token | Id | Role |
|---|---|---|
| `<seed:bos>` | 0 | turn start (`<seed:bos>role\n`), special |
| `<seed:pad>` | 1 | padding, special |
| `<seed:eos>` | 2 | turn end and the **only stop token** (`generation_config.json` `eos_token_id: 2`), special |
| `<seed:think>`, `</seed:think>` | 3, 4 | reasoning, added (not special) |
| `<seed:cot_budget_reflect>`, `</seed:cot_budget_reflect>` | 5, 6 | budget notes inside reasoning, added (not special) |
| `<seed:tool_call>`, `</seed:tool_call>` | 7, 8 | one call, added (not special) |

From `tokenizer_config.json` `added_tokens_decoder` at rev `497f1dc`. Because the `seed:` tags are not special, `skip_special_tokens=True` keeps them in the text.

## Template sources

| Model | Revision | `chat_template.jinja` sha256 |
|---|---|---|
| [`ByteDance-Seed/Seed-OSS-36B-Instruct`](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct/blob/497f1dca95ebdec98e41d517b9f060ee753c902f/chat_template.jinja) | `497f1dca95ebdec98e41d517b9f060ee753c902f` (2025-08-26) | `3f9e0dc4adeb8032287d0129dc75b6298fee8698a52ab41e863dbc89929330d3` |

llama.cpp's copy, [`models/templates/ByteDance-Seed-OSS.jinja` at `a25c9865`](https://github.com/ggml-org/llama.cpp/blob/a25c9865fe03c954c93fd755b5d79ae86ba99750/models/templates/ByteDance-Seed-OSS.jinja), is byte-identical (same sha256).

## Engine parser mapping

| Engine (pin) | Tool parser | Reasoning parser |
|---|---|---|
| vLLM v0.30.0 (`ced6857`) | `seed_oss` → `SeedOssEngineToolParser` ([`vllm/tool_parsers/__init__.py` L193-L196](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/tool_parsers/__init__.py#L193-L196)) | `seed_oss` → `SeedOssParserReasoningAdapter` ([`vllm/reasoning/__init__.py` L143-L146](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/reasoning/__init__.py#L143-L146)) |
| SGLang v0.5.20 (`94602c9`) | **none**: no Seed entry in `ToolCallParserEnum` ([`function_call_parser.py` L72-L112](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/function_call/function_call_parser.py#L72-L112)) | **none**: no Seed entry in `DetectorMap` ([`reasoning_parser.py` L2164-L2196](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/parser/reasoning_parser.py#L2164-L2196)) |
| llama.cpp `a25c9865` | the generic autoparser, derived from the chat template (`common/chat.cpp` has no Seed-specific handler). Tested in [`tests/test-chat.cpp` L3433-L3538](https://github.com/ggml-org/llama.cpp/blob/a25c9865fe03c954c93fd755b5d79ae86ba99750/tests/test-chat.cpp#L3433-L3538). `convert_hf_to_gguf` converts `SeedOssForCausalLM` ([`conversion/olmo.py` L42-L45](https://github.com/ggml-org/llama.cpp/blob/a25c9865fe03c954c93fd755b5d79ae86ba99750/conversion/olmo.py#L42-L45)) | same |
| Ollama `7af39318` | **none**: no Seed parser in `ParserForName` ([`model/parsers/parsers.go` L49-L108](https://github.com/ollama/ollama/blob/7af393188defd52d370464de0d2064649cab9b41/model/parsers/parsers.go#L49-L108)) | none |
| HF transformers 5.17.0 | **none**: the repo's `tokenizer_config.json` ships no `response_template` | none |

**vLLM's grammar.** `vllm/parser/seed_oss.py` ([source](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/parser/seed_oss.py)) subclasses `Qwen3Parser` and overrides only the four wrapper strings and the turn boundaries. Its docstring says "`<function=...>` and `<parameter=...>` are byte-identical, so the entire transition table and `_qwen3_arg_converter` are inherited from `Qwen3Parser` unchanged." That inherits `_trim_wrapping_newlines` ([`vllm/parser/qwen3.py` L60-L66](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/parser/qwen3.py#L60-L66)), which strips one leading and one trailing newline from every value, the Qwen template's markup but data in Seed-OSS (see point 2 above). The parser starts in the reasoning state unless `chat_template_kwargs.enable_thinking` is false.

**Official recommendation:** the model card serves vLLM with `--enable-auto-tool-choice --tool-call-parser seed_oss` and the repo's `chat_template.jinja`. It names no reasoning parser; the matrix uses vLLM's `seed_oss` reasoning parser, the one registered for the model.

## Results at the pins

Local replays on 2026-11-19 (macOS arm64, `--env HF_HUB_OFFLINE=1`), not a nightly snapshot. SGLang, Ollama and transformers report every fixture `unsupported` (no parser).

**vLLM 0.30.0:** `seed-oss pass=19 soft_pass=3 fail=8`. The failures, each checked against the fixture:

- `trailing-newline-value`, `llamacpp-quotes-and-trailing-newlines`: the newline before `</parameter>` is stripped (`_trim_wrapping_newlines`), so the written file loses its final newline.
- `marker-in-arguments`: `<seed:tool_call>` inside a string value is matched as a marker (the same class of bug as the other families' marker fixtures).
- `nested-python-repr`: the object and array come back as strings (`"{'theme': 'dark', …}"`). `boolean-python-repr` (`True`) passes.
- `truncated-in-parameter-value`, `truncated-before-close-function`, `truncated-before-close-tool-call`, `truncated-second-parallel-call`: an unfinished call is returned (the `truncation_policy` findings of the [recovery contract](recovery.md#2-truncated-before-the-outer-close-max_tokens), as for `qwen3-xml`).

The `soft_pass` cells keep the separator newline after `</seed:think>` or around content in `content`.

**llama.cpp `a25c9865`** (vocab-only GGUF from `scripts/engines/gguf_vocab.sh`, template from the GGUF, format `peg-native`): `seed-oss pass=23 soft_pass=1 fail=6`.

- `boolean-python-repr`, `nested-python-repr`: non-streaming raises `The model produced output that does not match the expected peg-native format`; streams return a partial call with invalid JSON arguments before that. The PEG grammar accepts only JSON for non-string values, so the template's own `True` and `{'…': …}` forms are rejected.
- The same four truncation fixtures as vLLM return an unfinished call.
- It keeps trailing newlines and handles `marker-in-arguments`.

## Fixtures in this repo

All fixtures are generated by [`scripts/fixtures/seed-oss/build.py`](../../scripts/fixtures/seed-oss/build.py) (`uv run --script`), deterministically:

- `rendered.jsonl`: 20 `template_render` fixtures from the official template, default thinking mode (one, `budget-reflect-in-reasoning`, with `thinking_budget=512`).
- `truncated.jsonl`: 6 token-prefix cuts of those renders (`truncated`), labelled by section 2 of the recovery contract.
- `imported.jsonl`: 4 `engine_test` fixtures copied verbatim from llama.cpp's `tests/test-chat.cpp` (MIT), via `imported.py`.

Tags: `x-python-repr` marks values in the template's Python `str()` form, `x-significant-whitespace` marks values whose leading or trailing whitespace is data, `x-thinking-budget` marks the budget render.

**Not covered yet:** `thinking_budget=0` (see above), malformed outputs (no Seed-OSS bug report with a raw output was found), and recorded generations: every rendered fixture is a history render, and whether the model itself writes JSON or Python literals for non-string values has not been checked against a real generation.

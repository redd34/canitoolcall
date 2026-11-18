# NVIDIA Nemotron 3 and 3.5, slug `nemotron`

**Models** (Hugging Face `nvidia/` org, by downloads on 2026-11-19):
- NVIDIA-Nemotron-3-Nano-4B-BF16 (2026-03, the most-downloaded Nemotron 3/3.5 text model) and Nemotron-3-Nano-30B-A3B (2025-12)
- NVIDIA-Nemotron-3-Super-120B-A12B (2026-03)
- NVIDIA-Nemotron-3.5-Lightning-30B-A3B (2026-08)
- NVIDIA-Nemotron-3-Ultra-550B-A55B (2026-06)

All five are `NemotronHForCausalLM` (`model_type: nemotron_h`) and share one vocabulary. **Not covered:**
NVIDIA-Nemotron-3.5-Bolt (`Gemma4ForConditionalGeneration`, a Gemma 4 style `<|tool_call>call:NAME{…}<tool_call|>`
template; see [`gemma4.md`](gemma4.md)), the image-text NVIDIA-Nemotron-3.5-Super-VL-120B-A12B (template not checked),
and the earlier Nemotron Nano v2 (`<TOOLCALL>` JSON).

**Why a family of its own.** The tool-call block is exactly the Qwen3-Coder XML format of [`qwen3-xml`](qwen3-xml.md),
and every engine reuses its Qwen3-Coder tool parser. The reasoning framing is not Qwen's (`</think>\n` vs
`</think>\n\n`, an empty `<think></think>` when thinking is off), the vocabulary is different, and vLLM, SGLang and
Ollama each select a Nemotron-specific reasoning parser or wrapper. A separate slug keeps those results apart.

## Format

The tool-call block is identical to `qwen3-xml`:
- `<tool_call>` + `\n` + `<function=NAME>` + `\n`
- zero or more `<parameter=KEY>` + `\n` + VALUE + `\n` + `</parameter>` + `\n`
- `</function>` + `\n` + `</tool_call>` + `\n`

Parallel calls are consecutive blocks; each ends with its own `\n`, so the last call is followed by `\n` and then the
stop token `<|im_end|>`.

**Values are raw text, not JSON.** Strings are inserted verbatim (quotes, `<`, newlines). Objects and arrays are
`tojson`. Every other value goes through Jinja `| string`, so **`false`/`true`/`null` render as the Python literals
`False`/`True`/`None`**, like the older Qwen3-Coder template. A parser needs the tool's JSON schema to coerce a value
back (`3` is an integer only if the schema says so; `None` is `null` for an `["integer", "null"]` schema).

Rendered with transformers 5.17.0 `apply_chat_template` from **NVIDIA-Nemotron-3-Nano-4B-BF16** (rev `dfaf35d`),
the conversation from [README.md §3](README.md#3-how-the-examples-were-produced-reproducible) (the system prompt and
the tool list are omitted here):

```
<|im_start|>user
Weather in Zürich and find cafés.<|im_end|>
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
</tool_call>
<|im_end|>
<|im_start|>user
<tool_response>
{"temp": 20}
</tool_response>
<tool_response>
[]
</tool_response>
<|im_end|>
<|im_start|>assistant
<think>
Summarize.
</think>
It is 20 °C.<|im_end|>
```

Nemotron-3-Nano-30B-A3B (same template file) and Nemotron-3-Super-120B-A12B render the same text. Super's template
only adds a `low_effort` kwarg that appends `{reasoning effort: low}` to the last user message.

**NVIDIA-Nemotron-3.5-Lightning-30B-A3B** (rev `a9904d2`) renders the same calls, but reasoning closes without the
newline before `</think>`, and a final answer follows `</think>` directly:

```
<|im_start|>assistant
<think>
I should call the tools.</think>
<tool_call>
<function=get_weather>
…
<|im_start|>assistant
<think>
Summarize.</think>It is 20 °C.<|im_end|>
```

Nemotron-3-Ultra-550B-A55B renders like Lightning. Its template only adds a `medium_effort` kwarg (appends
`{reasoning effort: efficient}`).

| Variant (`family.json`) | Repos | Reasoning before a call | Reasoning before an answer |
|---|---|---|---|
| `nemotron-3` | Nano-4B, Nano-30B-A3B, Super-120B-A12B | `R\n</think>\n<tool_call>` | `R\n</think>\nANSWER` |
| `nemotron-3.5` | Lightning-30B-A3B, Ultra-550B-A55B | `R</think>\n<tool_call>` | `R</think>ANSWER` |

Text content before a call is written as `CONTENT\n<tool_call>…`.

### Reasoning
- **Thinking is on by default** (`enable_thinking` defaults to true). The generation prompt is
  `<|im_start|>assistant\n<think>\n`, so a completion starts **inside** the reasoning, with no opening tag.
- **`enable_thinking=false`:** the generation prompt is `<|im_start|>assistant\n<think></think>`. A call then starts
  with `\n<tool_call>`, and an answer starts directly after `</think>`.
- **History:** `truncate_history_thinking` (default true) drops the reasoning of assistant turns before the last user
  message. The reasoning of tool-calling turns after it is kept.
- The tool instructions in the system prompt say: "You may provide optional reasoning for your function call in
  natural language BEFORE the function call, but NOT after".

## Special tokens (all five repos)

| id | token | special? |
|---|---|---|
| 2 | `</s>` (`generation_config.json` `eos_token_id`) | yes |
| 10 | `<\|im_start\|>` | yes |
| 11 | `<\|im_end\|>` (tokenizer `eos_token`; in `eos_token_id` except for Nano-4B) | yes |
| 12 / 13 | `<think>` / `</think>` | no |
| 14 / 15 | `<tool_call>` / `</tool_call>` | no |
| 16 / 17 | `<tool_response>` / `</tool_response>` | no |

`<function=`, `<parameter=`, `</parameter>` and `</function>` are **not** single tokens (`<function=` is
`[1060, 5165, 1061]`), so they can be split across stream deltas.

**Stop tokens.** Nano-4B's `generation_config.json` lists only `eos_token_id: 2` (`</s>`), while its tokenizer's
`eos_token` is `<|im_end|>`, the token that ends every assistant turn. The other four list `[2, 11]`. The fixtures cut
at either.

## Template sources

| Model | URL | `template_sha256` |
|---|---|---|
| Nemotron-3-Nano-4B-BF16 | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/blob/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f/chat_template.jinja | `ab7813c3abdd9cb655905a410728b26c7884eca45ddfab8d9f931553485a7862` |
| Nemotron-3-Nano-30B-A3B-BF16 | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/bf77c3174f68ad409e1c2aa60daeb46e32d1c606/chat_template.jinja | same file as Nano-4B |
| Nemotron-3-Super-120B-A12B-BF16 | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16/blob/2dc98e2afe4face0e4ce40972a915c45368bd34a/chat_template.jinja | `575fb74f54ed264df9047d0ecce3c98938aae953fb4f50356675706264cbb68a` |
| Nemotron-3.5-Lightning-30B-A3B-BF16 | https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/blob/a9904d24bcc1d289a1950fa9d2b978c47cf903b9/chat_template.jinja | `58933db77d3099b4f78c55a38347a72e1ea05b97d6bd8f38775303dc0194e0a9` |
| Nemotron-3-Ultra-550B-A55B-BF16 | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/blob/77df655d5e9f8362164ed14dd8b48f8bce657498/chat_template.jinja | `82753bef5cedc4932c1ed509b5c9a12be680fd86d1adb65bc3f7398d11c8eebc` |

llama.cpp ships the Nano template as
[`models/templates/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.jinja`](https://github.com/ggml-org/llama.cpp/blob/a25c9865fe03c954c93fd755b5d79ae86ba99750/models/templates/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.jinja)
(byte-identical, same sha256).

**Licenses:** Nano and Super are under the
[NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/)
(`LICENSES/nvidia-nemotron-open-model-license.txt`); Lightning and Ultra under OpenMDW-1.1
(`LICENSES/openmdw-1.1.txt`, copied from Lightning's `LICENSE`).

## Engine parser mapping

| Engine (pin) | Tool parser | Reasoning parser |
|---|---|---|
| vLLM v0.30.0 | `qwen3_coder` → `Qwen3EngineToolParser` ([`vllm/tool_parsers/__init__.py#L185-L188`](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/tool_parsers/__init__.py#L185-L188)) | `nemotron_v3` → `NemotronV3ParserReasoningAdapter` ([`vllm/reasoning/__init__.py#L127-L130`](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/reasoning/__init__.py#L127-L130)); grammar [`vllm/parser/nemotron_v3.py`](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/parser/nemotron_v3.py): `qwen3_config` with `strip_trailing_reasoning_whitespace=True`, and reasoning/content swapped when `enable_thinking=false` and the content is empty |
| SGLang v0.5.20 | `qwen3_coder` → `Qwen3CoderDetector` ([`function_call_parser.py#L98`](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/function_call/function_call_parser.py#L98)) | `nemotron_3` → `Nemotron3Detector` ([`reasoning_parser.py#L1018-L1043`](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/parser/reasoning_parser.py#L1018-L1043), registered at [#L2190](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/parser/reasoning_parser.py#L2190)): `<think>`/`</think>`, `tool_start_token="<tool_call>"`, default from `enable_thinking` |
| llama.cpp `a25c9865` | specialized handler "Qwen3-Coder", selected from the template ([`common/chat.cpp#L1212-L1220`](https://github.com/ggml-org/llama.cpp/blob/a25c9865fe03c954c93fd755b5d79ae86ba99750/common/chat.cpp#L1212-L1220): "Qwen3-Coder XML tool calls, also used by Nemotron Nano 3, …"). Tests: [`tests/test-chat.cpp#L2686-L2687`](https://github.com/ggml-org/llama.cpp/blob/a25c9865fe03c954c93fd755b5d79ae86ba99750/tests/test-chat.cpp#L2686-L2687) | same handler |
| Ollama `7af39318` | `nemotron-3-nano` and `nemotron-3.5-nano` both construct `Nemotron3NanoParser` ([`model/parsers/parsers.go#L84-L85`](https://github.com/ollama/ollama/blob/7af393188defd52d370464de0d2064649cab9b41/model/parsers/parsers.go#L84-L85)), which wraps `Qwen3CoderParser` ([`model/parsers/nemotron3nano.go`](https://github.com/ollama/ollama/blob/7af393188defd52d370464de0d2064649cab9b41/model/parsers/nemotron3nano.go)). `create/metadata.go` picks the name for `nemotron_h` architectures, `nemotron-3.5-nano` only when the template contains `{reasoning effort: efficient}` (Ultra) ([#L187-L201](https://github.com/ollama/ollama/blob/7af393188defd52d370464de0d2064649cab9b41/create/metadata.go#L187-L201), [#L238-L239](https://github.com/ollama/ollama/blob/7af393188defd52d370464de0d2064649cab9b41/create/metadata.go#L238-L239)) | built in (`Nemotron3NanoParser`) |
| transformers 5.17.0 | none: no Nemotron 3/3.5 text repo ships a `response_template` (only the Gemma 4 based Bolt does), so the adapter reports `unsupported` | |

**Official recommendations (model cards):**
- Nemotron-3-Super-120B-A12B ([README.md#L279-L281](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16/blob/2dc98e2afe4face0e4ce40972a915c45368bd34a/README.md#L279-L281)) and Nemotron-3.5-Lightning ([README.md#L295-L296](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/blob/a9904d24bcc1d289a1950fa9d2b978c47cf903b9/README.md#L295-L296)): vLLM `--reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder`.
- The same cards for SGLang ([Super #L305-L306](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16/blob/2dc98e2afe4face0e4ce40972a915c45368bd34a/README.md#L305-L306), [Lightning #L363-L364](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/blob/a9904d24bcc1d289a1950fa9d2b978c47cf903b9/README.md#L363-L364)): `--reasoning-parser nemotron_3 --tool-call-parser qwen3_coder`.
- The Nano cards ([Nano-4B #L255-L257](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/blob/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f/README.md#L255-L257), [Nano-30B #L324-L326](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/bf77c3174f68ad409e1c2aa60daeb46e32d1c606/README.md#L324-L326)) recommend `--tool-call-parser qwen3_coder` with a reasoning-parser **plugin** downloaded from the model repo (`nano_v3_reasoning_parser.py`, `--reasoning-parser nano_v3`). That is repository code, not engine code, so this suite does not run it; the adapters use the engine's built-in `nemotron_v3` / `nemotron_3`, which the Super and Lightning cards name for the same format. The Nano-30B card's SGLang line (`--reasoning-parser nano_v3`, [#L401-L402](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/bf77c3174f68ad409e1c2aa60daeb46e32d1c606/README.md#L401-L402)) names a parser that is not registered in SGLang 0.5.20's `DetectorMap`.

## Known edge cases

No public engine issue specific to Nemotron 3 tool calls was found (GitHub search of the four engine repos,
2026-11-19). The tool-call parsers are the Qwen3-Coder ones, so the `qwen3-xml` edge cases apply. Replays of this
corpus at the pins (vLLM 0.30.0 and SGLang 0.5.20, 2026-11-19, macOS arm64) show:

| Fixture | Observation |
|---|---|
| `*-python-literals` | vLLM and SGLang return `"None"` (a string) for an `["integer", "null"]` parameter rendered as `None`; `False` is coerced correctly |
| `*-marker-in-arguments` | both split the value at `<parameter=KEY>` inside it (a phantom `KEY` argument). SGLang's per-token stream also re-enters reasoning at the literal `<think>` in the value |
| `*-truncated-*`, `v3-truncated-second-parallel-call` | both return the unfinished call (the `truncated-call-returned` policy finding of [`recovery.md`](recovery.md) §2) |
| reasoning / content whitespace | SGLang keeps the `\n` before `</think>` and returns the `\n` markup after the calls as content (`soft_pass`); vLLM's `nemotron_v3` strips the trailing reasoning newline |

## Fixtures (`fixtures/nemotron/`)

| File | Source | Generator |
|---|---|---|
| `v3-render.jsonl` | template renders of Nemotron-3-Nano-4B (reference); Nano-30B-A3B and Super-120B-A12B are listed in `models` because they render byte-identical ids | `scripts/fixtures/nemotron/build.py` |
| `v35-render.jsonl` | template renders of Nemotron-3.5-Lightning-30B-A3B (reference); Ultra-550B-A55B listed when identical | same |
| `truncated.jsonl` | token prefixes of those renders (what `max_tokens` produces), labelled by [`recovery.md`](recovery.md) §2 | same |

Conventions:
- Default fixtures leave `thinking` unset (the template default, on). `raw_output` then starts inside the reasoning,
  and `generation_prompt` is recorded only when it differs from the default (`enable_thinking=false`).
- Expected `reasoning_content` and `content` omit the markup newlines around `</think>` and between text and
  `<tool_call>`, so engines that keep them get `soft_pass`.
- Scalar arguments are expected in their JSON-schema type (`False` with a boolean schema is `false`, `None` with
  `["integer", "null"]` is `null`).
- `*-marker-in-arguments` values contain `<tool_call>`, `</tool_call>` and `<think>` (as their marker tokens) and the
  text `<function=NAME>` and `<parameter=KEY>`, none at a line start and no `</parameter>`. A value always ends with
  `\n</parameter>`, so only one parse is consistent with the template.
- Tags `x-nemotron3` / `x-nemotron35` name the reference template.

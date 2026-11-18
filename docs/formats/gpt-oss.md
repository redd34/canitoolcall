# gpt-oss (Harmony), slug `gpt-oss`

**Models:** gpt-oss-20b and gpt-oss-120b (2025-08), plus gpt-oss-safeguard-20b/120b (2025-09). OpenAI has released no newer open-weight LLM on HF as of 2026-09-25.

## Format

gpt-oss uses the **Harmony** response format. Every message is:

```
<|start|>{role}{header}<|message|>{content}{stop}
```

- `{header}` holds the channel (`<|channel|>analysis|commentary|final`), an optional recipient (`to=functions.NAME`) and an optional content type (`<|constrain|>json`, or plain ` json`).
- **Reasoning** is the `analysis` channel.
- **Answers** are the `final` channel.
- **Tool calls** are `commentary`-channel messages with a recipient, terminated by `<|call|>`.
- **Preambles**, which are user-visible "I'll do X" text before calls, are `commentary` messages *without* a recipient.

What the model **generates** after the prompt `<|start|>assistant`. This example is from the official Harmony spec at `openai/harmony@abd677f` `docs/format.md` §"Receiving tool calls":

```
<|channel|>analysis<|message|>Need to use function get_weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"San Francisco"}<|call|>
```

The spec says: **"The recipient might be defined in the role or channel section of the header."** Both of these therefore occur, and a parser must accept both:

```
<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{…}<|call|>
<|start|>assistant to=functions.get_weather<|channel|>commentary <|constrain|>json<|message|>{…}<|call|>
```

The spec's own examples also differ in spacing before `<|constrain|>`. One has `commentary to=functions.generate_file<|constrain|>json`, with no space.

### How the history is rendered: three sources that disagree

1. **`openai-harmony` 0.0.8** `render_conversation`, run locally:
   ```
   <|start|>assistant to=functions.get_weather<|channel|>commentary <|constrain|>json<|message|>{"city":"Zürich"}<|call|><|start|>functions.get_weather to=assistant<|channel|>commentary<|message|>{"temp":20}<|end|><|start|>assistant<|channel|>final<|message|>It is 20C.<|end|>
   ```
   The earlier `analysis` message was **dropped** because a `final` message follows, so CoT is dropped for completed turns.

2. **HF `chat_template.jinja`** (gpt-oss-20b rev `6cee5e8`), run with transformers 5.17:
   ```
   <|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{"city": "Zürich", "unit": "c"}<|call|><|start|>functions.get_weather to=assistant<|channel|>commentary<|message|>"{\"temp\": 20}"<|end|><|start|>functions.get_weather to=assistant<|channel|>commentary<|message|>"[]"<|end|><|start|>assistant<|channel|>final<|message|>It is 20C.<|return|>
   ```
   Differences from the harmony library, all observed:
   - The content type is `commentary json`, not `commentary <|constrain|>json`.
   - **Only `tool_calls[0]` is rendered.** The template has `{%- set tool_call = message.tool_calls[0] %}`, so our second parallel call (`search`) silently disappeared. The second tool result is also mislabelled `functions.get_weather`.
   - String tool results are `tojson`-encoded, which gives `"{\"temp\": 20}"`.
   - Reasoning must be passed as `message.thinking`, not `reasoning_content`. Passing both `content` and `thinking` on a tool-call message raises `Cannot pass both content and thinking in an assistant message with tool calls!`.

3. **vLLM's Harmony parser** accepts both content-type spellings: `_JSON_CONSTRAINS = [" json", " <|constrain|>json"]` in `vllm/parser/harmony.py`.

**Parallel calls.** Harmony has no parallel-call wrapper. Multiple calls are consecutive `<|start|>assistant…<|call|>` messages, and in practice the model stops at the first `<|call|>`. See Ollama #12159 below.

## Special tokens (gpt-oss-20b `tokenizer_config.json`, o200k_harmony)

| id | token | role |
|---|---|---|
| 200006 | `<\|start\|>` | message start |
| 200008 | `<\|message\|>` | header → content |
| 200007 | `<\|end\|>` | message end (not a stop token) |
| 200005 | `<\|channel\|>` | channel name follows |
| 200003 | `<\|constrain\|>` | content-type follows |
| 200012 | `<\|call\|>` | **stop**: tool call complete |
| 200002 | `<\|return\|>` | **stop**: final answer complete (EOS) |

`generation_config.json` has `eos_token_id: [200002, 199999, 200012]`. The raw text an engine sees normally ends *without* the stop token, because stop tokens are stripped. Fixtures should include both the with-stop and without-stop forms.

## Template and spec sources

| Source | URL |
|---|---|
| HF chat template | https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/chat_template.jinja |
| Harmony spec | https://github.com/openai/harmony/blob/abd677f7ac962629c808197caa1feb9e3e95d2b0/docs/format.md |
| Reference renderer | `openai-harmony` (PyPI) 0.0.8, `HarmonyEncodingName.HARMONY_GPT_OSS` |

## Engine parser mapping

| Engine | Tool parser | Reasoning parser | Notes |
|---|---|---|---|
| vLLM v0.34.0 | `openai` (`GptOssToolParser`, `vllm/tool_parsers/gptoss_tool_parser.py`, which its docstring calls a "Stub tool parser"; the real parsing is in `vllm/parser/harmony.py`) | `openai_gptoss` (`GptOssReasoningParser`) | `vllm/parser/harmony.py` wraps the `openai_harmony` `StreamableParser`. Since vllm#55128 that module comes from the `oss-harmony` fork (>=0.0.11), not `openai-harmony` |
| SGLang v0.5.20 | `gpt-oss` (`GptOssDetector`, bot token `"<\|start\|>assistant<\|channel\|>commentary"`, eot `"<\|call\|>"`) | `gpt-oss` | From reading the source, the extract regex is `to=([a-zA-Z_][a-zA-Z0-9_.-]*)\s*<\|constrain\|>json<\|message\|>(.*?)(?:<\|call\|>\|$)`. It requires a channel-section recipient and the literal `<\|constrain\|>json`. Whether the role-section and ` json` variants are handled elsewhere is **not verified by running**. |
| llama.cpp v0.5.0 | PEG, from template. Test template: `models/templates/openai-gpt-oss-120b.jinja` | same | |
| Ollama v0.34.4 | `harmony` (top-level `harmony/` package) | built in | |

**Fixture sources (Apache-2.0):**
- `vllm/tests/parser/test_harmony.py`, which includes `test_tool_call_split_across_deltas`
- `vllm/tests/reasoning/test_gptoss_reasoning_parser.py`
- `vllm/tests/entrypoints/openai/parser/test_harmony_render_parity.py`

## Known edge cases (real reports)

| Edge case | URL |
|---|---|
| gpt-oss tool calls fail in stream mode | https://github.com/vllm-project/vllm/issues/26083 |
| Responses API: `ValueError: Unknown channel: comment` closes the SSE stream (the model emitted a truncated channel name) | https://github.com/vllm-project/vllm/issues/58384 |
| `openai_harmony.HarmonyError` with GPT-OSS-120B | https://github.com/vllm-project/vllm/issues/57099 |
| 500 "Unexpected token 200002 while expecting start token 200006" with `ignore_eos=true` | https://github.com/vllm-project/vllm/issues/50690 |
| SGLang gpt-oss tool parser not working | https://github.com/sgl-project/sglang/issues/10738 |
| Malformed channel header (`<\|channel\|>` + free text / `??` / `commentary?`) fails the final PEG parse and errors the whole turn | https://github.com/ggml-org/llama.cpp/issues/27720 |
| Template escapes arguments and responses twice | https://github.com/ggml-org/llama.cpp/issues/19520 |
| Thinking returned inside tool calls | https://github.com/ollama/ollama/issues/12203 |
| No parallel tool calling | https://github.com/ollama/ollama/issues/12159 |
| "invalid character … after top-level value" parsing a tool call | https://github.com/ollama/ollama/issues/12884 |

**Fixture tags to cover:**
- `recipient-in-role` and `recipient-in-channel`
- `constrain-token` / `constrain-text` / `constrain-nospace`
- `preamble-commentary`
- `analysis-then-call`
- `sequential-calls`
- `stop-token-stripped`
- `malformed-channel`
- `unicode`
- `nested-json`

## Fixtures (`fixtures/gpt-oss/`)

| File | Source | Generator |
|---|---|---|
| `rendered.jsonl` | `openai-harmony` 0.0.8 (`render_conversation`, `auto_drop_analysis=False`) and the HF `chat_template.jinja` at `6cee5e8` | `scripts/fixtures/gpt-oss/render_gpt_oss.py` |
| `imported.jsonl` | vLLM v0.30.0 `tests/parser/test_harmony.py`, llama.cpp v0.5.0 `tests/test-chat.cpp`, `openai/harmony` tests and spec at `abd677f`, bug reports (vLLM #58384, llama.cpp #27720) | `scripts/fixtures/gpt-oss/import_gpt_oss.py` |

Regenerate with `uv run --script scripts/fixtures/gpt-oss/<script>.py` (PEP 723 dependencies, Python 3.12). Conventions:
- `raw_output` ends before `<|call|>`/`<|return|>`. A history render closes the last `final` message with `<|end|>`, where a generation has the stop token `<|return|>`, so that trailing `<|end|>` is removed.
- Token ids come from the Harmony encoder (`encode(text, allowed_special="all")` for copied strings, as vLLM's tests do). Marker strings inside argument or reasoning text are ordinary text tokens.
- History renders put the recipient in the role header. Channel-header recipients (`<|channel|>commentary to=functions.X`) come only from engine tests and the Harmony spec.
- The render script checks each complete Harmony render against `openai-harmony`'s own parser (HF-template renders are not checked this way).
- Consecutive calls separated by `<|call|>` (`vllm-sequential-calls`, `vllm-call-then-final`) only reach a parser when the stop token is not applied.

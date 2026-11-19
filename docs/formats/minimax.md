# MiniMax M2.x (M2, M2.1, M2.5, M2.7), slug `minimax`

**Models (MiniMaxAI):**
- MiniMax-M2 (2025-10)
- MiniMax-M2.1 (2025-12), the reference model for the fixtures (modified MIT)
- MiniMax-M2.5 (2026-02)
- MiniMax-M2.7 (2026-04), the current M2.x release (non-commercial license, so it is covered by the render check but is not the reference)

MiniMax-M3 and M3.1-Flash use a different tool-call format (`]<]minimax[>[<tool_call>`, see [README.md](README.md)) and have their own engine parsers (`minimax_m3`). They are **not** part of this family and are a follow-up.

## Format

All tool calls of one assistant turn sit inside **one** `<minimax:tool_call>` section. Each call is an `<invoke name="NAME">` element holding one `<parameter name="KEY">VALUE</parameter>` element per argument:

```
<minimax:tool_call>
<invoke name="NAME">
<parameter name="KEY">VALUE</parameter>
...
</invoke>
</minimax:tool_call>
```

The function name and the parameter names are **XML attributes in double quotes**. Values follow the template rule `v | tojson(ensure_ascii=False) if v is not string else v`: strings are raw and unescaped, and everything else is JSON. A parser must therefore use the tool schema to tell the string `"3"` from the number `3`. Values have no wrapping newlines, so whitespace inside `<parameter>` is data.

Rendered from MiniMax-M2.1 (rev `917dbb7`) for a turn with reasoning and two parallel calls, followed by two tool results (the system prompt and tool list that precede it are omitted):

```
]~b]ai
<think>
I should call the tools.
</think>


<minimax:tool_call>
<invoke name="get_weather">
<parameter name="city">Zürich</parameter>
<parameter name="unit">c</parameter>
</invoke>
<invoke name="search">
<parameter name="query">café "best"</parameter>
<parameter name="filters">{"tags": ["a", "b"], "max": 3}</parameter>
</invoke>
</minimax:tool_call>[e~[
]~b]tool
<response>{"temp": 20}</response>
<response>[]</response>[e~[
]~b]ai
<think>

```

Scalars, rendered the same way (`{"flag": true, "n": 3, "x": null, "s": "  two  spaces\n"}`):

```
<parameter name="flag">true</parameter>
<parameter name="n">3</parameter>
<parameter name="x">null</parameter>
<parameter name="s">  two  spaces
</parameter>
```

A call without arguments is `<invoke name="get_time">` immediately followed by `</invoke>` on the next line.

### Reasoning and generation prompt

- The generation prompt is always `]~b]ai\n<think>\n`. The template has no switch to turn thinking off.
- The raw completion therefore starts **inside the reasoning**, with no opening `<think>`: `REASONING\n</think>\n\nCONTENT\n<minimax:tool_call>…`.
- With empty content the template still writes `</think>\n\n` and then `\n<minimax:tool_call>`, so three newlines follow `</think>`. These separators are format glue, not content.
- The template renders `reasoning_content` of a past assistant turn only when no user message follows it ("interleaved thinking"). A render of an assistant turn **without** reasoning does not start with the generation prompt, which is why every fixture has reasoning.
- Text after the tool-call section is not produced by the template. The model stops (`[e~[`) after `</minimax:tool_call>`, and the next turn starts after the tool results.

## Special tokens

Ids are identical in M2, M2.1, M2.5 and M2.7 (`tokenizer.json` has the same sha256 in all four).

| token | id | special? |
|---|---|---|
| `]~b]` (begin of turn, `bos_token_id` in `generation_config.json`) | 200019 | yes |
| `[e~[` (end of turn, `eos_token_id`, the only stop token) | 200020 | yes |
| `]~!b[` (start of the system message) | 200034 | yes |
| `<think>` / `</think>` | 200050 / 200051 | no (added tokens) |
| `<minimax:tool_call>` / `</minimax:tool_call>` | 200052 / 200053 | no (added tokens) |

`<invoke`, `</invoke>`, `<parameter`, `</parameter>`, `<response>` and `<tool>` are **ordinary text** that the tokenizer splits into several tokens. Only the `<think>` and `<minimax:tool_call>` markers are single tokens, so a token-id based parser can key on those two pairs and must scan text for the rest.

## Template sources

All four templates share the same assistant-turn rendering. They differ only in the model name in the default system prompt, and in how M2 and M2.1 handle `content is none` and OpenAI-style `tool_calls[-1].function` (M2.5 and M2.7 differ from each other only by the model name). The generator checks that all four render every fixture byte-identically.

| Model | URL |
|---|---|
| MiniMax-M2.7 | https://huggingface.co/MiniMaxAI/MiniMax-M2.7/blob/0a6536713b0b03f3be97494e5cf8d22f668a3164/chat_template.jinja |
| MiniMax-M2.5 | https://huggingface.co/MiniMaxAI/MiniMax-M2.5/blob/2515a3ea0afec1dd22d0c7794b3a91557804d195/chat_template.jinja |
| MiniMax-M2.1 (reference) | https://huggingface.co/MiniMaxAI/MiniMax-M2.1/blob/917dbb7d0761e24d8623c0a5d97624550181c8e6/chat_template.jinja |
| MiniMax-M2 | https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/04312f0cfbdeefb912575595b7c045512a29293c/chat_template.jinja |

Model-card documents describing the same format (M2.7 rev `0a65367`, read for the format description only; the fixtures are not rendered from it): `docs/tool_calling_guide.md`, `docs/vllm_deploy_guide.md`, `docs/sglang_deploy_guide.md` and `docs/transformers_deploy_guide.md` in https://huggingface.co/MiniMaxAI/MiniMax-M2.7/tree/0a6536713b0b03f3be97494e5cf8d22f668a3164.

**Licenses.** The reference model, MiniMax-M2.1, declares `license: other`, `license_name: modified-mit` in its model card, with `license_link` pointing at `https://github.com/MiniMax-AI/MiniMax-M2.1/blob/main/LICENSE`. The Hugging Face repository ships no LICENSE file, so the license text was not read for this repository and is not copied into `LICENSES/`; the fixtures' provenance records it as `LicenseRef-minimax-m2.1-modified-mit` (the license as declared). The fixtures contain only format syntax and text written for this repository. M2.7 ships a non-commercial license, which is why it is not the reference; its template and tokenizer are only compared against, and nothing from them is copied.

## Engine parser mapping

Checked in the source of the pinned releases (the PyPI artifacts of vLLM 0.30.0 and SGLang 0.5.20). The vLLM row was re-checked at the current pin, v0.34.0 ([`vllm/tool_parsers/__init__.py` L161-L164](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/tool_parsers/__init__.py#L161-L164), [`vllm/reasoning/__init__.py` L127-L134](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/reasoning/__init__.py#L127-L134)): the parser names and classes are the same, `minimax_m2_tool_parser.py` is unchanged, and `vllm/parser/minimax_m2.py` only adds a streaming argument-value extender (the argument converter still keeps values verbatim). The SGLang row was re-checked at the current pin, v0.5.23: the names and classes are the same, and `function_call/minimax_m2.py` is unchanged.

| Engine | Tool parser | Reasoning parser |
|---|---|---|
| vLLM v0.34.0 (`62d4bc3`) | `minimax_m2` → `MinimaxM2ToolParser`, a thin adapter over `MinimaxM2ParserToolAdapter`; the grammar is `vllm/parser/minimax_m2.py` (`minimax_m2_config`). It accepts double-quoted, single-quoted and unquoted names, and keeps parameter values verbatim as strings in its argument converter. | `minimax_m2` → `MiniMaxM2ReasoningParser` (`vllm/reasoning/minimax_m2_reasoning_parser.py`): starts in the reasoning state, since "MiniMax M2 models don't generate `<think>` start token, only `</think>` end token". `minimax_m2_append_think` also exists: it returns no reasoning and prepends `<think>` to the content. |
| SGLang v0.5.23 (`297e54f`) | `minimax-m2` → `MinimaxM2Detector` (`python/sglang/srt/function_call/minimax_m2.py`) | `minimax` → `Qwen3Detector` with `force_reasoning=True` (`python/sglang/srt/parser/reasoning_parser.py`); `minimax-append-think` also exists |
| llama.cpp v0.5.0 | **not verified.** There is no per-model format enum; the parser is derived from the chat template, and the adapter has no `minimax` entry, so the family is reported `unsupported`. | |
| Ollama v0.34.4 | **not verified.** The adapter has no `minimax` rule, so the family is reported `unsupported`. | |
| transformers 5.17.0 | none: the family is unsupported unless a repo ships `response_template`, and the four MiniMax repos were not checked, so the adapter does not list them. | |

**Official recommendation:** the model cards' deployment guides use `--tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think` (vLLM) and `--tool-call-parser minimax-m2 --reasoning-parser minimax-append-think` (SGLang). Those append-think parsers deliberately leave the reasoning inside `content`. The adapters map the family to the reasoning parsers that **separate** reasoning (`minimax_m2` and `minimax`), because the checks compare `reasoning_content`. The matrix therefore says nothing about the append-think parsers.

## Known edge cases

| Edge case | URL |
|---|---|
| `minimax_m2` (with `glm47_moe` and `deepseek_v4`) is reported to drop the last parameter when the closing tag is omitted | https://github.com/vllm-project/vllm/issues/57826 |

No fixture for #57826 is included: the issue was not read for this family, and a MiniMax string cannot be quoted from it. Adding it (derived, tagged `x-derived`) is a follow-up.

**Fixture tags to cover:** `x-number-looking-string`, `x-schema-coercion`, `x-json-looking-string`, `x-significant-whitespace`, `x-call-index-vs-tool-index`, `x-think-no-open-tag`, and `marker-in-arguments` (a value containing `<minimax:tool_call>`, `</minimax:tool_call>` and the opening `<invoke>` and `<parameter>` tags, but not `</parameter>`).

## Fixtures in this repo

`fixtures/minimax/` (23 fixtures) is built by `uv run --script scripts/fixtures/minimax/build.py`. The build is deterministic and downloads only tokenizer and template files at pinned revisions.

| File | Source | `models` |
|---|---|---|
| `render.jsonl` (18) | MiniMax-M2.1 `chat_template.jinja` @ `917dbb7` | M2.1, M2.7, M2.5, M2 (renders checked identical, tokens and text) |
| `truncated.jsonl` (5) | token prefixes of `minimax/single-call`: at the start of a value (right after the opening `<parameter name="unit">` tag), between parameters, before `</invoke>`, before `</minimax:tool_call>`, and inside the reasoning. Cut before `</minimax:tool_call>`, the call is already closed by `</invoke>`, so it is expected with `content: null` ([recovery contract](recovery.md#2-truncated-before-the-outer-close-max_tokens), section 2) | M2.1 |

Notes:
- **Expected content and reasoning** are the message as authored. The template separators (`</think>\n\n`, and the `\n` before `<minimax:tool_call>`) are format glue, not content. Engines that keep them get `soft_pass`.
- **No `thinking` field.** The template cannot disable thinking, so fixtures leave `thinking` unset and the generation prompt is the family default.
- **No engine-test or bug-report imports** are included yet.

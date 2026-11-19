# MiniMax M2.x and M3 (M2, M2.1, M2.5, M2.7, M3, M3.1-Flash), slug `minimax`

**Models (MiniMaxAI):**
- MiniMax-M2 (2025-10)
- MiniMax-M2.1 (2025-12), the reference model for the M2.x fixtures (modified MIT)
- MiniMax-M2.5 (2026-02)
- MiniMax-M2.7 (2026-04), the current M2.x release (non-commercial license, so it is covered by the render check but is not the reference)
- **MiniMax-M3** (2026-06), the reference model for the M3 fixtures (MiniMax Community License)
- MiniMax-M3.1-Flash (2026-10), same license as M3, covered by the render check

M2.x and M3 use **different tool-call encodings**, so the family has two sub-formats, selected by `models[0]` and named by the tags `x-m2` and `x-m3`, as `kimi` does for K2 and K3:

- **`m2-xml`** ([below](#format-m2x-m2-xml)): `<minimax:tool_call>` with `<invoke name="…">` and `<parameter name="…">` elements, `<think>` reasoning.
- **`m3-xml`** ([Sub-format M3](#sub-format-m3-m3-xml)): every structural tag is prefixed with the `]<]minimax[>[` token, each parameter is an element named after its key, nested values are expanded into elements, and reasoning is `<mm:think>`.

They share one family because the framing is the same: the same turn tokens (`]~b]ai\n` … `[e~[`, ids 200019 and 200020 in both tokenizers), tool results as `]~b]tool\n<response>…</response>`, and one tool-call section per turn holding one `<invoke name="NAME">` per call with one element per argument. Both need the tool schema to recover value types. Engines ship a separate parser for each (`minimax_m2` / `minimax_m3` in vLLM, `minimax-m2` / `minimax-m3` in SGLang), and the adapters pick it from `models[0]`.

## Format (M2.x, `m2-xml`)

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

## Special tokens (M2.x)

Ids are identical in M2, M2.1, M2.5 and M2.7 (`tokenizer.json` has the same sha256 in all four).

| token | id | special? |
|---|---|---|
| `]~b]` (begin of turn, `bos_token_id` in `generation_config.json`) | 200019 | yes |
| `[e~[` (end of turn, `eos_token_id`, the only stop token) | 200020 | yes |
| `]~!b[` (start of the system message) | 200034 | yes |
| `<think>` / `</think>` | 200050 / 200051 | no (added tokens) |
| `<minimax:tool_call>` / `</minimax:tool_call>` | 200052 / 200053 | no (added tokens) |

`<invoke`, `</invoke>`, `<parameter`, `</parameter>`, `<response>` and `<tool>` are **ordinary text** that the tokenizer splits into several tokens. Only the `<think>` and `<minimax:tool_call>` markers are single tokens, so a token-id based parser can key on those two pairs and must scan text for the rest.

## Template sources (M2.x)

All four templates share the same assistant-turn rendering. They differ only in the model name in the default system prompt, and in how M2 and M2.1 handle `content is none` and OpenAI-style `tool_calls[-1].function` (M2.5 and M2.7 differ from each other only by the model name). The generator checks that all four render every fixture byte-identically.

| Model | URL |
|---|---|
| MiniMax-M2.7 | https://huggingface.co/MiniMaxAI/MiniMax-M2.7/blob/0a6536713b0b03f3be97494e5cf8d22f668a3164/chat_template.jinja |
| MiniMax-M2.5 | https://huggingface.co/MiniMaxAI/MiniMax-M2.5/blob/2515a3ea0afec1dd22d0c7794b3a91557804d195/chat_template.jinja |
| MiniMax-M2.1 (reference) | https://huggingface.co/MiniMaxAI/MiniMax-M2.1/blob/917dbb7d0761e24d8623c0a5d97624550181c8e6/chat_template.jinja |
| MiniMax-M2 | https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/04312f0cfbdeefb912575595b7c045512a29293c/chat_template.jinja |

Model-card documents describing the same format (M2.7 rev `0a65367`, read for the format description only; the fixtures are not rendered from it): `docs/tool_calling_guide.md`, `docs/vllm_deploy_guide.md`, `docs/sglang_deploy_guide.md` and `docs/transformers_deploy_guide.md` in https://huggingface.co/MiniMaxAI/MiniMax-M2.7/tree/0a6536713b0b03f3be97494e5cf8d22f668a3164.

**Licenses.** The reference model, MiniMax-M2.1, declares `license: other`, `license_name: modified-mit` in its model card, with `license_link` pointing at `https://github.com/MiniMax-AI/MiniMax-M2.1/blob/main/LICENSE`. The Hugging Face repository ships no LICENSE file, so the license text was not read for this repository and is not copied into `LICENSES/`; the fixtures' provenance records it as `LicenseRef-minimax-m2.1-modified-mit` (the license as declared). The fixtures contain only format syntax and text written for this repository. M2.7 ships a non-commercial license, which is why it is not the reference; its template and tokenizer are only compared against, and nothing from them is copied.

## Engine parser mapping (M2.x)

Checked in the source of the pinned releases (the PyPI artifacts of vLLM 0.30.0 and SGLang 0.5.20). The vLLM row was re-checked at the current pin, v0.34.0 ([`vllm/tool_parsers/__init__.py` L161-L164](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/tool_parsers/__init__.py#L161-L164), [`vllm/reasoning/__init__.py` L127-L134](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/reasoning/__init__.py#L127-L134)): the parser names and classes are the same, `minimax_m2_tool_parser.py` is unchanged, and `vllm/parser/minimax_m2.py` only adds a streaming argument-value extender (the argument converter still keeps values verbatim).

| Engine | Tool parser | Reasoning parser |
|---|---|---|
| vLLM v0.34.0 (`62d4bc3`) | `minimax_m2` → `MinimaxM2ToolParser`, a thin adapter over `MinimaxM2ParserToolAdapter`; the grammar is `vllm/parser/minimax_m2.py` (`minimax_m2_config`). It accepts double-quoted, single-quoted and unquoted names, and keeps parameter values verbatim as strings in its argument converter. | `minimax_m2` → `MiniMaxM2ReasoningParser` (`vllm/reasoning/minimax_m2_reasoning_parser.py`): starts in the reasoning state, since "MiniMax M2 models don't generate `<think>` start token, only `</think>` end token". `minimax_m2_append_think` also exists: it returns no reasoning and prepends `<think>` to the content. |
| SGLang v0.5.20 (`94602c9`) | `minimax-m2` → `MinimaxM2Detector` (`python/sglang/srt/function_call/minimax_m2.py`) | `minimax` → `Qwen3Detector` with `force_reasoning=True` (`python/sglang/srt/parser/reasoning_parser.py`); `minimax-append-think` also exists |
| llama.cpp v0.5.0 | **not verified.** There is no per-model format enum; the parser is derived from the chat template, and the adapter has no `minimax` entry, so the family is reported `unsupported`. | |
| Ollama v0.34.4 | **not verified.** The adapter has no `minimax` rule, so the family is reported `unsupported`. | |
| transformers 5.17.0 | none: the family is unsupported unless a repo ships `response_template`, and the four MiniMax repos were not checked, so the adapter does not list them. | |

**Official recommendation:** the model cards' deployment guides use `--tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think` (vLLM) and `--tool-call-parser minimax-m2 --reasoning-parser minimax-append-think` (SGLang). Those append-think parsers deliberately leave the reasoning inside `content`. The adapters map the family to the reasoning parsers that **separate** reasoning (`minimax_m2` and `minimax`), because the checks compare `reasoning_content`. The matrix therefore says nothing about the append-think parsers.

## Known edge cases (M2.x)

| Edge case | URL |
|---|---|
| `minimax_m2` (with `glm47_moe` and `deepseek_v4`) is reported to drop the last parameter when the closing tag is omitted | https://github.com/vllm-project/vllm/issues/57826 |

No fixture for #57826 is included: the issue was not read for this family, and a MiniMax string cannot be quoted from it. Adding it (derived, tagged `x-derived`) is a follow-up.

**Fixture tags to cover:** `x-number-looking-string`, `x-schema-coercion`, `x-json-looking-string`, `x-significant-whitespace`, `x-call-index-vs-tool-index`, `x-think-no-open-tag`, and `marker-in-arguments` (a value containing `<minimax:tool_call>`, `</minimax:tool_call>` and the opening `<invoke>` and `<parameter>` tags, but not `</parameter>`).

## Sub-format M3 (`m3-xml`)

All tool calls of one assistant turn sit inside **one** section. Every structural tag is prefixed with the namespace token `]<]minimax[>[` (one added token, id 200058). Each call is an `<invoke name="NAME">` element, and each argument is an element **named after its key**:

```
]<]minimax[>[<tool_call>
]<]minimax[>[<invoke name="NAME">]<]minimax[>[<KEY>VALUE]<]minimax[>[</KEY>…]<]minimax[>[</invoke>
…
]<]minimax[>[</tool_call>
```

Rendered from MiniMax-M3 (rev `d95e30c`) for a turn with reasoning and two parallel calls (fixture `minimax/m3-parallel-mixed-nested`; the stop token `[e~[` follows):

```
<mm:think>Check the weather, then search for cafés.</mm:think>]<]minimax[>[<tool_call>
]<]minimax[>[<invoke name="get_weather">]<]minimax[>[<city>Zürich]<]minimax[>[</city>]<]minimax[>[<unit>celsius]<]minimax[>[</unit>]<]minimax[>[</invoke>
]<]minimax[>[<invoke name="search_web">]<]minimax[>[<query>best cafés]<]minimax[>[</query>]<]minimax[>[<filters>]<]minimax[>[<site>example.com]<]minimax[>[</site>]<]minimax[>[<max_results>5]<]minimax[>[</max_results>]<]minimax[>[<tags>]<]minimax[>[<item>coffee]<]minimax[>[</item>]<]minimax[>[<item>wifi]<]minimax[>[</item>]<]minimax[>[</tags>]<]minimax[>[</filters>]<]minimax[>[</invoke>
]<]minimax[>[</tool_call>
```

**Values** follow the template macro `to_xml`:
- an object becomes one child element per key, an array one `<item>` element per entry (recursively);
- a boolean is `tojson` (`true`/`false`); every other scalar is written with `{{ val }}`, so a string is raw and unescaped and a number is its bare text (`3`, `21.5`);
- a key whose value is `None` is **omitted**, at the top level and inside objects (the template's comment: "The convention is: omit the field entirely");
- an empty list or object is an empty element (`]<]minimax[>[<attendees>]<]minimax[>[</attendees>`).

So the text carries no types at all: `3` and `"3"`, `[]`, `{}` and `""`, and a one-element list and its element look the same. **A parser needs the tool schema** to rebuild the arguments, and for a schema-less `object` the value cannot be recovered. Only a tag with the `]<]minimax[>[` prefix is structure, so `<tool_call>` or `</content>` without it is value text. The prefix token itself cannot be escaped.

A call without arguments is `]<]minimax[>[<invoke name="get_time">]<]minimax[>[</invoke>`. Content written before the calls sits between the reasoning and the section with no separator.

### M3 reasoning and generation prompt

The template takes `thinking_mode` (`enabled`, `disabled` or `adaptive`; unset means adaptive):

| `thinking_mode` | generation prompt | the completion starts with |
|---|---|---|
| unset or `adaptive` (default) | `]~b]ai\n` | `<mm:think>REASONING</mm:think>` or a bare `</mm:think>` (the model decides) |
| `enabled` | `]~b]ai\n<mm:think>` | the reasoning itself (no opening tag) |
| `disabled` | `]~b]ai\n</mm:think>` | the content |

A past assistant turn is rendered as `<mm:think>REASONING</mm:think>` when it has reasoning and as a bare `</mm:think>` otherwise, in every turn. That bare `</mm:think>` is format glue, not content. vLLM's and SGLang's M3 reasoning parsers both drop one leading `</mm:think>` and both start in the reasoning state only for `thinking_mode: "enabled"`.

**History render vs generation.** The fixtures are history renders in the default (adaptive) mode, so each completion contains its own `<mm:think>` or bare `</mm:think>`. That the model emits the bare closer when it does not think is what the template's own system prompt instructs ("When thinking is disabled, begin your response directly after the `</mm:think>` prefix") and what both engines' parsers expect; it is not checked against a recorded generation.

### M3 special tokens

Ids are identical in MiniMax-M3 and M3.1-Flash (`tokenizer.json` has the same Git blob, `9065e14`). The turn tokens have the same ids as in M2.x.

| token | id | special? |
|---|---|---|
| `]~b]` (begin of turn) / `[e~[` (end of turn, `eos_token_id`, the only stop token) | 200019 / 200020 | yes |
| `]~!b[` (start of the system message) | 200034 | yes |
| `<tool_call>` / `</tool_call>` | 200052 / 200053 | no (added tokens) |
| `]<]minimax[>[` (namespace prefix) | 200058 | no (added token) |
| `<mm:think>` / `</mm:think>` | 200059 / 200060 | no (added tokens) |

`<invoke`, `</invoke>`, the parameter tags and `<item>` are ordinary text. The tokenizer also merges the `>` of an opening tag with the first characters of its value (`<unit>celsius` is `<`, `unit`, `>c`, `elsius`).

### M3 template sources

| Model | URL | `chat_template.jinja` sha256 |
|---|---|---|
| MiniMax-M3 (reference) | https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/d95e30cb912fe3021bb9c2ba66a8ce60b6c937a9/chat_template.jinja | `11421244f67553498e5c8112dae02802025bcc4305ec45ad380af95c96f9fe64` |
| MiniMax-M3.1-Flash | https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/ce7f4df87d2e05262f5775f373dac4772f134041/chat_template.jinja | `961e3f96b2c08d26555efdfeb3b384d5e81181bf0cfc04edbadc31c6e507b431` |

The M3 template has not changed since the repository's first commit (`3a41b31`, same Git blob `93022eb` at `3a41b31`, `f0e1c1e` and `d95e30c`). The M3.1-Flash template differs only outside the assistant turn: the system message (current date and location, a `<model_id>`, `reasoning_effort`), mid-conversation system messages, and the generation prompt, which ignores `thinking_mode` and is always the adaptive `]~b]ai\n`. The generator checks that M3.1-Flash renders every M3 fixture byte-identically (text and token ids). llama.cpp's curated `models/templates/MiniMax-M3.jinja` at `a25c9865` is byte-identical to the M3 template.

**License.** Both repositories ship the **MiniMax Community License** (`license: other`, `license_name: minimax-community`, with a `LICENSE` file). It grants use "for non-commercial purposes" and adds conditions for commercial use (a "Built with MiniMax M3" notice, and a notice to or authorization from MiniMax). As for Kimi-K3, the text is copied verbatim to `LICENSES/minimax-m3-community.txt` (byte-identical to the M3 `LICENSE`, Git blob `f413dfa`), and the fixtures record `LicenseRef-minimax-m3-community`. The fixtures contain only format syntax and text written for this repository: no system prompt or other template text is stored. Whether the corpus may ship in the wheel under these terms is a maintainer decision; it is recorded here, not decided.

### M3 engine parser mapping

Checked in the source at each adapter's pin.

| Engine | Tool parser | Reasoning parser |
|---|---|---|
| vLLM v0.34.0 (`62d4bc3`) | `minimax_m3` → `MinimaxM3ToolParser` ([`vllm/tool_parsers/__init__.py` L165-L168](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/tool_parsers/__init__.py#L165-L168)), a thin adapter over `MinimaxM3ParserToolAdapter`; the grammar is `vllm/parser/minimax_m3.py` (`minimax_m3_config`). It converts elements with the tool schema (`coerce_to_schema_type`) and streams each call's arguments in one piece (`stream_arg_deltas=False`). | `minimax_m3` → `MiniMaxM3ReasoningParser` ([`vllm/reasoning/__init__.py` L135-L138](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/reasoning/__init__.py#L135-L138)) |
| SGLang v0.5.20 (`94602c9`) | `minimax-m3` → `MinimaxM3Detector` ([`function_call_parser.py` L103](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/function_call/function_call_parser.py#L103), `python/sglang/srt/function_call/minimax_m3.py`) | `minimax-m3` → `MiniMaxM3Detector` ([`reasoning_parser.py` L2185](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/parser/reasoning_parser.py#L2185)); `force_reasoning` comes from `thinking_mode == "enabled"` ([L2224-L2227](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/parser/reasoning_parser.py#L2224-L2227)) |
| llama.cpp `a25c9865` | a specialized handler, `common_chat_params_init_minimax_m3` (`common/parsers/minimax-m3.cpp`, format `peg-minimax-m3`), chosen when the template contains `]<]minimax[>[`, `<tool_call>` and `<invoke name=` ([`common/chat.cpp` L1173-L1180](https://github.com/ggml-org/llama.cpp/blob/a25c9865fe03c954c93fd755b5d79ae86ba99750/common/chat.cpp#L1173-L1180)). **Not replayed:** the adapter has no `minimax` entry (it would need a vocab-only GGUF and a llama.cpp build), so the family stays `unsupported`. | (same handler: `<mm:think>` / `</mm:think>`) |
| Ollama `7af39318` | none: `model/parsers/parsers.go` has no MiniMax parser and no path in the tree names MiniMax. `unsupported`. | |
| transformers 5.17.0 | none: neither M3 nor M3.1-Flash ships a `response_template` in `tokenizer_config.json` at the pinned revisions. `unsupported`. | |
| mlx-lm 0.32.2 | none: `mlx_lm/tool_parsers/` has `minimax_m2.py` but no M3 parser. The adapter maps no `minimax` repo. | |

vLLM v0.34.0 also has M3 parsers in its Rust frontend (`rust/src/parser/src/tool/minimax_m3.rs`, `rust/src/parser/src/reasoning/minimax_m3.rs`). The adapter replays the Python registry only, so the matrix says nothing about the Rust ones.

**Official recommendation:** vLLM's recipe `recipes/models/MiniMaxAI/MiniMax-M3.yaml` (at `62d4bc3`) uses `--tool-call-parser minimax_m3 --reasoning-parser minimax_m3` and recommends `--default-chat-template-kwargs '{"thinking_mode": "enabled"}'` for agentic workloads. The adapters map M3 to the same parsers. All M3 fixtures leave `thinking` unset (the template default, adaptive), so the `enabled` mode is not covered yet; see the notes below.

## Fixtures in this repo

`fixtures/minimax/` (51 fixtures) is built by `uv run --script scripts/fixtures/minimax/build.py`. The build is deterministic and downloads only tokenizer and template files at pinned revisions.

| File | Source | `models` |
|---|---|---|
| `render.jsonl` (18) | MiniMax-M2.1 `chat_template.jinja` @ `917dbb7` | M2.1, M2.7, M2.5, M2 (renders checked identical, tokens and text) |
| `truncated.jsonl` (5) | token prefixes of `minimax/single-call`: at the start of a value (right after the opening `<parameter name="unit">` tag), between parameters, before `</invoke>`, before `</minimax:tool_call>`, and inside the reasoning. Cut before `</minimax:tool_call>`, the call is already closed by `</invoke>`, so it is expected with `content: null` ([recovery contract](recovery.md#2-truncated-before-the-outer-close-max_tokens), section 2) | M2.1 |
| `m3-render.jsonl` (22) | MiniMax-M3 `chat_template.jinja` @ `d95e30c`, ids `minimax/m3-*`, tag `x-m3` | M3, M3.1-Flash (renders checked identical, tokens and text) |
| `m3-truncated.jsonl` (6) | token prefixes of `minimax/m3-single-call`: inside the reasoning, between parameters, inside the second parameter's opening tag (the tokenizer merges its `>` with the value, so no cut falls right after it), before `]<]minimax[>[</invoke>`, before `]<]minimax[>[</tool_call>`; and of `minimax/m3-parallel-same-tool` inside the second call. Calls closed by `]<]minimax[>[</invoke>` before the cut are expected, with `content: null`; unfinished ones must not be returned (recovery contract, section 2) | M3 |

Notes:
- **Expected content and reasoning** are the message as authored. The M2.x template separators (`</think>\n\n`, and the `\n` before `<minimax:tool_call>`) and the M3 bare `</mm:think>` are format glue, not content. Engines that keep the M2.x separators get `soft_pass`.
- **No `thinking` field.** The M2.x template cannot disable thinking, and the M3 fixtures use the template default (adaptive), so every fixture leaves `thinking` unset and the generation prompt is the reference model's default. M3 fixtures for `thinking_mode: "enabled"` (prompt pre-fills `<mm:think>`, as vLLM's recipe recommends for agents) are a follow-up: the fixture's `thinking: true` would have to reach the template and the parsers as `thinking_mode: "enabled"`, which neither the vLLM nor the SGLang adapter sends yet.
- **M3 arguments.** No M3 fixture has a `None` argument, because the template drops such keys (the correct parse of the generation has no key, which differs from the authored message). Every M3 tool has a complete schema, because a schema-less `object` cannot be typed back from the XML; for that reason the M2.x `update_settings` tool is not offered in M3 fixtures, and M3 fixtures add `create_order` (an array of objects).
- **M3 tags:** `x-m3-bare-think-close` (no reasoning, the completion starts with `</mm:think>`), `x-m3-single-item-array`, `x-m3-empty-container`, and the shared ones above (`x-number-looking-string`, `x-schema-coercion`, `x-json-looking-string`, `x-significant-whitespace`, `x-call-index-vs-tool-index`, `marker-in-arguments`).
- **No engine-test or bug-report imports** are included yet.

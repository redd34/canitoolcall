# Tool-call format notes

One file for each model family. Each file gives the exact raw text the model emits for tool calls, and for reasoning where the family has it. It also lists the special tokens, the source template (URL and revision), the engine parsers that handle the format, and known edge cases with bug-report URLs.

These notes are the reference for writing fixtures. Every rendered example in these files came from **running the official template or encoder locally**. Nothing here was typed in by hand. Section 3 lists the tools used.

## 1. Families covered (checked 2026-09-25)

| Slug | Family | Current models (HF, 2026-09) | Tool-call shape |
|---|---|---|---|
| [`qwen3-hermes`](qwen3-hermes.md) | Qwen3 (2504 and 2507), Qwen2.5 | Qwen3-0.6B…32B, Qwen3-*-2507 | `<tool_call>\n{"name":…, "arguments":…}\n</tool_call>` + `<think>` |
| [`qwen3-xml`](qwen3-xml.md) | Qwen3-Coder, Qwen3.5, Qwen3.6, Qwen3.8 | Qwen3-Coder-30B-A3B, Qwen3-Coder-Next, Qwen3.5-9B, Qwen3.6-35B-A3B, Qwen3.8-27B | `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>…` |
| [`gpt-oss`](gpt-oss.md) | gpt-oss (Harmony) | gpt-oss-20b, gpt-oss-120b, gpt-oss-safeguard | `<\|start\|>assistant<\|channel\|>commentary to=functions.NAME <\|constrain\|>json<\|message\|>{…}<\|call\|>` |
| [`deepseek`](deepseek.md) | DeepSeek V3/R1 → V3.1 → V3.2 → V4 → V4.1 | DeepSeek-V4.1-Flash, V4-Flash(-0731), V4-Pro, V3.2, V3.1, R1-0528 | 5 incompatible formats: full-width `<｜tool▁calls▁begin｜>` tokens (V3/R1, V3.1), then DSML XML (V3.2 `function_calls`, V4 `tool_calls`, V4.1 ` calls`) |
| [`kimi`](kimi.md) | Kimi K2.x and K3 | Kimi-K2-Instruct(-0905), K2-Thinking, K2.5, K2.6, K2.7-Code, K3 | K2: `<\|tool_calls_section_begin\|>…<\|tool_call_begin\|>functions.NAME:IDX<\|tool_call_argument_begin\|>{…}`; K3: "XTML" `<\|open\|>call tool="…"<\|sep\|>…` |
| [`glm`](glm.md) | GLM-4.5, 4.6, 4.7, 5.x | GLM-5.3, GLM-5.3-Flash, GLM-5.2, GLM-4.7(-Flash), GLM-4.5 | `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>` |
| [`llama`](llama.md) | Llama 3.1, 3.2, 3.3 and 4 | Llama-3.1-8B, 3.2-1B/3B, 3.3-70B, Llama-4-Scout/Maverick | 3.1/3.3: bare JSON `{"name":…, "parameters":…}`; 3.2/4: pythonic `[f(a=1)]` |
| [`mistral`](mistral.md) | Mistral, Magistral, Devstral, Ministral | Mistral-Small-4, Mistral-Medium-3.5, Ministral-3, Devstral-2, Magistral-2509, Mistral-Small-3.2 | depends on tokenizer version: `[TOOL_CALLS][{…}]` (v3/v7), `[TOOL_CALLS]NAME[CALL_ID]ID[ARGS]{…}` (v11), `[TOOL_CALLS]NAME[ARGS]{…}` (v13+) |
| [`gemma4`](gemma4.md) | Gemma 4 (and a note on Gemma 3) | gemma-4-31B-it, 26B-A4B-it, 12B-it, E4B/E2B-it | `<\|tool_call>call:NAME{key:<\|"\|>str<\|"\|>,n:3}<tool_call\|>`: **not JSON** |

Cross-family rules for malformed and truncated output are in [`recovery.md`](recovery.md) (proposed, [#36](https://github.com/redd34/canitoolcall/issues/36)).

### How currency was checked
The HF API was queried by author on 2026-09-25, sorted by downloads and by creation date:
- **Qwen:** Qwen3.8 (2026-08) is current.
- **DeepSeek:** V4.1-Flash (2026-09-10) is current.
- **Moonshot:** Kimi-K3 (2026-06) is current.
- **Z.ai:** GLM-5.3 (2026-08) is current.
- **Google:** Gemma 4 (2026-03…05) is current.
- **Mistral:** Mistral-Medium-3.5 and Small-4 (2026-01…03) are current.
- **OpenAI:** gpt-oss (2025-08) is still the only open-weight LLM.
- **Meta:** there has been no new open-weight Llama since Llama 4 (2025-04). Llama is kept because 3.x is still among the most-downloaded instruct models (Llama-3.1-8B-Instruct has 6.2M downloads).

### Not covered yet (next candidates)
- **MiniMax M2.x/M3 and MiniMax-H3.** High download counts, and both vLLM and SGLang have dedicated parsers (`minimax_m2`, `minimax_m3`). Formats rendered during this research, from the official templates:
  - **M2.7** (rev `d494266`): `<minimax:tool_call>\n<invoke name="…">\n<parameter name="k">v</parameter>…`
  - **M3** (rev `f0e1c1e`): `]<]minimax[>[<tool_call>` with per-key XML elements such as `<city>Zürich]<]minimax[>[</city>`
- **NVIDIA Nemotron 3 / 3.5.** vLLM `nemotron_v3` reasoning parser, SGLang `nemotron_3`.

## 2. Engine versions the parser mappings refer to

| Engine | Version | Commit | Where parsers live |
|---|---|---|---|
| vLLM | v0.30.0 (2026-09-22) | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` | Names in `vllm/tool_parsers/__init__.py` and `vllm/reasoning/__init__.py`. Most grammars are now in `vllm/parser/<family>.py`, which is the new "parser engine". The `*_tool_parser.py` classes are thin adapters over it. |
| SGLang | v0.5.20 (2026-09-18) | `94602c9c2b7cbdb8efd5c52802dac6a1c180089e` | `python/sglang/srt/function_call/function_call_parser.py` (`ToolCallParserEnum`), `python/sglang/srt/parser/reasoning_parser.py` (`DetectorMap`) |
| llama.cpp | v0.5.0 (2026-09-23) | tag `v0.5.0` | `common/chat.cpp`, `common/chat-auto-parser*.cpp`, `common/chat-peg-parser.cpp`. There are no per-model format enums any more: formats are `PEG_SIMPLE`/`PEG_NATIVE`/`PEG_GEMMA4`/`PEG_MINIMAX_M3`, derived from the chat template. Test templates are in `models/templates/`. |
| Ollama | v0.34.4 (2026-09-23) | tag `v0.34.4` | `model/parsers/parsers.go` (names such as `qwen3`, `qwen3.5`, `qwen3-coder`, `harmony`, `deepseek3`, `glm-4.7`, `gemma4`, `ministral`). Other families go through the generic template-driven `tools/` package. |

vLLM and SGLang **names differ** for the same format. For example, DeepSeek V3.1 is `deepseek_v31` in vLLM and `deepseekv31` in SGLang, and gpt-oss is `openai` in vLLM and `gpt-oss` in SGLang. Each family file gives both.

## 3. How the examples were produced (reproducible)

The same conversation was rendered for every family:
- a system message and a user message
- an assistant message with `reasoning_content` and **two parallel tool calls**:
  - `get_weather({"city": "Zürich", "unit": "c"})`
  - `search({"query": "café \"best\"", "filters": {"tags": ["a","b"], "max": 3}})`
- two tool results
- a final assistant answer

This covers unicode, embedded quotes, nested objects, arrays, integers and parallel calls in a single render.

**Rendering tools** (in a throwaway venv, not part of the package):
- `transformers==5.17.0` `render_jinja_template` for all HF Jinja templates
- `openai-harmony==0.0.8` for gpt-oss
- `mistral-common==1.12.0` for tekken-tokenizer Mistral models
- DeepSeek's own `encoding/*.py` files for V3.2, V4 and V4.1
- Moonshot's `encoding_k3.py` for Kimi K3

Each HF file was downloaded at the pinned revision (`/resolve/<sha>/…`).

**Important caveat for fixture authors.** A chat template shows how a *past* assistant turn is serialized back into the prompt. That is usually, but not always, byte-identical to what the model *generates*. Each family file calls out known differences. Examples:
- gpt-oss: the HF template writes `commentary json`, but the harmony library writes `commentary <|constrain|>json`.
- Qwen3.5+: the generation prompt pre-fills `<think>\n`, so the raw completion starts *inside* the reasoning with no opening tag.
- Gemma 4: generation stops on `<|tool_response>`.

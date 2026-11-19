# Llama (3.1, 3.2, 3.3, 4), slug `llama`

**Models:**
- Llama-3.1-8B/70B/405B-Instruct (2024-07)
- Llama-3.2-1B/3B-Instruct (2024-09)
- Llama-3.3-70B-Instruct (2024-11)
- Llama-4-Scout-17B-16E and Llama-4-Maverick-17B-128E (2025-04)

**Currency:** Meta has published no newer open-weight Llama on HF. The newest meta-llama repos are Llama-Guard-4 and Prompt-Guard-2 (2025-04). Llama is still worth covering because Llama-3.2-1B and 3.1-8B-Instruct remain among the most-downloaded instruct models (7.4M and 6.2M).

**Access caveat:** all meta-llama HF repos are **gated** (`gated: manual`), so the HF templates could not be downloaded anonymously. The sources used instead are:
1. Meta's prompt-format docs in `meta-llama/llama-models`
2. llama.cpp's verbatim copies of the HF templates in `models/templates/`, used for rendering
3. vLLM's example templates

The HF repo revisions are still recorded for when access is granted:
- Llama-3.1-8B-Instruct `0e9e39f`
- Llama-3.3-70B-Instruct `6f6073b`
- Llama-4-Scout-17B-16E-Instruct `92f3b15`

## Two tool-call syntaxes

### A. JSON (the HF chat template for 3.1, 3.2 and 3.3)

Rendered from llama.cpp's copy `models/templates/meta-llama-Llama-3.1-8B-Instruct.jinja` @ `v0.5.0`. The 3.3-70B copy is **byte-identical** (same md5). The 3.2-3B copy differs elsewhere but renders the call the same way:

```
<|start_header_id|>assistant<|end_header_id|>

{"name": "get_weather", "parameters": {"city": "Zürich", "n": 2}}<|eot_id|><|start_header_id|>ipython<|end_header_id|>

"{\"temp\": 20}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

- The arguments key is **`parameters`**, not `arguments`.
- **Only one call per message.** With two calls the template raises `This model only supports single tool-calls at once!`, which we observed.
- Tool results use the **`ipython`** role, and string results are `tojson`-encoded.
- Meta's docs (`models/llama3_1/prompt_format.md`) show the model *may* prefix the JSON with `<|python_tag|>` and end with `<|eom_id|>` when `Environment: ipython` is in the system prompt. The HF template adds `Environment: ipython` whenever tools are given. Example from the doc:
  ```
  <|python_tag|>{
      "type": "function",
      "name": "trending_songs",
      "parameters": {
          "n": "10",
          "genre": "all"
      }
  }<|eom_id|>
  ```
  So in the wild, raw output can be any of:
  - bare JSON
  - `<|python_tag|>` + JSON
  - JSON with an extra `"type": "function"` key
  - `<|eom_id|>` or `<|eot_id|>` as the terminator
- **Built-in tools** (3.1 docs) use Python call syntax after `<|python_tag|>`, e.g. `<|python_tag|>brave_search.call(query="latest price of 1oz gold")<|eom_id|>`.

### B. Pythonic (Meta docs for 3.2 lightweight models and Llama 4)

From Meta's `models/llama3_2/text_prompt_format.md`:

```
[get_weather(city='San Francisco', metric='celsius'), get_weather(city='Seattle', metric='celsius')]<|eot_id|>
```

From Meta's `models/llama4/prompt_format.md`:

```
[get_weather(city="San Francisco"), get_weather(city="Seattle")]<|eot|>
```

- The whole reply is a Python list of calls with keyword arguments. Parallel calls are native.
- Both quote styles appear in Meta's own examples.
- The Llama 4 system prompt says "NEVER combine text and function calls in the same response". The model still does sometimes, and parsers disagree on the result (vLLM #56840).
- Meta's Llama 4 docs also show an optional custom format, `<function=trending_songs>{"n": 10}</function><|eot|>`.

**Discrepancy.** Meta's 3.2 doc describes pythonic output, but the 3.2 HF template (via llama.cpp's copy) renders the **JSON** format for assistant history. Engines split on this:
- vLLM offers both `llama3_json` and `pythonic`.
- For Llama 4, vLLM has `llama4_pythonic` and also `llama4_json`, because the Scout checkpoint is reported emitting JSON (vLLM #46863).

**Template artefact (observed).** vLLM's `examples/tool_chat_template_llama4_pythonic.jinja` renders our test arguments as `search(query="café "best"", filters="{'tags': ['a', 'b'], 'max': 3}")`. The embedded quote is not escaped and the dict becomes a Python-repr string. This is a community template, not Meta's, but it shows that the pythonic format has no defined escaping rules in the template layer.

## Special tokens

**Llama 3.x** (`models/llama3_1/prompt_format.md`):
- `<|begin_of_text|>`
- `<|start_header_id|>` / `<|end_header_id|>`
- `<|eot_id|>` (end of turn)
- `<|eom_id|>` (end of message, "a tool call needs to be made")
- `<|python_tag|>` ("used in the model's response to signify a tool call")
- `<|finetune_right_pad_id|>`
- Roles: `system`, `user`, `assistant`, `ipython`

**Llama 4** (`models/llama4/prompt_format.md`):
- `<|begin_of_text|>`
- `<|header_start|>` / `<|header_end|>`
- `<|eot|>` / `<|eom|>`
- Image tokens

Token IDs were not verified because the tokenizers are gated.

## Sources

| Source | URL |
|---|---|
| Llama 3.1 prompt format (Meta) | https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3_1/prompt_format.md |
| Llama 3.2 text prompt format (Meta) | https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3_2/text_prompt_format.md |
| Llama 3.3 prompt format (Meta) | https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3_3/prompt_format.md |
| Llama 4 prompt format (Meta) | https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama4/prompt_format.md |
| Meta reference tool-call parsing | https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3/tool_utils.py |
| HF template copy (3.1) | https://github.com/ggml-org/llama.cpp/blob/v0.5.0/models/templates/meta-llama-Llama-3.1-8B-Instruct.jinja |
| HF template copy (3.2) | https://github.com/ggml-org/llama.cpp/blob/v0.5.0/models/templates/meta-llama-Llama-3.2-3B-Instruct.jinja |
| HF (gated) | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/0e9e39f249a16976918f6564b8830bc894c89659/tokenizer_config.json |

## Engine parser mapping

| Engine | JSON format | Pythonic format |
|---|---|---|
| vLLM v0.34.0 | `llama3_json`, `llama4_json` (both `LlamaJsonEngineToolParser`, `vllm/tool_parsers/llama_json_engine_tool_parser.py`, grammar `vllm/parser/llama_json.py`: `TOOL_START = "<\|python_tag\|>"`, which it prepends when the output starts with `{`; calls are a `;`-separated sequence). vllm#63569 moved them to the parser engine; v0.30.0 had `Llama3JsonToolParser` in `vllm/tool_parsers/llama_tool_parser.py`. | `pythonic` (`PythonicToolParser`), `llama4_pythonic` (`Llama4PythonicToolParser`) |
| SGLang v0.5.23 | `llama3` (`Llama32Detector`, bot `"<\|python_tag\|>"`, separator `";"`) | `pythonic` (`PythonicDetector`) |
| llama.cpp v0.5.0 | PEG, from template (test templates above) | same |
| Ollama v0.34.4 | generic template-driven `tools/` detection; no dedicated Llama parser in `model/parsers/` | same |

There are no reasoning parsers: Llama has no reasoning channel.

**Fixture sources (Apache-2.0):**
- vLLM: `tests/tool_parsers/test_llama3_json_tool_parser.py`, `test_llama4_pythonic_tool_parser.py`, `test_pythonic_tool_parser.py`
- SGLang: `test/registered/unit/function_call/test_llama32_detector.py`

The `meta-llama/llama-models` repo licence is reported by GitHub as "Other" (NOASSERTION). Review `LICENSE` before vendoring anything from it. Quoting short format examples with attribution is lower risk than copying code.

## Known edge cases (real reports)

| Edge case | URL |
|---|---|
| `llama3_json` (and jamba, ernie45) drop tool calls when the whole message arrives in a single streaming delta | https://github.com/vllm-project/vllm/issues/48294 |
| Llama-4-Scout-FP8 tool calls left in content: pythonic parser vs JSON output mismatch | https://github.com/vllm-project/vllm/issues/46863 |
| `pythonic` / `llama4_pythonic` return a tool call when streaming but raw text when not (trailing prose, leading-underscore names) | https://github.com/vllm-project/vllm/issues/56840 |
| `llama4_pythonic` fails with SyntaxError on nested list parameters | https://github.com/vllm-project/vllm/issues/30722 |
| SGLang `llama3` parser silently deletes any leading JSON object from message content | https://github.com/sgl-project/sglang/issues/35562 |
| llama3.1 always uses a tool (Ollama) | https://github.com/ollama/ollama/issues/6127 |

**Fixture tags to cover:**
- `parameters-key`
- `python-tag-prefix`
- `eom-terminator`
- `extra-type-key`
- `single-call-only`
- `pythonic-parallel`
- `pythonic-nested-list`
- `pythonic-quote-escaping`
- `text-plus-call`
- `json-in-content-not-a-call` (a legitimate JSON answer must not become a call)
- `single-delta`

## Fixtures in this repo

`fixtures/llama/` (34 fixtures) is built by `uv run --script scripts/fixtures/llama/build.py`. The build is deterministic and never touches the gated meta-llama repos.

**Mirrors (verified by the build):**
- **Template:** `unsloth/Llama-3.3-70B-Instruct@99cd0d2` `chat_template.jinja` is byte-identical to llama.cpp's copy of the HF template. The build checks this against llama.cpp `a25c9865` on every run.
- **Llama 3 tokenizer:** the same mirror's `tokenizer.json` has the same sha256 as the gated `meta-llama/Llama-3.3-70B-Instruct` LFS object (`6b9e4e7f…`).
- **Llama 4 tokenizer:** `unsloth/Llama-4-Scout-17B-16E-Instruct@afd8e49` matches `meta-llama/Llama-4-Scout-17B-16E-Instruct` (`172c9eb4…`).
- **Llama 4 template:** the mirror's chat template is a **different** blob from Meta's, so there are no Llama 4 template renders.
- Every fixture's `tokenizer` pin names its mirror; `models[0]` stays the meta-llama repo.

| File | Source |
|---|---|
| `l3-render.jsonl` | Llama 3.3 template (renders checked identical for the 3.1-8B and 3.2-3B templates) |
| `recorded.jsonl` | "Model Response Format" blocks of Meta's `llama3_3` and `llama4` prompt-format docs @ `0e0b8c5`, checked verbatim against the doc by the build, plus a token-prefix truncation |
| `truncated.jsonl` | a token prefix of a render |
| `imported.jsonl` | vLLM v0.30.0 and SGLang v0.5.20 tests (Apache-2.0), plus SGLang #35562 and vLLM #48294 and #56840 |

Notes:
- **Syntax is selected by `models[0]`:** Llama-3* uses JSON and Llama-4* uses pythonic. Meta's docs also show Llama 3.2/3.3 emitting *pythonic* lists when the system prompt asks for them. These are not included, because a family/model key cannot select a different parser for the same model. A per-fixture variant would be needed.
- **Text around calls.** The template drops `content` next to a call and renders only one call per message. Text-plus-call and multi-call shapes therefore come from engine tests and Meta's recordings only.


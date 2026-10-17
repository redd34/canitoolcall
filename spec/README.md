# CanIToolCall fixture spec v0.1

A **fixture** is one raw model output together with the tools offered and the parse a correct engine must produce. Fixtures are language-neutral JSON Lines files, so any harness (Python, Rust, TypeScript, Go) can replay them.

| File | What it defines |
|---|---|
| [`fixture.schema.json`](fixture.schema.json) | One fixture record (one JSONL line) |
| [`family.schema.json`](family.schema.json) | `fixtures/<slug>/family.json`, the metadata shared by a family |
| [`results.schema.json`](results.schema.json) | The results file written by `canitoolcall run` |

Validate everything with:

```sh
uv run canitoolcall validate            # all of fixtures/
uv run canitoolcall validate fixtures/qwen3-hermes/
```

## Layout

```
fixtures/
  <family-slug>/
    family.json           # family metadata (markers, reference models, notes link)
    <topic>.jsonl         # one fixture per line, e.g. basic.jsonl, parallel.jsonl, edge.jsonl
```

The family slug is the directory name, the `family` field, and the prefix of every `id` (`qwen3-hermes/parallel-two-calls-unicode`). Ids are stable forever. If content changes meaningfully, add a new id and delete the old one; never repurpose an id.

## Fixture fields

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | `<family>/<kebab-name>`, globally unique |
| `family` | yes | family slug |
| `models` | yes | HF repo ids that emit exactly this format. `models[0]` is the reference model for tokenizer and template |
| `spec_version` | yes | `"0.1"` |
| `provenance` | yes | `{kind, source_url, revision, license, generator?, template_sha256?, attribution?}` (see below) |
| `tools` | yes | the tools offered, in OpenAI function format |
| `raw_output` | yes | the **exact** completion: after the generation prompt, before the stop token, with special tokens written literally |
| `expected` | one of | `{content, reasoning_content, tool_calls: [{name, arguments}]}` |
| `expected_error` | one of | `{reason, accept: [no_tool_calls \| content_passthrough \| exception]}` for truncated or malformed output |
| `tags` | yes | edge-case tags (enum in the schema, or custom `x-*`) |
| `output_token_ids` | recommended | token ids of `raw_output` under `tokenizer` |
| `tokenizer` | with ids | `{repo, revision, mode: hf\|mistral}`. `repo`@`revision` must be `models[0]` at its pinned revision in `family.json`, or that reference model's declared `mirror` (for gated repos). Revisions are full 40-hex commit shas, never branch names: adapters load tokenizers from this pin, and only a short reviewed allowlist of pins may run repository code (`trust_remote_code`) |
| `generation_prompt` | optional | the text `add_generation_prompt` appends, when it matters (e.g. it pre-fills `<think>\n`) |
| `thinking` | optional | whether thinking was enabled in the request; `null` means the template default |
| `assumes` | optional | unique replay requirements; currently only `stop_token_ignored` |
| `finish` | optional | `{reason: stop\|length, token: string\|null}`; actual termination, not candidate stops |
| `notes` | optional | free text |

### Provenance is mandatory. Never invent formats.

| `kind` | Use when | `source_url` points to | Typical license |
|---|---|---|---|
| `template_render` | You rendered an assistant tool-call message through the model's **official** chat template or encoder (HF Jinja, `openai-harmony`, `mistral_common`, DeepSeek/Moonshot encoder scripts) | the template file at the pinned revision | the model repo's license |
| `engine_test` | You copied a raw output from an engine's test suite | the test file at a commit, with line anchor | vLLM/SGLang/transformers Apache-2.0, llama.cpp/Ollama MIT |
| `bug_report` | The output reproduces a public issue | the issue or PR (comment) URL | `NOASSERTION` plus a short quote |
| `recorded` | A real generation captured from the model | the recording log or dataset | as stated |
| `spec_example` | An example quoted verbatim from the format's official specification (e.g. openai/harmony `docs/format.md`) | the spec document at a commit, with line anchor | the spec's license |

`template_render` fixtures must name the `generator` script in `scripts/fixtures/` that reproduces them, and should record `template_sha256`.

**History rendering is not generation.** A chat template serializes a *past* assistant turn. That can differ from what the model generates, and engines parse *generations*. Known cases (see `docs/formats/`):
- Mistral v11 history renders add `[CALL_ID]<id>` when the call has an id. mistral-common's generation grammar has no `[CALL_ID]`, but that grammar only constrains guided decoding, and llama.cpp's own Mistral-Small-3.2 tests assume the model emits it. Which form the model generates freely is **unverified**, so both shapes are covered (engine tests for `[CALL_ID]`, id-less renders otherwise; see `docs/formats/mistral.md`).
- gpt-oss templates put the recipient in the role header (`<|start|>assistant to=functions.X<|channel|>commentary`), while generations put it after the channel. The Harmony spec allows both (`docs/format.md`: "The recipient might be defined in the role or channel section of the header"). Role-header fixtures are tagged `x-recipient-in-role`, channel-form ones `x-recipient-in-channel`, so the matrix and triage can count one gap once rather than once per fixture.
- The generation stop token is cut: `<|call|>` for Harmony, `<|im_end|>` for Qwen, `<tool_call|>`/`<|tool_response>` for Gemma 4.

When the two differ, prefer `engine_test` or `recorded` sources, and say so in `notes`.

**Derived fixtures are tagged.** A fixture whose `raw_output` is not quoted verbatim from its `source_url` (for example a `bug_report` fixture built by applying the reported malformation to a template render, or a token-prefix truncation of a `spec_example`) carries `x-derived` or `x-derived-truncation`, and its `notes` say how it was made.

### Replay conditions and generation termination

`assumes` is input applicability, not an expected parse. `stop_token_ignored`
marks output that only reaches a parser when normal stop-token handling is
bypassed, such as the two imported gpt-oss sequences continuing after `<|call|>`.
No current adapter implements that requirement: these two cases are reported
as `unsupported` with a reason and no checks, not as parser failures or passes.
Unknown assumption names and contradictory metadata are validation errors.

An optional `finish` records **the actual ending**:

- `{"reason": "length", "token": null}` means the output was cut; append no stop.
- `{"reason": "stop", "token": "<literal token>"}` identifies the actual token,
  which can be a non-first entry of the reference model's `stop_tokens`.
- A length finish with a token, a stop finish with the legacy `truncated` tag,
  or an output already ending in its declared terminal token is invalid.
- Stop-without-token semantics remain deferred. Do not invent a token or turn
  an unknown ending into a length finish.

When `finish` is absent, **all legacy adapter behavior is retained**, including
existing family-specific stop selection and the `truncated` tag. No corpus-wide
migration is required. Absent fields stay absent through serialization; an
explicit empty `assumes` list stays present. `ReplayInput.stop_tokens` remains
the full set of candidate stops, separate from `ReplayInput.finish`.

The nonabstract `Adapter.supports_replay(raw)` sees only replay input, never
`expected` or `expected_error`. Its default preserves the original family/model
`Support` and reason for unmarked legacy inputs, and declines unimplemented
conditions. An override may narrow that support, never widen it; the worker
retains the family refusal guard and checks applicability before `units`, either
parse path, or `parser_config`. Putting `unsupported` in a scored `CheckResult`
is an error, not a way to manufacture a passing case.

SGLang and llama.cpp consume explicit finish metadata in their existing shared
stop-plan/request paths. The other adapters conservatively decline explicit
finishes until their serving-path handling is implemented and tested. No
adapter opts into `stop_token_ignored`. Parser configuration retains the explicit
`replay_conditions` plus any engine-specific stop observation. Legacy parser
configuration is unchanged when both optional fields are absent.

### Token ids come first

Re-encoding text is lossy for some tokenizers (`mistral_common` never maps the text `[TOOL_CALLS]` to its control token). Generators therefore render with `apply_chat_template(tokenize=True)`, slice off the prompt ids, cut at the first stop id from `generation_config.json`, and store the ids in `output_token_ids` together with the `tokenizer` pin. `raw_output` is then the ids decoded with `skip_special_tokens=False`. Adapters must use `output_token_ids` when present and only fall back to tokenizing `raw_output` otherwise.

## Expected results

- `content` and `reasoning_content` are strings or `null`. `null` means absent; `""` is treated as `null`.
- **Content around tool calls** is ordinary content: the text before and after a call (or a tool-calls section) is concatenated verbatim, in order, with nothing inserted or removed between the segments. Fixtures whose expected value depends on this rule rather than on the format itself are tagged `x-policy`.
- `tool_calls` are listed in emission order. `arguments` is a JSON **object** and is compared after parsing, so key order and whitespace do not matter, but value types do (`3` ≠ `"3"`).
- Use `expected_error` for outputs that are truncated or malformed. The parser passes if it does one of the `accept` outcomes:
  - `no_tool_calls`: no exception, no calls
  - `content_passthrough`: no calls, and the text comes back as content
  - `exception`: the engine's parser raised

  Fixtures made by cutting a template render short (a token prefix, which is what `max_tokens` produces) accept all three outcomes. They carry the `truncated` tag, which means the generation ended at `max_tokens`: the model never emitted a stop token, so adapters must replay the finish as `finish_reason: "length"` and never append a stop token. A fixture copied from an engine test or a bug report may accept fewer when its source says which outcome is correct, and its `notes` say why.

  Returning a tool call for output that was cut off is deliberately **not** an accepted outcome (there is no `partial_call`). The model never finished that call. Its arguments are either invalid JSON, which crashes clients that parse them, or they were silently completed by the parser, and the client would then run a call the model never made. This includes a (partial) call that a streaming parse returned before it raised: the client had already received it. Some OpenAI-compatible servers return such a call together with `finish_reason: "length"`. A client that checks `finish_reason` can cope with that, but because it is unsafe by default, these fixtures judge it `fail`. The matrix shows these results under the `expected_error` check, so they can be told apart from mis-parses of complete output. When two parallel calls are cut in the second one, fixtures expect the first, complete call to be kept.

## Checks

For each fixture and engine, the runner produces one non-streaming parse and one streaming parse per chunking strategy. Then it applies these checks:

| Check | Passes when |
|---|---|
| `expected_match` | the result equals `expected` (per strategy, including `nonstream`) |
| `expected_error` | the outcome is in `expected_error.accept` |
| `stream_equals_nonstream` | every streaming result equals the non-streaming result |
| `split_invariance` | all streaming results are equal to one another |
| `no_leakage` | no family marker appears in content, reasoning or argument strings unless `expected` contains it verbatim |
| `arguments_json` | every `arguments_raw` parses as a JSON object |
| `arguments_schema` | every call's arguments validate against the tool's `parameters` schema, and the name is one of the offered tools |
| `parallel_order` | the number of calls and their order of names match `expected` |

A case's status is the worst over its checks: `fail` > `error` > `soft_pass` > `pass`. It is `unsupported` when the adapter has no parser configuration for the family or model, and the case then records the adapter's `reason`. `error` is reserved for **harness** problems; an exception raised by the engine's own parser is a parse outcome and is judged by the checks.

Check details (policy of `canitoolcall.checks`):

- **Rows.** Each check emits one row per parse it applies to (`nonstream` or a strategy id). `split_invariance` emits a single row with strategy `*`, comparing every realistic stream with the `one` stream. `arguments_json` and `arguments_schema` emit rows only for parses that returned tool calls. `parallel_order` applies only when `expected` has two or more calls.
- **Arguments** are compared as parsed JSON: key order and whitespace don't matter, and value types do (`3` ≠ `"3"`, `true` ≠ `1`). `1` and `1.0` are the same JSON number. Text that is not strict JSON never equals a valid parse. That includes `""` for a no-argument call, a double-encoded string, `NaN`/`Infinity`, an object with a duplicate key (clients may keep either value), and a missing (`null`) `arguments_raw`.
- **Exceptions.** Two parses that raised are equal when the exception *type* matches; messages may differ.
- **`expected_error`.** A result is `content_passthrough` when its content equals `raw_output`. Equality after stripping surrounding whitespace also counts, but that alone gives `soft_pass`.
- **`no_leakage`** also scans tool names and JSON object keys. For `expected_error` fixtures, markers in `content` are allowed, since passing the raw text through is a graceful outcome.
- **`arguments_schema`.** A call whose arguments equal the expected arguments passes even if they violate the schema: the model emitted them, and the engine returned them faithfully. An invalid tool schema is a fixture problem and gives `error`.
- **Synthetic streams** get rows like any other stream, but those rows never count toward the case status, and `split_invariance` leaves them out. `char:<seed>` is always synthetic. Multi-token strategies (`one`, `special`, `rand:<seed>:<max>` with `max` > 1) are synthetic for engines whose server emits one event per generated token (llama-server, Ollama, transformers `serve`; the adapter's `tokens_per_step = "one"`), because no user of those engines can receive such a delta. The results file lists them in `run.synthetic_strategies`.
- **Skipped strategies.** When the worker cannot run a requested strategy for a case, the case lists it in `skipped_strategies` with the reason. Examples: `special` when the adapter doesn't expose special token ids, and `char:*` when the adapter has no text-delta path.

## Chunking strategies

Streams are built from groups of **token ids**, never from characters, because engines never split a token and special tokens are atomic. Each engine detokenizes the groups with its own incremental detokenizer.

| id | Meaning |
|---|---|
| `one` | the whole output in one delta |
| `special` | split at special-token boundaries: every special token is its own delta, and each run of ordinary tokens between them is one delta. The adapter supplies the engine tokenizer's special ids |
| `token` | one token per delta |
| `rand:<seed>:<max>` | seeded random groups of 1..`max` tokens (Python `random.Random(seed)`) |
| `char:<seed>` | synthetic per-character stress (can split special tokens). Opt-in only and reported separately, because it is not realistic |

Which strategies count depends on the engine: see "Synthetic streams" above.

The default set is `one`, `special`, `token`, and `rand:1:8` … `rand:5:8`. The same seed always gives the same grouping, on every machine.

## Normalization policy `soft-v1`

Engines disagree on whitespace in ways users rarely notice. Results distinguish:
- **strict** equality: `""` becomes `null`; everything else is compared exactly (arguments as parsed JSON).
- **soft** equality (`soft-v1`): `content` and `reasoning_content` are stripped of leading and trailing whitespace, and a whitespace-only string becomes `null`.

Strict equality gives `pass`. Soft-only equality gives `soft_pass`, which the matrix shows separately. Everything else is `fail`.

## Versioning

`spec_version` follows `MAJOR.MINOR`. Adding optional fields or tags is a minor bump. Changing required fields or semantics is a major bump. Harnesses must reject fixtures with an unknown major version.

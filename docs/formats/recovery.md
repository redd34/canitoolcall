# Malformed and truncated output: the recovery contract

**Status: proposed** in [#36](https://github.com/redd34/canitoolcall/issues/36). Section 3 is implemented in
`canitoolcall.checks.compare_modes`. The rest of the code changes the rule implies are listed under
[Follow-up work](#follow-up-work).

The [spec](../../spec/README.md#expected-results) says what well-formed output must parse to. This page says what a
parser must do when the model's output is **malformed** (a marker is missing, extra or garbled, but generation ended
normally) or **truncated** (generation hit `max_tokens`, recorded as the `truncated` tag or `finish.reason: "length"`).
It applies to every family, so it lives here rather than in one family's notes.

## The principle

A client acts on tool calls without reading them. That gives one ordering of outcomes, from best to worst:

1. **Recovered:** the call the model meant, with the arguments it wrote.
2. **Rejected cleanly:** no call, the text kept as content (or dropped), or an exception, the same way in every mode.
3. **Wrong:** a call whose arguments differ from what the model wrote: a value dropped, cut, merged or replaced by
   `{}`, or a marker leaked into it. The client runs a call the model never made.

Outcome 3 is always `fail`. Which of 1 and 2 is required depends on whether the model finished.

## 1. Missing inner close, generation finished

*Example: `glm/glm47-missing-last-close-arg-value`, where the last `</arg_value>` is missing but `</tool_call>` closes the
call.*

**Rule.** When the call's extent is unambiguous, **the parser must recover the value**. The extent is unambiguous when
the unclosed value is ended by a marker the format reserves (the outer close `</tool_call>`, `</function>`, or the next
`<parameter=` or `<arg_key>`), and the recovered value then contains no family marker. The fixture keeps `expected` set
to the recovered parse. A clean rejection is `fail`. A silently wrong value is `fail`, and is the worse finding.

**Why recovery, not "a clean error is fine":**

- The model made the call, and the generation ended normally. Rejecting it fails the turn for output whose meaning
  is not in doubt, and an agent loop cannot do anything with that.
- These are real generations, not stress strings. `qwen3-xml/bug-missing-close-parameter-before-function` is
  Qwen3.8-27B at temperature 0 ([vllm#57699](https://github.com/vllm-project/vllm/issues/57699)), and the GLM case is
  the malformation reported in [vllm#57826](https://github.com/vllm-project/vllm/issues/57826). vLLM fixed the Qwen3
  equivalent in [vllm#57707](https://github.com/vllm-project/vllm/issues/57707), and its own test
  (`qwen3-xml/vllm-coder-missing-close-parameter`) asserts recovery.
- The matrix answers "will this model's tool calls work on this engine?". For these inputs, an engine that rejects
  them does not work.

Triage still separates the two kinds of `fail`: a clean rejection is a lesser finding than a corrupted value, and its
triage summary should say "rejected, not corrupted".

**Ambiguous extents** are not covered by this rule. One example is two calls where the first has an unclosed value
(the known limitation in [#37](https://github.com/redd34/canitoolcall/issues/37)): the scanner cannot tell that from a
literal `<tool_call>…</tool_call>` inside a value. A fixture for such a case states its own accepted outcomes and says
why in `notes`. Merging two calls into one is a wrong value under any reading.

**Effect today** (2026-09-25 snapshot unless noted):

| Fixture | vLLM | SGLang | llama.cpp | Ollama |
|---|---|---|---|---|
| `glm/glm47-missing-last-close-arg-value` | fail: drops the last argument | fail | fail: leaks `</tool_call>` into the value | pass at `7af3931`. Fail with [ollama#18663](https://github.com/ollama/ollama/pull/18663): a clean rejection, so a regression |
| `qwen3-xml/bug-missing-close-parameter-before-function` | fail: `{}` non-streaming | pass | fail | fail |
| `qwen3-xml/vllm-coder-missing-close-parameter` | soft_pass | soft_pass | fail | fail |

So #37 is a regression under this rule, and the patch drafted there (recover at `done` by splitting at the first
`</tool_call>`) is the fix to offer upstream.

## 2. Truncated before the outer close (`max_tokens`)

**Rule.** The spec's existing truncation policy stands. **The unfinished call must not be returned**, in either mode,
not even as a partial call that a stream emits before it raises. Returning the text as content, dropping it, or (for
token-prefix truncations) raising are all accepted. A call that was complete before the cut is kept: the
`*-truncated-second-parallel-call` fixtures already expect the first call.

**Why.** The model never finished the call. Its arguments are either invalid JSON or were completed by the parser, and
for a mutating tool a call completed by guesswork is the dangerous outcome. Ollama rejects unfinished calls on purpose
for this reason. According to
[oscar030406's data on this issue](https://github.com/redd34/canitoolcall/issues/36#issuecomment-6872104911), vLLM's Rust
frontend does too (the `is_incomplete_tool_call` branch in `rust/src/chat/src/output/default/unified.rs`). The same data
shows that 12 vLLM parsers already return the cut text as content at every cut.

**Not decided here: text the model completed before the cut.** `no_tool_calls` currently also accepts dropping
complete reasoning or prose that came before the unfinished call. oscar030406's cuts show this "drops all" outcome is
common (for example 131 of 181 cuts for `deepseek_v3`). Requiring that this text be kept is a reasonable tightening, but
it changes the contract of about 20 template fixtures. It gets its own decision once derived-cut fixtures exist to
measure it (see [Follow-up work](#follow-up-work)).

**Effect today:** none. The five `*-truncated-call-returned` triage findings (70 cases) remain `truncation_policy`
failures. So does `mistral/v13think-stop-at-open-marker` on vLLM, where every stream leaves an empty call after
`[TOOL_CALLS]`.

## 3. Streaming vs non-streaming

**Rule.**

- **Tool calls and exceptions must be identical in every mode, always.** If one mode recovers a call and another
  rejects it, or one raises and another does not, the result is `fail`.
- For `expected` fixtures, nothing changes. Every mode must equal `expected`, so the modes equal each other.
- For `expected_error` fixtures, **each mode is judged on its own by `expected_error`**. When the modes return the same
  tool calls (normally none) and the same exception type but **different recovered text**, `stream_equals_nonstream` and
  `split_invariance` give `soft_pass` instead of `fail`.

**Why.** Text recovered from malformed output has no reference value: the fixture lists acceptable outcomes, not a
string. A streaming parser that held back a possible call cannot take back deltas it already sent, and a non-streaming
parser can choose its fallback after seeing the whole output, so byte equality between two unspecified fallbacks tests
an implementation detail. A differing call or a differing raise changes what the client does, so those stay `fail`.
`soft_pass` keeps the difference visible on the matrix without counting it as a mis-parse.

**Effect today:** 16 cells move from `fail` to `soft_pass`, because their only failures are cross-mode text
differences on `expected_error` fixtures:

| Engine | Fixtures |
|---|---|
| vLLM (6) | `deepseek/vllm-v3-malformed-missing-call-tokens`, `gpt-oss/vllm-malformed-headers`, `mistral/vllm-v3-malformed-not-json`, `qwen3-hermes/bug-sglang-30480-truncated-at-opener`, `qwen3-hermes/sglang-malformed-json-in-tags`, `qwen3-hermes/truncated-after-open-tag` |
| SGLang (5) | `gemma4/vllm-malformed-no-brace`, `gemma4/truncated-in-name`, `llama/l4-truncated-missing-close-bracket`, `qwen3-hermes/bug-sglang-30480-truncated-at-opener`, `qwen3-hermes/sglang-malformed-json-in-tags` |
| llama.cpp (4) | `gemma4/vllm-malformed-no-brace`, `gemma4/truncated-in-name`, `kimi/k3-vllm-truncated-tools`, `qwen3-hermes/truncated-after-open-tag` |
| Ollama (1) | `gpt-oss/harmony-truncated-in-arguments` |

This resolves the six vLLM cases in `vllm-malformed-content-recovery-contract` and `vllm-harmony-recovery-newline`
from [#13](https://github.com/redd34/canitoolcall/pull/13).

## 4. Raised exceptions

**Rule.** An exception raised by the engine's parser is a **parse outcome, never `error`**. `error` stays reserved for
harness problems, as the spec already says.

- For an `expected` fixture, a raise is `fail` (a rejection under section 1).
- For an `expected_error` fixture, a raise passes only if `exception` is in `accept`.
- A stream that emitted a call and then raised is `fail`, because the client already received the call.

**Which fixtures accept `exception`.** Token-prefix truncations of a template render accept it, as they do today. A
malformed fixture leaves it out when its source shows that raising is the harm: a bug report about the turn failing,
or an engine test asserting a non-raising fallback. Current labels already follow this, so none change. For example,
`gpt-oss/bug-garbled-channel-commentary-question` comes from
[llama.cpp#27720](https://github.com/ggml-org/llama.cpp/issues/27720), and raising there throws away the complete
`analysis` message that follows the garbled header.

**Why not count a raise as `error`?** An `error` cell means "we could not test this", and it hides a real behaviour
behind a harness label. In most servers a parser exception fails the whole response, which is exactly what a user of
the matrix needs to see.

**Effect today:** none. `gpt-oss/bug-garbled-channel-commentary-question` on vLLM (`HarmonyError` in every mode)
stays `fail`. Its triage moves from `untriaged_discrepancy` to a finding under this rule.

## The eight vLLM cases from #13

| Fixture | Rule | Result |
|---|---|---|
| `gpt-oss/vllm-malformed-headers` | 3 | `soft_pass`: the modes differ only by an inter-message newline |
| `deepseek/vllm-v3-malformed-missing-call-tokens`, `mistral/vllm-v3-malformed-not-json`, `qwen3-hermes/bug-sglang-30480-truncated-at-opener`, `qwen3-hermes/sglang-malformed-json-in-tags`, `qwen3-hermes/truncated-after-open-tag` | 3 | `soft_pass`: no calls in any mode, and the recovered text differs |
| `gpt-oss/bug-garbled-channel-commentary-question` | 4 | `fail`: raises in every mode, and the fixture does not accept `exception` |
| `deepseek/vllm-v3-malformed-missing-brace` | 1 | `fail`: returns `get_weather` with invalid arguments in some modes. That is a wrong value, and vLLM's own test marks it `xfail` |

## Fixture labels

**No fixture's expected value or accepted outcomes change.** Every malformed fixture with `expected` is a section 1
case with an unambiguous extent, and every `expected_error` fixture's `accept` list already follows sections 2 and 4.
The rule changes how two checks compare modes (section 3), not what any fixture expects.

## Follow-up work

1. **Checks (done: `compare_modes`):** in `check_stream_equals_nonstream` and `check_split_invariance`, give `soft_pass` for
   `expected_error` fixtures when the calls and exception type are equal and only `content` or `reasoning_content`
   differ. Document it under "Normalization policy" in `spec/README.md`, with a regression test for each direction:
   text difference → `soft_pass`, call or raise difference → `fail`.
2. **Triage (after the next run):** regenerate `triage.jsonl`. Retire `vllm-malformed-content-recovery-contract` and
   `vllm-harmony-recovery-newline`, and file `vllm-harmony-malformed-header-rejection` and
   `vllm-deepseek-v3-malformed-call-fragments` under this rule. Move `glm/glm47-missing-last-close-arg-value` out of
   `sglang-other` and `llamacpp-other`.
3. **Derived-cut fixtures** (offered by oscar030406): take a small set of cuts at structural boundaries from
   `engine_test` fixtures, not every token boundary. Tag them `x-derived-truncation` and `truncated`, and give each one
   the outcome section 2 assigns. These cuts are the evidence for the "text completed before the cut" decision.
4. **#37:** post the recovery patch on ollama#18663, following Ollama's contribution policy.

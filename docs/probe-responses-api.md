# `probe --api responses`: the OpenAI Responses API

`canitoolcall probe --api responses` sends the same scenarios as the default
`/chat/completions` probe to `POST {base_url}/responses`, the OpenAI Responses
API that agent CLIs such as Codex CLI use. `--api both` runs both APIs and adds
an `api` column to the table. The default stays `--api chat`, so existing runs,
reports and exit codes do not change.

The judgments are the same as for chat: every call must name an offered tool,
its arguments must be a JSON object that validates against the tool's
`parameters` schema, no format marker may leak into the text, a tool name or an
argument value, and the streamed result must have the same shape as the
non-streamed one. Only the wire format differs. This page lists the shapes the
probe sends and expects, and where each one comes from.

## Sources

Everything below was checked on 2026-11-19 against:

- OpenAI API reference, Responses: <https://developers.openai.com/api/reference/resources/responses>
- OpenAI API reference, Responses streaming events: <https://developers.openai.com/api/reference/resources/responses/streaming-events>
- OpenAI function calling guide (request, `function_call_output` and the
  "Output events" stream example): <https://developers.openai.com/api/docs/guides/function-calling>
- The types generated from OpenAI's OpenAPI spec in `openai-python`
  [v3.55.1](https://github.com/openai/openai-python/tree/v3.55.1/src/openai/types/responses)
  (latest release on that date): `function_tool.py`, `tool_choice_function_param.py`,
  `easy_input_message_param.py`, `response_function_tool_call_param.py`,
  `response_input_item_param.py` (`FunctionCallOutput`), `response.py`,
  `response_output_message.py`, `response_reasoning_item.py`,
  `response_function_tool_call.py` and the `response_*_event.py` stream events.
- `openai-python` v3.55.1 stream helper
  [`lib/streaming/responses/_responses.py`](https://github.com/openai/openai-python/blob/v3.55.1/src/openai/lib/streaming/responses/_responses.py):
  `get_final_response()` raises ``Didn't receive a `response.completed` event.``
  when the stream ends without one, and argument deltas are appended to the
  output item that `response.output_item.added` announced at the same
  `output_index`.

## Request

| Chat Completions (`--api chat`) | Responses (`--api responses`) |
|---|---|
| `messages` | `input`: `{role: "user", content}` messages and, for the follow-up scenario, the earlier call as `{type: "function_call", call_id, name, arguments}` and its result as `{type: "function_call_output", call_id, output}` |
| `tools: [{type: "function", function: {name, description, parameters}}]` | `tools: [{type: "function", name, description, parameters}]` |
| `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "function", name}` |
| `parallel_tool_calls`, `temperature: 0`, `stream` | the same fields |

## Non-streaming response

The probe reads the `output` list of the `response` object:

- `message` items: the `output_text` parts are the content.
- `reasoning` items: the `reasoning_text` parts of `content` are the reasoning.
  If there are none, the `summary_text` parts of `summary` are used.
- `function_call` items: `name`, `arguments` (a JSON **string**) and `call_id`.

Failures: `arguments` that are not a string, a call without a `name`, and a
call without a `call_id`. The reference marks `call_id` as required, and a
`function_call_output` can only refer to a call by its `call_id`. (In chat, a
missing `id` is only a warning.) A `status` other than `completed` is a warning.
An `error` object with no output is reported as an error.

## Streaming response

The probe accumulates the SSE events the way the OpenAI SDK builds its
snapshot:

- `response.output_item.added` with a `function_call` item opens a call at its
  `output_index`.
- `response.function_call_arguments.delta` appends `delta` to that call.
- `response.output_text.delta` is content. `response.reasoning_text.delta` is
  reasoning, with `response.reasoning_summary_text.delta` as the fallback.
- `error` and `response.failed` are reported as errors.
  `response.incomplete` is a warning.

It then checks the result against the final events:

| Condition | Result |
|---|---|
| The stream ends without `response.completed` (or `response.incomplete`) | fail |
| An argument delta arrives for an `output_index` whose `function_call` item was never added | fail |
| `response.function_call_arguments.done` `arguments` differ from the concatenated deltas | fail |
| `response.output_item.done` items, or a non-empty `response.completed` `output`, list other calls than the deltas | fail |
| A call has no `call_id` or no `name` | fail |
| No `response.function_call_arguments.done`, or no `response.output_item.done`, for a call | warning |
| `stream: true` answered with plain JSON | fail (same as chat) |

Reasoning from the Responses API is shown under `reasoning_content` in the
report, so the marker checks treat it the same way as chat's `reasoning_content`.

## Not covered

- Built-in tools (web search, file search, MCP, custom tools) and `previous_response_id` / stored responses.
  The probe sends every turn in `input`.
- `response.content_part.*`, `response.output_text.done`, `sequence_number` ordering and
  `item_id` cross-references are not judged.

## Tests

`tests/probe/conftest.py` serves `/v1/responses` from the same canned turns as
`/v1/chat/completions`. It returns a `response` object, or the event sequence
from the function-calling guide's "Output events" example. The quirk flags
`done_arguments`, `omit_completed`, `omit_id`, `ignore_stream` and `error_event`
produce the failures that `tests/probe/test_probe_responses.py` checks. The mock
is hand-written test data, not a recording of a real server.

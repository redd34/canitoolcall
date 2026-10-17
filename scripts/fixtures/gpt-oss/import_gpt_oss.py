# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["transformers==5.17.0", "jinja2>=3.1", "huggingface_hub>=1.0", "openai-harmony==0.0.8"]
# ///
"""Import gpt-oss (Harmony) fixtures from engine test suites, the Harmony library's own tests and
spec, and public bug reports.

Every raw output below is COPIED VERBATIM from the cited source (file + line at a pinned commit, or
issue URL). Token ids are produced exactly the way those tests produce them:
``openai_harmony`` ``encoding.encode(text, allowed_special="all")``. The only transformations are:

* a trailing stop token (``<|call|>`` / ``<|return|>``) is removed, because fixture ``raw_output``
  ends before the stop token (spec/README.md);
* a leading ``<|start|>assistant`` is removed where a source includes the generation prompt.

Each is recorded in the fixture's ``notes``. Tool schemas are request context the sources do not
always spell out; they are written to match the arguments in the source.

Run from the repo root::

    uv run --script scripts/fixtures/gpt-oss/import_gpt_oss.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import OUT_DIR, check_ids, expected, hf_tokenizer, record, write_jsonl
from openai_harmony import (
    Conversation,
    HarmonyEncodingName,
    Message,
    RenderConversationConfig,
    Role,
    load_harmony_encoding,
)

GENERATOR = "scripts/fixtures/gpt-oss/import_gpt_oss.py"

VLLM_SHA = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"  # vLLM v0.30.0
VLLM_HARMONY = f"https://github.com/vllm-project/vllm/blob/{VLLM_SHA}/tests/parser/test_harmony.py"
LLAMA_SHA = "7fe450e19305b828c199d602c23a8337aaa1f03b"  # commit of llama.cpp tag v0.5.0 (tag object c13fcbf6)
LLAMA_TEST = f"https://github.com/ggml-org/llama.cpp/blob/{LLAMA_SHA}/tests/test-chat.cpp"
HARMONY_SHA = "abd677f7ac962629c808197caa1feb9e3e95d2b0"
HARMONY_TESTS = f"https://github.com/openai/harmony/blob/{HARMONY_SHA}/tests/test_harmony.py"
HARMONY_SPEC = f"https://github.com/openai/harmony/blob/{HARMONY_SHA}/docs/format.md"

VLLM_ATTR = "Copyright contributors to the vLLM project (Apache-2.0)."
LLAMA_ATTR = "Copyright (c) 2023-2026 The ggml authors (MIT)."
HARMONY_ATTR = "Copyright OpenAI (openai/harmony, Apache-2.0)."


def obj(**props: Any) -> dict[str, Any]:
    return {"type": "object", "properties": props}


def tool(name: str, params: dict[str, Any], description: str | None = None) -> dict[str, Any]:
    fn: dict[str, Any] = {"name": name, "parameters": params}
    if description:
        fn["description"] = description
    return {"type": "function", "function": fn}


S = {"type": "string"}
T_WEATHER_LOC = tool("get_weather", obj(location=S))
T_WEATHER_CITY = tool("get_weather", {**obj(city=S), "required": ["city"]})
T_TIME = tool("get_time", obj(timezone=S))
T_USER_LOC = tool("get_user_location", {"type": "object", "properties": {}, "required": []})
T_WEATHER_LATLON = tool("get_weather", obj(latitude={"type": "number"}, longitude={"type": "number"}))
T_GENERATE_FILE = tool("generate_file", obj(template=S, path=S))
# llama.cpp tests/test-chat.cpp special_function_tool (L425) and the inline "edit" tool (L6474-L6495).
T_SPECIAL = tool(
    "special_function",
    {"type": "object", "properties": {"arg1": {"type": "integer", "description": "The arg."}}, "required": ["arg1"]},
    "I'm special",
)
T_EDIT = tool(
    "edit",
    {
        "type": "object",
        "properties": {
            "oldString": {"type": "string", "description": "Old string to replace."},
            "newString": {"type": "string", "description": "New replacement string."},
            "replaceAll": {"type": "boolean", "description": "Whether to replace all occurrences."},
        },
        "required": ["oldString", "newString"],
    },
    "Edit a file",
)


@dataclass
class Src:
    kind: str
    url: str
    revision: str
    license: str
    attribution: str | None = None

    def prov(self, line: str | None) -> dict[str, Any]:
        p: dict[str, Any] = {
            "kind": self.kind,
            "source_url": f"{self.url}#{line}" if line else self.url,
            "revision": self.revision,
            "license": self.license,
            "generator": GENERATOR,
        }
        if self.attribution:
            p["attribution"] = self.attribution
        return p


VLLM = Src("engine_test", VLLM_HARMONY, VLLM_SHA, "Apache-2.0", VLLM_ATTR)
LLAMA = Src("engine_test", LLAMA_TEST, LLAMA_SHA, "MIT", LLAMA_ATTR)
HARMONY_T = Src("engine_test", HARMONY_TESTS, HARMONY_SHA, "Apache-2.0", HARMONY_ATTR)
HARMONY_S = Src("spec_example", HARMONY_SPEC, HARMONY_SHA, "Apache-2.0", HARMONY_ATTR)


def vllm_analysis(enc: Any, text: str) -> str:
    """tests/parser/test_harmony.py::get_model_output_str([assistant(text, "analysis")]) (L108-123)."""
    cfg = RenderConversationConfig(auto_drop_analysis=False)
    user = [Message.from_role_and_content(Role.USER, "x")]
    prompt = enc.render_conversation_for_completion(Conversation.from_messages(user), Role.ASSISTANT, config=cfg)
    msg = Message.from_role_and_content(Role.ASSISTANT, text).with_channel("analysis")
    full = enc.render_conversation(Conversation.from_messages([*user, msg]), config=cfg)
    return str(enc.decode_utf8(full[len(prompt) :]))


def build(enc: Any) -> list[dict[str, Any]]:
    analysis = vllm_analysis(enc, "reasoning")
    cases: list[dict[str, Any]] = [
        # ---------------------------------------------------------------- vLLM tests/parser/test_harmony.py
        dict(
            name="vllm-channel-first-recipient",
            src=VLLM,
            line="L917-L920",
            text=analysis + "<|start|>assistant<|channel|>commentary to=functions.get_weather"
            ' <|constrain|>json<|message|>{"city": "Tokyo"}<|call|>',
            tools=[T_USER_LOC, T_WEATHER_CITY],
            expected=expected(None, "reasoning", [("get_weather", {"city": "Tokyo"})]),
            tags=["single-call", "reasoning", "x-recipient-in-channel", "x-constrain-token"],
            notes="TestAdjustRequest.TOOL_CALL_2_CHANNEL_FIRST: the analysis prefix is ANALYSIS = "
            "get_model_output_str([assistant('reasoning', 'analysis')]) rendered with openai-harmony, as in the test.",
        ),
        dict(
            name="vllm-channel-first-empty-arguments",
            src=VLLM,
            line="L907-L910",
            text=analysis + "<|start|>assistant<|channel|>commentary to=functions.get_user_location"
            " <|constrain|>json<|message|>{}<|call|>",
            tools=[T_USER_LOC, T_WEATHER_CITY],
            expected=expected(None, "reasoning", [("get_user_location", {})]),
            tags=["single-call", "reasoning", "empty-arguments", "x-recipient-in-channel"],
            notes="TestAdjustRequest.TOOL_CALL_1_CHANNEL_FIRST.",
        ),
        dict(
            name="vllm-split-deltas-constrain-nospace",
            src=VLLM,
            line="L580-L612",
            text="<|channel|>analysis<|message|>Thinking<|end|>"
            "<|start|>assistant to=functions.get_weather<|channel|>commentary"
            '<|constrain|>json<|message|>{"location": ' + '"Paris"}<|call|>',
            tools=[T_WEATHER_LOC],
            expected=expected(None, "Thinking", [("get_weather", {"location": "Paris"})]),
            tags=["single-call", "reasoning", "x-recipient-in-role", "x-constrain-nospace"],
            notes="test_tool_call_split_across_deltas[commentary]: the two deltas of the test, concatenated.",
        ),
        dict(
            name="vllm-call-on-analysis-channel",
            src=VLLM,
            line="L580-L612",
            text="<|channel|>analysis<|message|>Thinking<|end|>"
            "<|start|>assistant to=functions.get_weather<|channel|>analysis"
            '<|constrain|>json<|message|>{"location": ' + '"Paris"}<|call|>',
            tools=[T_WEATHER_LOC],
            expected=expected(None, "Thinking", [("get_weather", {"location": "Paris"})]),
            tags=["single-call", "reasoning", "x-recipient-in-role", "x-call-on-analysis", "x-constrain-nospace"],
            notes="test_tool_call_split_across_deltas[analysis]: a functions.* recipient on the analysis channel "
            "is still a tool call.",
        ),
        dict(
            name="vllm-preamble-call-end-terminated",
            src=VLLM,
            line="L490-L505",
            text="<|channel|>commentary<|message|>Let me check the weather.<|end|>"
            "<|start|>assistant to=functions.get_weather<|channel|>commentary"
            '<|message|>{"location": "SF"}<|end|>',
            tools=[T_WEATHER_LOC],
            expected=expected("Let me check the weather.", None, [("get_weather", {"location": "SF"})]),
            tags=[
                "single-call",
                "text-before-call",
                "x-preamble-commentary",
                "x-no-content-type",
                "x-recipient-in-role",
            ],
            notes="test_commentary_with_recipient_excluded. The call message is closed by <|end|> instead of the "
            "<|call|> stop token.",
        ),
        dict(
            name="vllm-sequential-calls",
            assumes=["stop_token_ignored"],
            src=VLLM,
            line="L723-L745",
            text="<|channel|>analysis<|message|>Thinking<|end|>"
            "<|start|>assistant to=functions.get_weather<|channel|>commentary"
            '<|constrain|>json<|message|>{"location": "Paris"}<|call|>'
            "<|start|>assistant to=functions.get_time<|channel|>commentary"
            '<|constrain|>json<|message|>{"timezone": "UTC"}<|call|>',
            tools=[T_WEATHER_LOC, T_TIME],
            expected=expected(
                None, "Thinking", [("get_weather", {"location": "Paris"}), ("get_time", {"timezone": "UTC"})]
            ),
            tags=["parallel-calls", "reasoning", "x-sequential-calls", "x-recipient-in-role"],
            notes="test_tool_index_across_calls: the two deltas concatenated. Harmony has no parallel-call "
            "wrapper; consecutive call messages are separated by <|call|>, which is a stop token, so this "
            "output only reaches a parser when the stop is not applied (e.g. ignore_eos, vLLM #50690).",
        ),
        dict(
            name="vllm-call-then-final",
            assumes=["stop_token_ignored"],
            src=VLLM,
            line="L703-L718",
            text="<|channel|>analysis<|message|>Reasoning about query...<|end|>"
            "<|start|>assistant to=functions.search<|channel|>commentary"
            '<|constrain|>json<|message|>{"query": "vllm"}<|call|>'
            "<|start|>assistant<|channel|>final<|message|>Done",
            tools=[tool("search", obj(query=S))],
            expected=expected("Done", "Reasoning about query...", [("search", {"query": "vllm"})]),
            tags=["single-call", "reasoning", "text-after-call", "x-recipient-in-role"],
            notes="test_cross_channel_with_tool. As in vllm-sequential-calls, text after <|call|> is only seen "
            "when the stop token is not applied.",
        ),
        dict(
            name="vllm-truncated-final",
            src=VLLM,
            line="L427-L441",
            text="<|channel|>analysis<|message|>I'm thinking.<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
            "I'm in the middle of answering",
            tools=[T_WEATHER_LOC],
            expected=expected("I'm in the middle of answering", "I'm thinking.", []),
            tags=["no-call", "reasoning", "truncated"],
            notes="test_truncated_output: the final message is cut off (max_tokens).",
        ),
        dict(
            name="vllm-interrupted-final-only",
            src=VLLM,
            line="L399-L411",
            text="<|channel|>final<|message|>I'm in the middle of answering",
            tools=[T_WEATHER_LOC],
            expected=expected("I'm in the middle of answering", None, []),
            tags=["no-call", "truncated"],
            notes="test_interrupted_first_message.",
        ),
        dict(
            name="vllm-malformed-headers",
            src=VLLM,
            line="L79-L85",
            text="<|channel|>analysis<|message|>thinking<|end|>"
            "<|start|>assistant<|channel|>commentary<|message|>thinking<|end|>"
            '<|start|>assistant<|channel|>final {"answer": "hi"}<|return|>',
            tools=[T_WEATHER_LOC],
            expected_error={
                "reason": "The last message header has no <|message|> token (malformed). There is no tool call; "
                "a parser must not crash (vLLM recovers the raw text as content).",
                "accept": ["no_tool_calls", "content_passthrough"],
            },
            tags=["malformed", "no-call", "reasoning"],
            notes="malformed_msgs_str fixture, joined as in test_malformed_msgs_recovers_raw_content.",
        ),
        # ---------------------------------------------------------------- llama.cpp tests/test-chat.cpp (MIT)
        dict(
            name="llamacpp-commentary-content-only",
            src=LLAMA,
            line="L6404-L6405",
            text="<|channel|>commentary<|message|>Hello, world!\nWhat's up?",
            tools=[T_SPECIAL],
            expected=expected("Hello, world!\nWhat's up?", None, []),
            tags=["no-call", "x-preamble-commentary"],
            notes="'Basic content only - commentary channel'.",
        ),
        dict(
            name="llamacpp-recipient-in-channel-analysis",
            src=LLAMA,
            line="L6437-L6442",
            text='<|channel|>analysis to=functions.special_function<|message|>{"arg1": 1}',
            tools=[T_SPECIAL],
            expected=expected(None, None, [("special_function", {"arg1": 1})]),
            tags=["single-call", "numeric-arguments", "x-recipient-in-channel", "x-call-on-analysis"],
            notes="'Tool call with recipient in channel header'.",
        ),
        dict(
            name="llamacpp-recipient-in-channel-commentary",
            src=LLAMA,
            line="L6451-L6456",
            text='<|channel|>commentary to=functions.special_function<|message|>{"arg1": 1}',
            tools=[T_SPECIAL],
            expected=expected(None, None, [("special_function", {"arg1": 1})]),
            tags=["single-call", "numeric-arguments", "x-recipient-in-channel", "x-no-content-type"],
            notes="'Tool call in commentary channel (channel header variant)'.",
        ),
        dict(
            name="llamacpp-recipient-in-role-constrain",
            src=LLAMA,
            line="L6444-L6449",
            text=' to=functions.special_function<|channel|>analysis <|constrain|>json<|message|>{"arg1": 1}',
            tools=[T_SPECIAL],
            expected=expected(None, None, [("special_function", {"arg1": 1})]),
            tags=["single-call", "numeric-arguments", "x-recipient-in-role", "x-call-on-analysis"],
            notes="'Tool call with constraint': the output starts right after the <|start|>assistant prompt.",
        ),
        dict(
            name="llamacpp-edit-code-arguments",
            src=LLAMA,
            line="L6467-L6502",
            text="<|channel|>analysis<|message|>Thinking about edit...<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.edit <|constrain|>json"
            '<|message|>{"oldString": "if (part < railCount - 1) {", "newString": "if (part < 4) {", '
            '"replaceAll": false}',
            tools=[T_EDIT],
            expected=expected(
                None,
                "Thinking about edit...",
                [
                    (
                        "edit",
                        {
                            "oldString": "if (part < railCount - 1) {",
                            "newString": "if (part < 4) {",
                            "replaceAll": False,
                        },
                    )
                ],
            ),
            tags=["single-call", "reasoning", "x-recipient-in-channel", "x-code-in-arguments"],
            notes="'Complex tool calling'.",
        ),
        dict(
            name="llamacpp-builtin-python-recipient",
            src=LLAMA,
            line="L6529-L6536",
            text="<|channel|>analysis<|message|>I will execute python to say hello<|end|>"
            '<|start|>assistant<|channel|>commentary to=python <|constrain|>code<|message|>print("hello")',
            tools=[T_SPECIAL],
            expected=expected(None, "I will execute python to say hello", []),
            tags=["no-call", "reasoning", "x-builtin-recipient", "x-recipient-in-channel"],
            notes="Unsolicited call to the built-in python tool, which was not offered: it is not a function "
            "tool call, and llama.cpp returns empty content.",
        ),
        dict(
            name="llamacpp-stray-commentary-header",
            src=LLAMA,
            line="L6540-L6546",
            text="<|channel|>commentary to=assistant<|channel|>analysis<|message|>I'm\nthinking<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Hello, world!\nWhat's up?",
            tools=[T_SPECIAL],
            expected_error={
                "reason": "Stray '<|channel|>commentary to=assistant' header before the analysis channel. llama.cpp "
                "recovers reasoning 'I'm\\nthinking' and content 'Hello, world!\\nWhat's up?'; openai-harmony's "
                "non-strict parser keeps the content but drops the reasoning. Either way there is no tool call and "
                "the turn must not error.",
                "accept": ["no_tool_calls"],
            },
            tags=["no-call", "reasoning", "malformed", "x-malformed-channel"],
            notes="Edge case '<|channel|>commentary to=assistant' before reasoning (the stray header llama.cpp "
            "#21286 tolerates).",
        ),
        # ---------------------------------------------------------------- openai/harmony tests + spec
        dict(
            name="harmony-test-channel-first-constrain-adjacent",
            src=HARMONY_T,
            line="L300-L322",
            text="<|start|>assistant<|channel|>commentary to=functions.get_weather"
            '<|constrain|>json<|message|>{"latitude":48.8566,"longitude":2.3522}<|call|>',
            strip_prompt=True,
            tools=[T_WEATHER_LATLON],
            expected=expected(None, None, [("get_weather", {"latitude": 48.8566, "longitude": 2.3522})]),
            tags=["single-call", "numeric-arguments", "x-recipient-in-channel", "x-constrain-nospace"],
            notes="test_tool_call_with_channel_before_recipient_and_constrain_adjacent.",
        ),
        dict(
            name="harmony-test-role-recipient-constrain-adjacent",
            src=HARMONY_T,
            line="L272-L292",
            text="<|start|>assistant to=functions.get_weather<|channel|>commentary"
            '<|constrain|>json<|message|>{"location": "Tokyo"}<|call|>',
            strip_prompt=True,
            tools=[T_WEATHER_LOC],
            expected=expected(None, None, [("get_weather", {"location": "Tokyo"})]),
            tags=["single-call", "x-recipient-in-role", "x-constrain-nospace"],
            notes="test_tool_call_with_constrain_marker_adjacent: 'the model might not output a space before "
            "constrain'.",
        ),
        dict(
            name="harmony-spec-receiving-tool-calls",
            src=HARMONY_S,
            line="L395-L403",
            text="<|channel|>analysis<|message|>Need to use function get_weather.<|end|><|start|>assistant"
            '<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"San Francisco"}'
            "<|call|>",
            tools=[T_WEATHER_LOC],
            expected=expected(
                None, "Need to use function get_weather.", [("get_weather", {"location": "San Francisco"})]
            ),
            tags=["single-call", "reasoning", "x-recipient-in-channel", "x-constrain-token"],
            notes="The Harmony format spec's example of what the model generates when it calls a tool "
            "(section 'Receiving tool calls'), from the reference library's docs.",
        ),
        dict(
            name="harmony-spec-preamble-action-plan",
            src=HARMONY_S,
            line="L468-L474",
            text="<|channel|>analysis<|message|>{long chain of thought}<|end|><|start|>assistant<|channel|>commentary"
            "<|message|>**Action plan**:\n1. Generate an HTML file\n2. Generate a JavaScript for the Node.js server\n"
            "3. Start the server\n---\nWill start executing the plan step by step<|end|><|start|>assistant"
            "<|channel|>commentary to=functions.generate_file<|constrain|>json<|message|>"
            '{"template": "basic_html", "path": "index.html"}<|call|>',
            tools=[T_GENERATE_FILE],
            expected=expected(
                "**Action plan**:\n1. Generate an HTML file\n2. Generate a JavaScript for the Node.js server\n"
                "3. Start the server\n---\nWill start executing the plan step by step",
                "{long chain of thought}",
                [("generate_file", {"template": "basic_html", "path": "index.html"})],
            ),
            tags=[
                "single-call",
                "reasoning",
                "text-before-call",
                "x-preamble-commentary",
                "x-recipient-in-channel",
                "x-constrain-nospace",
            ],
            notes="The Harmony spec's preamble example (section 'Preambles'). '{long chain of thought}' is the "
            "spec's literal placeholder text.",
        ),
    ]
    return cases + bug_reports(enc)


def bug_reports(enc: Any) -> list[dict[str, Any]]:
    def render(msg: Message) -> str:
        cfg = RenderConversationConfig(auto_drop_analysis=False)
        user = [Message.from_role_and_content(Role.USER, "x")]
        prompt = enc.render_conversation_for_completion(Conversation.from_messages(user), Role.ASSISTANT, config=cfg)
        full = enc.render_conversation(Conversation.from_messages([*user, msg]), config=cfg)
        return str(enc.decode_utf8(full[len(prompt) :]))

    unknown = render(Message.from_role_and_content(Role.ASSISTANT, "some text").with_channel("comment"))
    issue = "https://github.com/vllm-project/vllm/issues/58384"
    llama_issue = "https://github.com/ggml-org/llama.cpp/issues/27720"
    return [
        dict(
            name="bug-unknown-channel-comment",
            src=Src("bug_report", issue, "issue-58384", "NOASSERTION"),
            line=None,
            text=unknown,
            tools=[T_WEATHER_LOC],
            expected_error={
                "reason": "The model emitted a message on the out-of-spec channel 'comment'. The report asks for "
                "it to be dropped (with a warning), never to fail the request.",
                "accept": ["no_tool_calls", "content_passthrough"],
            },
            tags=["malformed", "no-call", "regression", "x-malformed-channel"],
            notes="Raw output rendered with openai-harmony from the issue's own reproduction: "
            "Message.from_role_and_content(Role.ASSISTANT, 'some text').with_channel('comment'). The trailing "
            "<|end|> is kept as rendered.",
        ),
        dict(
            name="bug-garbled-channel-question-marks",
            src=Src("bug_report", llama_issue, "issue-27720", "NOASSERTION"),
            line=None,
            text="<|channel|>??<|end|><|start|>assistant<|channel|>analysis<|message|>We must comply with the gating "
            'rules. Input: "should I seal the driveway this weekend"',
            tools=[T_WEATHER_LOC],
            expected_error={
                "reason": "Malformed channel header '<|channel|>??' before a well-formed analysis message. The "
                "report's complaint is that the whole turn errors; a parser should skip the garbage and keep going.",
                "accept": ["no_tool_calls", "content_passthrough"],
            },
            tags=["malformed", "truncated", "no-call", "regression", "x-malformed-channel"],
            notes="Quoted from the issue's log line '1.06 W common_chat_peg_parse: unparsed peg-native output: "
            "<|channel|>??<|end|>...'. The log truncates the output (' ...'), so this fixture ends where the "
            "quoted text ends and is also a truncated output.",
        ),
        dict(
            name="bug-garbled-channel-commentary-question",
            src=Src("bug_report", llama_issue, "issue-27720", "NOASSERTION"),
            line=None,
            text="<|channel|>commentary?commentary?We need temperature difference. Also need outside temperature. "
            "Get outdoor temperature.<|end|><|start|>assistant<|channel|>analysis<|message|>Need GetLiveContext "
            "for Outdoor Temperature.",
            tools=[T_WEATHER_LOC],
            expected_error={
                "reason": "Malformed channel header ('commentary?commentary?' with no <|message|>) followed by a "
                "well-formed analysis message. There is no tool call; the turn must not error.",
                "accept": ["no_tool_calls", "content_passthrough"],
            },
            tags=["malformed", "truncated", "no-call", "regression", "x-malformed-channel"],
            notes="Quoted from the issue's log line at 179.42 s, which is cut at '<|end|' by the log; this fixture "
            "stops before that partial token.",
        ),
    ]


def main() -> None:
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    tok = hf_tokenizer()
    recs = []
    for c in build(enc):
        text: str = c["text"]
        notes = [c.get("notes") or ""]
        if c.get("strip_prompt"):
            assert text.startswith("<|start|>assistant")
            text = text.removeprefix("<|start|>assistant")
            notes.append("The source's leading <|start|>assistant is the generation prompt and was removed.")
        for stop in ("<|call|>", "<|return|>"):
            if text.endswith(stop):
                text = text.removesuffix(stop)
                notes.append(f"The trailing stop token {stop} was removed (raw_output ends before the stop token).")
        ids = list(enc.encode(text, allowed_special="all"))
        check_ids(tok, ids, text)
        recs.append(
            record(
                name=c["name"],
                provenance=c["src"].prov(c["line"]),
                tools=c["tools"],
                raw_output=text,
                output_token_ids=ids,
                expected=c.get("expected"),
                expected_error=c.get("expected_error"),
                tags=c["tags"],
                notes=" ".join(n for n in notes if n),
            )
        )
        if "assumes" in c:
            recs[-1]["assumes"] = c["assumes"]
    write_jsonl(OUT_DIR / "imported.jsonl", recs)


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "transformers==5.17.0",
#     "tokenizers==0.23.2",
#     "tiktoken==0.12.0",
#     "jinja2==3.1.6",
#     "huggingface-hub==1.33.0",
#     "jsonschema==4.25.1",
# ]
# ///
"""Build the ``minimax`` fixture corpus (MiniMax M2, M2.1, M2.5, M2.7, M3, M3.1-Flash).

Provenance of every fixture written here is ``template_render``: an assistant
message rendered through the OFFICIAL chat template of the reference model
(``chat_template.jinja`` at a pinned revision: MiniMax-M2.1 for the M2.x
sub-format, MiniMax-M3 for the M3 sub-format) with
``apply_chat_template(tokenize=True)``; see ``_shared.core.render``. Truncated
fixtures are a token prefix of such a render, which is what ``max_tokens``
produces. Sibling repositories (M2, M2.5, M2.7; M3.1-Flash) are listed in
``models`` only when their own template and tokenizer render every fixture of
their sub-format byte-identically.

Every M2.x template opens ``<think>\\n`` in the generation prompt, so the raw
completion starts inside the reasoning (no opening tag) and each fixture has
reasoning. The template has no switch that turns thinking off.

The M3 template prefixes every structural tag with the ``]<]minimax[>[`` token,
names each parameter element after its key, and expands nested objects and
arrays into child elements. Its default (``thinking_mode`` unset, "adaptive")
generation prompt is ``]~b]ai\\n``, so the completion carries ``<mm:think>`` or
a bare ``</mm:think>`` itself. M3 fixtures are tagged ``x-m3``.

Run from the repo root (deterministic; re-running leaves ``git diff`` clean)::

    uv run --script scripts/fixtures/minimax/build.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _shared.core import (  # noqa: E402
    REPO_ROOT,
    Message,
    Provenance,
    Record,
    Rendered,
    Tool,
    assistant,
    call,
    cut_before,
    expected_from,
    render,
    sha256_file,
    tool,
    validate_records,
    write_jsonl,
)
from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SLUG = "minimax"
OUT = REPO_ROOT / "fixtures" / SLUG
GENERATOR = f"scripts/fixtures/{SLUG}/build.py"

# --------------------------------------------------------------------------- models

REFERENCE = ("MiniMaxAI/MiniMax-M2.1", "917dbb7d0761e24d8623c0a5d97624550181c8e6")
LICENSE = "LicenseRef-minimax-m2.1-modified-mit"
SIBLINGS = (
    ("MiniMaxAI/MiniMax-M2.7", "0a6536713b0b03f3be97494e5cf8d22f668a3164"),
    ("MiniMaxAI/MiniMax-M2.5", "2515a3ea0afec1dd22d0c7794b3a91557804d195"),
    ("MiniMaxAI/MiniMax-M2", "04312f0cfbdeefb912575595b7c045512a29293c"),
)
"""(repo, revision) whose rendered output is checked to be identical, then listed in ``models``."""

M3_REFERENCE = ("MiniMaxAI/MiniMax-M3", "d95e30cb912fe3021bb9c2ba66a8ce60b6c937a9")
M3_LICENSE = "LicenseRef-minimax-m3-community"
M3_SIBLINGS = (("MiniMaxAI/MiniMax-M3.1-Flash", "ce7f4df87d2e05262f5775f373dac4772f134041"),)
NS = "]<]minimax[>["
"""The M3 namespace token that prefixes every structural tag."""


def stop_ids(repo: str, revision: str) -> list[int]:
    gen = json.loads(Path(hf_hub_download(repo, "generation_config.json", revision=revision)).read_text())
    stops = gen["eos_token_id"]
    return [int(s) for s in (stops if isinstance(stops, list) else [stops])]


# --------------------------------------------------------------------------- tools

T_WEATHER = tool(
    "get_weather",
    "Get the current weather for a city.",
    {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
    ["city"],
)
T_SEARCH = tool(
    "search_web",
    "Search the web.",
    {
        "query": {"type": "string"},
        "filters": {
            "type": "object",
            "properties": {
                "site": {"type": "string"},
                "max_results": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    ["query"],
)
T_TIME = tool("get_time", "Get the current UTC time.")
T_WRITE = tool(
    "write_file",
    "Write a text file.",
    {"path": {"type": "string"}, "content": {"type": "string"}},
    ["path", "content"],
)
T_TIMER = tool(
    "set_timer",
    "Start a countdown timer.",
    {"seconds": {"type": "integer"}, "label": {"type": "string"}},
    ["seconds"],
)
T_EVENT = tool(
    "create_event",
    "Create a calendar event.",
    {
        "title": {"type": "string"},
        "attendees": {"type": "array", "items": {"type": "string"}},
        "duration_minutes": {"type": "integer"},
        "all_day": {"type": "boolean"},
        "reminder_minutes": {"type": ["integer", "null"]},
    },
    ["title"],
)
T_ORDER = tool(
    "lookup_order",
    "Look up an order by its id.",
    {"order_id": {"type": "string"}, "reference": {"type": "string"}},
    ["order_id"],
)
T_SETTINGS = tool(
    "update_settings",
    "Update user settings.",
    {"settings": {"type": "object"}, "dry_run": {"type": "boolean"}},
    ["settings"],
)
T_CONVERT = tool(
    "convert_units",
    "Convert a value between units.",
    {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}},
    ["value", "from_unit", "to_unit"],
)
TOOLS: list[Tool] = [T_WEATHER, T_SEARCH, T_TIME, T_WRITE, T_TIMER, T_EVENT, T_ORDER, T_SETTINGS, T_CONVERT]
T_CREATE_ORDER = tool(
    "create_order",
    "Create an order.",
    {
        "user_id": {"type": "integer"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"sku": {"type": "string"}, "quantity": {"type": "integer"}},
            },
        },
        "gift": {"type": "boolean"},
    },
    ["user_id", "items"],
)
M3_TOOLS: list[Tool] = [T_WEATHER, T_SEARCH, T_TIME, T_WRITE, T_TIMER, T_EVENT, T_ORDER, T_CONVERT, T_CREATE_ORDER]
"""M3 renders nested values as XML without types, so every M3 tool has a complete schema (no bare ``object``)."""

LONG_TEXT = "\n".join(f"{i:03d}: The quick brown fox jumps over the lazy dog." for i in range(60))


# --------------------------------------------------------------------------- cases


@dataclass
class Case:
    name: str
    tags: list[str]
    msg: Message
    context: list[Message] = field(default_factory=list)
    notes: str | None = None
    user: str = "Please help."


def tool_results(msg: Message) -> list[Message]:
    return [{"role": "tool", "content": '{"ok": true}'} for _ in msg.get("tool_calls") or []]


CASES: list[Case] = [
    Case(
        "single-call",
        ["single-call", "reasoning-prefilled"],
        assistant(
            reasoning="The user wants the current weather in Paris, in Celsius.",
            calls=[call("get_weather", {"city": "Paris", "unit": "celsius"})],
        ),
        user="What's the weather in Paris?",
    ),
    Case(
        "parallel-same-tool",
        ["parallel-calls", "reasoning-prefilled"],
        assistant(
            reasoning="Two cities, so two independent weather lookups.",
            calls=[call("get_weather", {"city": "Paris"}), call("get_weather", {"city": "Tokyo"})],
        ),
        user="Weather in Paris and Tokyo?",
        notes="Both <invoke> blocks sit inside ONE <minimax:tool_call> section.",
    ),
    Case(
        "parallel-mixed-nested",
        ["parallel-calls", "nested-json", "reasoning-prefilled"],
        assistant(
            reasoning="Check the weather, then search for cafés.",
            calls=[
                call("get_weather", {"city": "Zürich", "unit": "celsius"}),
                call(
                    "search_web",
                    {
                        "query": "best cafés",
                        "filters": {"site": "example.com", "max_results": 5, "tags": ["coffee", "wifi"]},
                    },
                ),
            ],
        ),
    ),
    Case(
        "content-before-call",
        ["single-call", "text-before-call", "reasoning-prefilled"],
        assistant(
            content="I'll look that up for you.",
            reasoning="Need live data, so call the tool.",
            calls=[call("get_weather", {"city": "Berlin"})],
        ),
        notes="The template writes '</think>\\n\\n' before the content and '\\n' before <minimax:tool_call>; the "
        "correct content is the message text without those separators.",
    ),
    Case(
        "no-call",
        ["no-call", "reasoning-prefilled"],
        assistant(content="Paris is the capital of France.", reasoning="A simple fact; no tool is needed."),
        user="What is the capital of France?",
    ),
    Case(
        "nested-object",
        ["single-call", "nested-json", "numeric-arguments", "reasoning-prefilled"],
        assistant(
            reasoning="Apply the dark theme and disable SMS.",
            calls=[
                call(
                    "update_settings",
                    {
                        "settings": {
                            "theme": {"mode": "dark", "accent": [255, 128, 0]},
                            "notifications": {"email": True, "sms": False},
                        },
                        "dry_run": False,
                    },
                )
            ],
        ),
    ),
    Case(
        "unicode-emoji",
        ["single-call", "unicode", "reasoning-prefilled"],
        assistant(
            reasoning="Search with the user's exact wording.",
            calls=[call("search_web", {"query": "Zürich café ☕ 東京の天気 🌸"})],
        ),
    ),
    Case(
        "empty-arguments",
        ["single-call", "empty-arguments", "reasoning-prefilled"],
        assistant(reasoning="Just fetch the time.", calls=[call("get_time", {})]),
    ),
    Case(
        "marker-in-arguments",
        ["single-call", "marker-in-arguments", "reasoning-prefilled"],
        assistant(
            reasoning="Write the format note to the docs file.",
            calls=[
                call(
                    "write_file",
                    {
                        "path": "docs/format.md",
                        "content": 'Wrap calls in <minimax:tool_call> and </minimax:tool_call>; each <invoke name="x"> '
                        'holds <parameter name="k"> elements.',
                    },
                )
            ],
        ),
        notes="The string value contains <minimax:tool_call>, </minimax:tool_call> and the opening <invoke> and "
        "<parameter> tags. It does not contain </parameter> or </invoke>, so the value is still delimited "
        "unambiguously.",
    ),
    Case(
        "typed-values",
        ["single-call", "numeric-arguments", "x-typed-values", "reasoning-prefilled"],
        assistant(
            reasoning="Create the standup with a null reminder.",
            calls=[
                call(
                    "create_event",
                    {
                        "title": "Standup",
                        "attendees": ["ana@example.com", "bo@example.com"],
                        "duration_minutes": 15,
                        "all_day": False,
                        "reminder_minutes": None,
                    },
                )
            ],
        ),
        notes="Non-string values are rendered with tojson (15, false, null, a JSON array); strings are raw.",
    ),
    Case(
        "number-looking-string",
        ["single-call", "x-number-looking-string", "reasoning-prefilled"],
        assistant(
            reasoning="Look up the order id exactly as given.",
            calls=[call("lookup_order", {"order_id": "123_456", "reference": "0042"})],
        ),
        notes="String values are raw, so 123_456 and 0042 must stay strings (the schema says string).",
    ),
    Case(
        "schema-coercion",
        ["single-call", "numeric-arguments", "x-schema-coercion", "reasoning-prefilled"],
        assistant(
            reasoning="Three second timer labelled 3.",
            calls=[call("set_timer", {"seconds": 3, "label": "3"})],
        ),
        notes="Both values render as the bare text 3; only the tool schema tells the integer from the string.",
    ),
    Case(
        "string-escapes",
        ["single-call", "string-escapes", "reasoning-prefilled"],
        assistant(
            reasoning="Save the note verbatim.",
            calls=[
                call(
                    "write_file",
                    {
                        "path": "C:\\Users\\ana\\notes.txt",
                        "content": 'line 1\n\tline 2 with "quotes" and a backslash \\ end',
                    },
                )
            ],
        ),
        notes="String values are written raw (real newline, tab, quotes and backslashes), not JSON-escaped.",
    ),
    Case(
        "significant-whitespace",
        ["single-call", "x-significant-whitespace", "reasoning-prefilled"],
        assistant(
            reasoning="Keep the spacing exactly.",
            calls=[call("write_file", {"path": "pad.txt", "content": "  two  spaces\n"})],
        ),
        notes="Values have no wrapping newlines, so leading and trailing whitespace inside <parameter> is data.",
    ),
    Case(
        "long-arguments",
        ["single-call", "long-arguments", "reasoning-prefilled"],
        assistant(
            reasoning="Write the long file.",
            calls=[call("write_file", {"path": "fox.txt", "content": LONG_TEXT})],
        ),
    ),
    Case(
        "json-looking-string",
        ["single-call", "x-json-looking-string", "reasoning-prefilled"],
        assistant(
            reasoning="Search for the literal JSON text.",
            calls=[call("search_web", {"query": '{"a": 1}'})],
        ),
        notes="The string parameter holds text that is valid JSON; the schema says string, so it must stay "
        'the string "{\\"a\\": 1}".',
    ),
    Case(
        "call-order-vs-tool-order",
        ["parallel-calls", "reasoning-prefilled", "x-call-index-vs-tool-index"],
        assistant(
            reasoning="Weather, then search, then weather again.",
            calls=[
                call("get_weather", {"city": "Oslo"}),
                call("search_web", {"query": "Oslo museums"}),
                call("get_weather", {"city": "Bergen"}),
            ],
        ),
        notes="The same tool is called twice around another one; call indices must be 0,1,2, not the tools-list "
        "position.",
    ),
    Case(
        "multi-turn-second-call",
        ["single-call", "multi-turn", "reasoning-prefilled"],
        assistant(
            reasoning="Now convert the temperature.",
            calls=[call("convert_units", {"value": 21.5, "from_unit": "celsius", "to_unit": "fahrenheit"})],
        ),
        context=[
            {"role": "user", "content": "Weather in Rome, in Fahrenheit please."},
            assistant(reasoning="Get the weather first.", calls=[call("get_weather", {"city": "Rome"})]),
            {"role": "tool", "content": '{"temp_c": 21.5}'},
        ],
    ),
]

M3_CASES: list[Case] = [
    Case(
        "single-call",
        ["single-call", "reasoning"],
        assistant(
            reasoning="The user wants the current weather in Paris, in Celsius.",
            calls=[call("get_weather", {"city": "Paris", "unit": "celsius"})],
        ),
        user="What's the weather in Paris?",
    ),
    Case(
        "parallel-same-tool",
        ["parallel-calls", "reasoning"],
        assistant(
            reasoning="Two cities, so two independent weather lookups.",
            calls=[call("get_weather", {"city": "Paris"}), call("get_weather", {"city": "Tokyo"})],
        ),
        user="Weather in Paris and Tokyo?",
        notes="Both <invoke> elements sit inside ONE ]<]minimax[>[<tool_call> section.",
    ),
    Case(
        "parallel-mixed-nested",
        ["parallel-calls", "nested-json", "numeric-arguments", "reasoning"],
        assistant(
            reasoning="Check the weather, then search for cafés.",
            calls=[
                call("get_weather", {"city": "Zürich", "unit": "celsius"}),
                call(
                    "search_web",
                    {
                        "query": "best cafés",
                        "filters": {"site": "example.com", "max_results": 5, "tags": ["coffee", "wifi"]},
                    },
                ),
            ],
        ),
        notes="The nested object becomes child elements and the array becomes <item> elements; only the tool "
        "schema says that max_results is an integer and tags is an array.",
    ),
    Case(
        "content-before-call",
        ["single-call", "text-before-call", "reasoning"],
        assistant(
            content="I'll look that up for you.",
            reasoning="Need live data, so call the tool.",
            calls=[call("get_weather", {"city": "Berlin"})],
        ),
        notes="The template writes the content between </mm:think> and the tool-call section with no separator.",
    ),
    Case(
        "no-call",
        ["no-call", "reasoning"],
        assistant(content="Paris is the capital of France.", reasoning="A simple fact; no tool is needed."),
        user="What is the capital of France?",
    ),
    Case(
        "no-reasoning-call",
        ["single-call", "x-m3-bare-think-close"],
        assistant(calls=[call("get_weather", {"city": "Madrid"})]),
        user="Weather in Madrid?",
        notes="Without reasoning the template writes a bare </mm:think> first. It is format glue, not content, and "
        "there is no reasoning.",
    ),
    Case(
        "no-reasoning-no-call",
        ["no-call", "x-m3-bare-think-close"],
        assistant(content="Paris is the capital of France."),
        user="What is the capital of France?",
        notes="Without reasoning the template writes a bare </mm:think> first. It is format glue, not content.",
    ),
    Case(
        "array-of-objects",
        ["single-call", "nested-json", "numeric-arguments", "reasoning"],
        assistant(
            reasoning="Order two books and one pen as a gift.",
            calls=[
                call(
                    "create_order",
                    {
                        "user_id": 42,
                        "items": [{"sku": "book-001", "quantity": 2}, {"sku": "pen-7", "quantity": 1}],
                        "gift": True,
                    },
                )
            ],
        ),
        notes="Each array element is an <item> holding one element per key. The integers render as bare text and "
        "true as JSON; the schema gives their types.",
    ),
    Case(
        "single-item-array",
        ["single-call", "nested-json", "reasoning", "x-m3-single-item-array"],
        assistant(
            reasoning="Search with a single tag.",
            calls=[call("search_web", {"query": "espresso", "filters": {"tags": ["coffee"]}})],
        ),
        notes="One <item> element. The schema says tags is an array, so the value is a one-element list, not a string.",
    ),
    Case(
        "empty-array",
        ["single-call", "reasoning", "x-m3-empty-container"],
        assistant(
            reasoning="Create the event without attendees.",
            calls=[call("create_event", {"title": "Focus time", "attendees": []})],
        ),
        notes="An empty list renders as an empty element (]<]minimax[>[<attendees>]<]minimax[>[</attendees>). The "
        "schema says array, so the correct value is [], not the empty string.",
    ),
    Case(
        "unicode-emoji",
        ["single-call", "unicode", "reasoning"],
        assistant(
            reasoning="Search with the user's exact wording.",
            calls=[call("search_web", {"query": "Zürich café ☕ 東京の天気 🌸"})],
        ),
    ),
    Case(
        "empty-arguments",
        ["single-call", "empty-arguments", "reasoning"],
        assistant(reasoning="Just fetch the time.", calls=[call("get_time", {})]),
    ),
    Case(
        "marker-in-arguments",
        ["single-call", "marker-in-arguments", "reasoning"],
        assistant(
            reasoning="Write the format note to the docs file.",
            calls=[
                call(
                    "write_file",
                    {
                        "path": "docs/format.md",
                        "content": 'Wrap calls in <tool_call> and </tool_call>; each <invoke name="x"> holds '
                        "<content> and </content> elements, closed by </invoke>.",
                    },
                )
            ],
        ),
        notes="The string value contains <tool_call>, </tool_call>, <invoke name=...>, </invoke> and this "
        "parameter's own <content>/</content> tags, all without the ]<]minimax[>[ prefix. Only a tag with the "
        "prefix is structure, so the value is still delimited unambiguously. (The prefix token itself cannot be "
        "escaped, so a value containing it is not a fixture.)",
    ),
    Case(
        "typed-values",
        ["single-call", "numeric-arguments", "x-typed-values", "reasoning"],
        assistant(
            reasoning="Create the standup.",
            calls=[
                call(
                    "create_event",
                    {
                        "title": "Standup",
                        "attendees": ["ana@example.com", "bo@example.com"],
                        "duration_minutes": 15,
                        "all_day": False,
                    },
                )
            ],
        ),
        notes="Integers render as bare text, booleans as JSON (false), arrays as <item> elements, strings raw; "
        "only the schema gives the types.",
    ),
    Case(
        "number-looking-string",
        ["single-call", "x-number-looking-string", "reasoning"],
        assistant(
            reasoning="Look up the order id exactly as given.",
            calls=[call("lookup_order", {"order_id": "123_456", "reference": "0042"})],
        ),
        notes="String values are raw, so 123_456 and 0042 must stay strings (the schema says string).",
    ),
    Case(
        "schema-coercion",
        ["single-call", "numeric-arguments", "x-schema-coercion", "reasoning"],
        assistant(
            reasoning="Three second timer labelled 3.",
            calls=[call("set_timer", {"seconds": 3, "label": "3"})],
        ),
        notes="Both values render as the bare text 3; only the tool schema tells the integer from the string.",
    ),
    Case(
        "string-escapes",
        ["single-call", "string-escapes", "reasoning"],
        assistant(
            reasoning="Save the note verbatim.",
            calls=[
                call(
                    "write_file",
                    {
                        "path": "C:\\Users\\ana\\notes.txt",
                        "content": 'line 1\n\tline 2 with "quotes" and a backslash \\ end',
                    },
                )
            ],
        ),
        notes="String values are written raw (real newline, tab, quotes and backslashes), not JSON-escaped.",
    ),
    Case(
        "significant-whitespace",
        ["single-call", "x-significant-whitespace", "reasoning"],
        assistant(
            reasoning="Keep the spacing exactly.",
            calls=[call("write_file", {"path": "pad.txt", "content": "  two  spaces\n"})],
        ),
        notes="Values have no wrapping whitespace, so leading and trailing whitespace inside an element is data.",
    ),
    Case(
        "long-arguments",
        ["single-call", "long-arguments", "reasoning"],
        assistant(
            reasoning="Write the long file.",
            calls=[call("write_file", {"path": "fox.txt", "content": LONG_TEXT})],
        ),
    ),
    Case(
        "json-looking-string",
        ["single-call", "x-json-looking-string", "reasoning"],
        assistant(
            reasoning="Search for the literal JSON text.",
            calls=[call("search_web", {"query": '{"a": 1}'})],
        ),
        notes="The string parameter holds text that is valid JSON; the schema says string, so it must stay "
        'the string "{\\"a\\": 1}".',
    ),
    Case(
        "call-order-vs-tool-order",
        ["parallel-calls", "reasoning", "x-call-index-vs-tool-index"],
        assistant(
            reasoning="Weather, then search, then weather again.",
            calls=[
                call("get_weather", {"city": "Oslo"}),
                call("search_web", {"query": "Oslo museums"}),
                call("get_weather", {"city": "Bergen"}),
            ],
        ),
        notes="The same tool is called twice around another one; call indices must be 0,1,2, not the tools-list "
        "position.",
    ),
    Case(
        "multi-turn-second-call",
        ["single-call", "multi-turn", "numeric-arguments", "reasoning"],
        assistant(
            reasoning="Now convert the temperature.",
            calls=[call("convert_units", {"value": 21.5, "from_unit": "celsius", "to_unit": "fahrenheit"})],
        ),
        context=[
            {"role": "user", "content": "Weather in Rome, in Fahrenheit please."},
            assistant(reasoning="Get the weather first.", calls=[call("get_weather", {"city": "Rome"})]),
            {"role": "tool", "content": '{"temp_c": 21.5}'},
        ],
        notes="The number 21.5 renders as bare text; the schema says number.",
    ),
]
"""M3 cases (ids ``minimax/m3-<name>``). No argument is ``None``: the M3 template omits such keys."""


# --------------------------------------------------------------------------- build


@dataclass(frozen=True)
class SubFormat:
    """One sub-format of the family: its reference model, tools, cases and id prefix."""

    tag: str
    reference: tuple[str, str]
    license: str
    siblings: tuple[tuple[str, str], ...]
    tools: list[Tool]
    cases: list[Case]
    id_prefix: str

    @property
    def template_url(self) -> str:
        return f"https://huggingface.co/{self.reference[0]}/blob/{self.reference[1]}/chat_template.jinja"

    @property
    def tokenizer_pin(self) -> dict[str, str]:
        return {"repo": self.reference[0], "revision": self.reference[1], "mode": "hf"}

    def by_name(self, name: str) -> Case:
        return next(c for c in self.cases if c.name == name)


M2 = SubFormat("x-m2", REFERENCE, LICENSE, SIBLINGS, TOOLS, CASES, "")
M3 = SubFormat("x-m3", M3_REFERENCE, M3_LICENSE, M3_SIBLINGS, M3_TOOLS, M3_CASES, "m3-")


def render_case(tok: Any, stops: list[int], case: Case, tools: list[Tool]) -> Rendered:
    context = case.context or [{"role": "user", "content": case.user}]
    return render(
        tok,
        context,
        case.msg,
        tools=tools,
        stop_ids=stops,
        followups=tool_results(case.msg),
        require_stop=True,
    )


def default_gen_prompt(tok: Any) -> str:
    probe = [{"role": "user", "content": "x"}]
    full = tok.apply_chat_template(probe, add_generation_prompt=True, tokenize=False)
    bare = tok.apply_chat_template(probe, add_generation_prompt=False, tokenize=False)
    return str(full)[len(str(bare)) :]


def provenance(sub: SubFormat, sha: str) -> Provenance:
    return Provenance("template_render", sub.template_url, sub.reference[1], sub.license, GENERATOR, sha)


def build_renders(
    sub: SubFormat, tok: Any, stops: list[int], sha: str
) -> tuple[list[dict[str, Any]], dict[str, Rendered], list[str]]:
    renders = {c.name: render_case(tok, stops, c, sub.tools) for c in sub.cases}
    models = [sub.reference[0]]
    for repo, rev in sub.siblings:
        stok = AutoTokenizer.from_pretrained(repo, revision=rev)
        sstops = stop_ids(repo, rev)
        sib = {c.name: render_case(stok, sstops, c, sub.tools) for c in sub.cases}
        if all(
            renders[c.name].raw_output == sib[c.name].raw_output
            and renders[c.name].output_token_ids == sib[c.name].output_token_ids
            for c in sub.cases
        ):
            models.append(repo)
        else:
            print(f"  sibling {repo} renders differently; not listed")
    default_gp = default_gen_prompt(tok)
    records: list[dict[str, Any]] = []
    for case in sub.cases:
        r = renders[case.name]
        records.append(
            Record(
                id=f"{SLUG}/{sub.id_prefix}{case.name}",
                family=SLUG,
                models=models,
                provenance=provenance(sub, sha),
                tools=sub.tools,
                raw_output=r.raw_output,
                output_token_ids=r.output_token_ids,
                tokenizer=sub.tokenizer_pin,
                generation_prompt=r.generation_prompt if r.generation_prompt != default_gp else None,
                expected=expected_from(case.msg),
                tags=[*case.tags, sub.tag],
                notes=case.notes,
            ).to_dict()
        )
    return records, renders, models


def _truncated_recorder(sub: SubFormat, sha: str, out: list[dict[str, Any]]) -> Any:
    def rec(name: str, source: str, raw: str, ids: list[int], **kw: Any) -> None:
        out.append(
            Record(
                id=f"{SLUG}/{sub.id_prefix}{name}",
                family=SLUG,
                models=[sub.reference[0]],
                provenance=provenance(sub, sha),
                tools=sub.tools,
                raw_output=raw,
                output_token_ids=ids,
                tokenizer=sub.tokenizer_pin,
                **kw,
            ).to_dict()
        )
        src = f"{SLUG}/{sub.id_prefix}{source}"
        out[-1]["notes"] = f"Token prefix of {src}." + (f" {kw['notes']}" if "notes" in kw else "")

    return rec


TRUNCATED_ACCEPT = ["no_tool_calls", "content_passthrough", "exception"]


def build_truncated(tok: Any, renders: dict[str, Rendered], sha: str) -> list[dict[str, Any]]:
    """Token-prefix cuts of M2.x renders: what ``max_tokens`` produces."""
    out: list[dict[str, Any]] = []
    accept = TRUNCATED_ACCEPT
    rec = _truncated_recorder(M2, sha, out)

    single = M2.by_name("single-call")
    r = renders[single.name]
    raw, ids = cut_before(r, tok, "celsius")
    rec(
        "truncated-at-param-value-start",
        single.name,
        raw,
        ids,
        expected_error={
            "reason": (
                'max_tokens hit right after the opening <parameter name="unit"> tag, '
                "before any value text; the call never closes."
            ),
            "accept": accept,
        },
        tags=["truncated", "x-m2"],
    )
    raw, ids = cut_before(r, tok, "</invoke>")
    rec(
        "truncated-before-close-invoke",
        single.name,
        raw,
        ids,
        expected_error={
            "reason": "max_tokens hit after the last </parameter>; </invoke> is missing.",
            "accept": accept,
        },
        tags=["truncated", "x-m2"],
    )
    raw, ids = cut_before(r, tok, "</minimax:tool_call>")
    rec(
        "truncated-before-close-section",
        single.name,
        raw,
        ids,
        expected=expected_from(single.msg),
        tags=["truncated", "x-m2"],
        notes=(
            "max_tokens hit after </invoke>; only </minimax:tool_call> is missing. The call is closed, so a correct "
            "parser keeps it with content null (docs/formats/recovery.md, section 2)."
        ),
    )
    raw, ids = cut_before(r, tok, '<parameter name="unit">')
    rec(
        "truncated-between-params",
        single.name,
        raw,
        ids,
        expected_error={"reason": "max_tokens hit between two parameters of an open call.", "accept": accept},
        tags=["truncated", "x-m2"],
    )
    # Truncated inside the reasoning: well defined (everything is reasoning, no call).
    raw, ids = cut_before(r, tok, " in Celsius")
    rec(
        "truncated-in-reasoning",
        single.name,
        raw,
        ids,
        expected={"content": None, "reasoning_content": raw, "tool_calls": []},
        tags=["truncated", "reasoning-prefilled", "x-m2", "x-think-no-open-tag"],
        notes="The generation prompt opened <think>, so the unterminated text is reasoning, not content.",
    )
    return out


def build_m3_truncated(tok: Any, renders: dict[str, Rendered], sha: str) -> list[dict[str, Any]]:
    """Token-prefix cuts of M3 renders, judged by docs/formats/recovery.md section 2."""
    out: list[dict[str, Any]] = []
    accept = TRUNCATED_ACCEPT
    rec = _truncated_recorder(M3, sha, out)

    single = M3.by_name("single-call")
    r = renders[single.name]
    raw, ids = cut_before(r, tok, " in Celsius")
    if not raw.startswith("<mm:think>"):
        raise AssertionError(f"M3 render does not open <mm:think>: {raw!r}")
    rec(
        "truncated-in-reasoning",
        single.name,
        raw,
        ids,
        expected={"content": None, "reasoning_content": raw[len("<mm:think>") :], "tool_calls": []},
        tags=["truncated", "reasoning", "x-m3"],
        notes="Cut inside the open <mm:think> block: the unterminated text is reasoning, not content.",
    )
    raw, ids = cut_before(r, tok, f"{NS}<unit>")
    rec(
        "truncated-between-params",
        single.name,
        raw,
        ids,
        expected_error={"reason": "max_tokens hit between two parameters of an open call.", "accept": accept},
        tags=["truncated", "x-m3"],
    )
    raw, ids = cut_before(r, tok, "celsius")
    rec(
        "truncated-in-param-open-tag",
        single.name,
        raw,
        ids,
        expected_error={
            "reason": "max_tokens hit inside the opening tag of the second parameter (]<]minimax[>[<unit, before "
            "its '>', which the tokenizer merges with the value); the call never closes.",
            "accept": accept,
        },
        tags=["truncated", "x-m3"],
    )
    raw, ids = cut_before(r, tok, f"{NS}</invoke>")
    rec(
        "truncated-before-close-invoke",
        single.name,
        raw,
        ids,
        expected_error={
            "reason": "max_tokens hit after the last parameter element; ]<]minimax[>[</invoke> is missing.",
            "accept": accept,
        },
        tags=["truncated", "x-m3"],
    )
    raw, ids = cut_before(r, tok, f"{NS}</tool_call>")
    rec(
        "truncated-before-close-section",
        single.name,
        raw,
        ids,
        expected=expected_from(single.msg),
        tags=["truncated", "reasoning", "x-m3"],
        notes=(
            "max_tokens hit after ]<]minimax[>[</invoke>; only the section close is missing. The call is closed, so "
            "a correct parser keeps it with content null (docs/formats/recovery.md, section 2)."
        ),
    )
    par = M3.by_name("parallel-same-tool")
    raw, ids = cut_before(renders[par.name], tok, f"{NS}</invoke>", occurrence=2)
    first = expected_from(par.msg)["tool_calls"][0]
    rec(
        "truncated-second-parallel-call",
        par.name,
        raw,
        ids,
        expected={"content": None, "reasoning_content": par.msg["reasoning_content"], "tool_calls": [first]},
        tags=["truncated", "parallel-calls", "reasoning", "x-m3"],
        notes=(
            "max_tokens hit inside the second <invoke>, before its close. The first call is closed, so a correct "
            "parser keeps it, drops the unfinished second one, and returns content null "
            "(docs/formats/recovery.md, section 2)."
        ),
    )
    return out


MARKERS = [
    "<minimax:tool_call>",
    "</minimax:tool_call>",
    '<invoke name="',
    "</invoke>",
    '<parameter name="',
    "</parameter>",
    "<think>",
    "</think>",
    "<response>",
    "</response>",
    "]~!b[",
    "]~b]",
    "[e~[",
    NS,
    "<mm:think>",
    "</mm:think>",
]


def reference_model(sub: SubFormat, variant: str, tok: Any, stops: list[int]) -> dict[str, Any]:
    return {
        "repo": sub.reference[0],
        "revision": sub.reference[1],
        "variant": variant,
        "default_generation_prompt": default_gen_prompt(tok),
        "stop_tokens": [tok.convert_ids_to_tokens(i) for i in stops],
        "tokenizer_mode": "hf",
    }


def family_json(m2: tuple[Any, list[int]], m3: tuple[Any, list[int]]) -> dict[str, Any]:
    return {
        "slug": SLUG,
        "name": "MiniMax (M2, M2.1, M2.5, M2.7, M3, M3.1-Flash)",
        "spec_version": "0.2",
        "has_reasoning": True,
        "markers": MARKERS,
        "reference_models": [
            reference_model(M2, "m2-xml", *m2),
            reference_model(M3, "m3-xml", *m3),
        ],
        "format_notes": "docs/formats/minimax.md",
        "notes": "Two sub-formats selected by models[0] (tags x-m2 / x-m3). M2.x: one <minimax:tool_call> section "
        'holding one <invoke name="..."> per call, each with <parameter name="..."> elements; string values are '
        "raw, other values JSON; the generation prompt always opens <think>, so the completion starts inside the "
        "reasoning. M3: every structural tag is prefixed with the ]<]minimax[>[ token, each parameter is an element "
        "named after its key, nested objects and arrays are expanded into child and <item> elements, scalars are "
        "bare text; reasoning is <mm:think>...</mm:think> (or a bare </mm:think> without reasoning). Both need the "
        "tool schema to recover value types.",
    }


def load(repo: str, revision: str) -> tuple[Any, list[int], str]:
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    sha = sha256_file(hf_hub_download(repo, "chat_template.jinja", revision=revision))
    return tok, stop_ids(repo, revision), sha


def main() -> None:
    tok, stops, sha = load(*REFERENCE)
    m3_tok, m3_stops, m3_sha = load(*M3_REFERENCE)
    OUT.mkdir(parents=True, exist_ok=True)
    fam = family_json((tok, stops), (m3_tok, m3_stops))
    (OUT / "family.json").write_text(json.dumps(fam, ensure_ascii=False, indent=2) + "\n")
    recs, renders, _models = build_renders(M2, tok, stops, sha)
    write_jsonl(OUT / "render.jsonl", recs)
    write_jsonl(OUT / "truncated.jsonl", build_truncated(tok, renders, sha))
    m3_recs, m3_renders, _m3_models = build_renders(M3, m3_tok, m3_stops, m3_sha)
    write_jsonl(OUT / "m3-render.jsonl", m3_recs)
    write_jsonl(OUT / "m3-truncated.jsonl", build_m3_truncated(m3_tok, m3_renders, m3_sha))

    everything = [json.loads(line) for p in sorted(OUT.glob("*.jsonl")) for line in p.read_text().splitlines()]
    validate_records(everything)
    print(f"validated {len(everything)} minimax fixtures")


if __name__ == "__main__":
    main()

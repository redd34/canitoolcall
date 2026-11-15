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
"""Build the ``minimax`` fixture corpus (MiniMax M2, M2.1, M2.5, M2.7).

Provenance of every fixture written here is ``template_render``: an assistant
message rendered through the OFFICIAL chat template of the reference model
(``chat_template.jinja`` at a pinned revision, MiniMax-M2.1) with
``apply_chat_template(tokenize=True)``; see ``_shared.core.render``. Truncated
fixtures are a token prefix of such a render, which is what ``max_tokens``
produces. The M2, M2.5 and M2.7 repositories are listed in ``models`` only when
their own template and tokenizer render every fixture byte-identically.

Every M2.x template opens ``<think>\\n`` in the generation prompt, so the raw
completion starts inside the reasoning (no opening tag) and each fixture has
reasoning. The template has no switch that turns thinking off.

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

TOKENIZER_PIN = {"repo": REFERENCE[0], "revision": REFERENCE[1], "mode": "hf"}
TEMPLATE_URL = f"https://huggingface.co/{REFERENCE[0]}/blob/{REFERENCE[1]}/chat_template.jinja"


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


# --------------------------------------------------------------------------- build


def render_case(tok: Any, stops: list[int], case: Case) -> Rendered:
    context = case.context or [{"role": "user", "content": case.user}]
    return render(
        tok,
        context,
        case.msg,
        tools=TOOLS,
        stop_ids=stops,
        followups=tool_results(case.msg),
        require_stop=True,
    )


def by_name(name: str) -> Case:
    return next(c for c in CASES if c.name == name)


def default_gen_prompt(tok: Any) -> str:
    probe = [{"role": "user", "content": "x"}]
    full = tok.apply_chat_template(probe, add_generation_prompt=True, tokenize=False)
    bare = tok.apply_chat_template(probe, add_generation_prompt=False, tokenize=False)
    return str(full)[len(str(bare)) :]


def provenance(sha: str) -> Provenance:
    return Provenance("template_render", TEMPLATE_URL, REFERENCE[1], LICENSE, GENERATOR, sha)


def build_renders(tok: Any, stops: list[int], sha: str) -> tuple[list[dict[str, Any]], dict[str, Rendered], list[str]]:
    renders = {c.name: render_case(tok, stops, c) for c in CASES}
    models = [REFERENCE[0]]
    for repo, rev in SIBLINGS:
        stok = AutoTokenizer.from_pretrained(repo, revision=rev)
        sstops = stop_ids(repo, rev)
        if all(renders[c.name].raw_output == render_case(stok, sstops, c).raw_output for c in CASES) and all(
            renders[c.name].output_token_ids == render_case(stok, sstops, c).output_token_ids for c in CASES
        ):
            models.append(repo)
        else:
            print(f"  sibling {repo} renders differently; not listed")
    default_gp = default_gen_prompt(tok)
    records: list[dict[str, Any]] = []
    for case in CASES:
        r = renders[case.name]
        records.append(
            Record(
                id=f"{SLUG}/{case.name}",
                family=SLUG,
                models=models,
                provenance=provenance(sha),
                tools=TOOLS,
                raw_output=r.raw_output,
                output_token_ids=r.output_token_ids,
                tokenizer=TOKENIZER_PIN,
                generation_prompt=r.generation_prompt if r.generation_prompt != default_gp else None,
                expected=expected_from(case.msg),
                tags=[*case.tags, "x-m2"],
                notes=case.notes,
            ).to_dict()
        )
    return records, renders, models


def build_truncated(tok: Any, renders: dict[str, Rendered], sha: str) -> list[dict[str, Any]]:
    """Token-prefix cuts of renders: what ``max_tokens`` produces."""
    out: list[dict[str, Any]] = []
    accept = ["no_tool_calls", "content_passthrough", "exception"]

    def rec(name: str, source: str, raw: str, ids: list[int], **kw: Any) -> None:
        out.append(
            Record(
                id=f"{SLUG}/{name}",
                family=SLUG,
                models=[REFERENCE[0]],
                provenance=provenance(sha),
                tools=TOOLS,
                raw_output=raw,
                output_token_ids=ids,
                tokenizer=TOKENIZER_PIN,
                **kw,
            ).to_dict()
        )
        out[-1]["notes"] = f"Token prefix of {SLUG}/{source}." + (f" {kw['notes']}" if "notes" in kw else "")

    single = by_name("single-call")
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
        expected_error={
            "reason": "max_tokens hit after </invoke>; </minimax:tool_call> is missing.",
            "accept": accept,
        },
        tags=["truncated", "x-m2"],
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
]


def family_json(tok: Any, stops: list[int]) -> dict[str, Any]:
    return {
        "slug": SLUG,
        "name": "MiniMax (M2, M2.1, M2.5, M2.7)",
        "spec_version": "0.2",
        "has_reasoning": True,
        "markers": MARKERS,
        "reference_models": [
            {
                "repo": REFERENCE[0],
                "revision": REFERENCE[1],
                "variant": "m2-xml",
                "default_generation_prompt": default_gen_prompt(tok),
                "stop_tokens": [tok.convert_ids_to_tokens(i) for i in stops],
                "tokenizer_mode": "hf",
            }
        ],
        "format_notes": "docs/formats/minimax.md",
        "notes": 'XML-ish format: one <minimax:tool_call> section holding one <invoke name="..."> per call, each '
        'with <parameter name="..."> elements. String values are raw (no escaping); other values are JSON, so a '
        "parser needs the tool schema. The generation prompt always opens <think>, so the completion starts inside "
        "the reasoning. M2, M2.1, M2.5 and M2.7 render identical assistant turns (see models). MiniMax-M3 uses a "
        "different format and is not part of this family.",
    }


def main() -> None:
    tok = AutoTokenizer.from_pretrained(REFERENCE[0], revision=REFERENCE[1])
    stops = stop_ids(*REFERENCE)
    sha = sha256_file(hf_hub_download(REFERENCE[0], "chat_template.jinja", revision=REFERENCE[1]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "family.json").write_text(json.dumps(family_json(tok, stops), ensure_ascii=False, indent=2) + "\n")
    recs, renders, _models = build_renders(tok, stops, sha)
    write_jsonl(OUT / "render.jsonl", recs)
    write_jsonl(OUT / "truncated.jsonl", build_truncated(tok, renders, sha))

    everything = [json.loads(line) for p in sorted(OUT.glob("*.jsonl")) for line in p.read_text().splitlines()]
    validate_records(everything)
    print(f"validated {len(everything)} minimax fixtures")


if __name__ == "__main__":
    main()

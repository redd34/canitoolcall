# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "transformers==5.17.0",
#     "tokenizers==0.23.2",
#     "jinja2==3.1.6",
#     "huggingface-hub==1.33.0",
#     "jsonschema==4.25.1",
# ]
# ///
"""Build the ``seed-oss`` fixture corpus (ByteDance Seed-OSS-36B-Instruct).

Provenance of every fixture written here:

* ``template_render``: an assistant message rendered through the OFFICIAL chat
  template of ``ByteDance-Seed/Seed-OSS-36B-Instruct`` (``chat_template.jinja`` at a
  pinned revision) with ``apply_chat_template(tokenize=True)``; see
  ``_shared.core.render``. Truncated fixtures are a token prefix of such a render,
  which is what ``max_tokens`` produces.
* ``engine_test``: raw strings copied from llama.cpp's ``tests/test-chat.cpp``, see
  ``imported.py``.

Run from the repo root (deterministic; re-running leaves ``git diff`` clean)::

    uv run --script scripts/fixtures/seed-oss/build.py
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import imported  # noqa: E402
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
    encode_raw,
    expected_from,
    render,
    sha256_file,
    tool,
    validate_records,
    write_jsonl,
)
from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SLUG = "seed-oss"
OUT = REPO_ROOT / "fixtures" / SLUG
GENERATOR = f"scripts/fixtures/{SLUG}/build.py"

REPO = "ByteDance-Seed/Seed-OSS-36B-Instruct"
REVISION = "497f1dca95ebdec98e41d517b9f060ee753c902f"
LICENSE = "Apache-2.0"
TEMPLATE_URL = f"https://huggingface.co/{REPO}/blob/{REVISION}/chat_template.jinja"
TOKENIZER_PIN = {"repo": REPO, "revision": REVISION, "mode": "hf"}


@dataclass
class Loaded:
    tok: Any
    stop_ids: list[int]
    template_sha256: str

    def provenance(self) -> Provenance:
        return Provenance("template_render", TEMPLATE_URL, REVISION, LICENSE, GENERATOR, self.template_sha256)


def load() -> Loaded:
    tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
    gen = json.loads(Path(hf_hub_download(REPO, "generation_config.json", revision=REVISION)).read_text())
    stops = gen["eos_token_id"]
    stop_ids = [int(s) for s in (stops if isinstance(stops, list) else [stops])]
    sha = sha256_file(hf_hub_download(REPO, "chat_template.jinja", revision=REVISION))
    return Loaded(tok, stop_ids, sha)


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
    {"query": {"type": "string"}, "max_results": {"type": "integer"}},
    ["query"],
)
T_TIME = tool("get_time", "Get the current UTC time.")
T_WRITE = tool(
    "write_file",
    "Write a text file.",
    {"path": {"type": "string"}, "content": {"type": "string"}},
    ["path", "content"],
)
T_ALARM = tool(
    "set_alarm",
    "Set an alarm.",
    {
        "hour": {"type": "integer"},
        "minute": {"type": "integer"},
        "volume": {"type": "number"},
        "repeat": {"type": "boolean"},
    },
    ["hour", "minute"],
)
T_TIMER = tool(
    "set_timer",
    "Start a countdown timer.",
    {"seconds": {"type": "integer"}, "label": {"type": "string"}},
    ["seconds"],
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
    {
        "settings": {"type": "object"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "dry_run": {"type": "boolean"},
    },
    ["settings"],
)
T_CONVERT = tool(
    "convert_units",
    "Convert a value between units.",
    {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}},
    ["value", "from_unit", "to_unit"],
)
TOOLS: list[Tool] = [T_WEATHER, T_SEARCH, T_TIME, T_WRITE, T_ALARM, T_TIMER, T_ORDER, T_SETTINGS, T_CONVERT]

LONG_TEXT = "\n".join(f"{i:03d}: The quick brown fox jumps over the lazy dog." for i in range(60))
BUDGET_REASONING = (
    "The user wants the weather in Paris, so I need to call get_weather with city Paris.\n"
    "<seed:cot_budget_reflect>I have used 21 tokens, and there are 491 tokens remaining for use."
    "</seed:cot_budget_reflect>\n"
    "Celsius is the natural unit here."
)


# --------------------------------------------------------------------------- cases


@dataclass
class Case:
    name: str
    tags: list[str]
    msg: Message
    user: str = "Please help."
    context: list[Message] = field(default_factory=list)
    template_kwargs: dict[str, Any] = field(default_factory=dict)
    thinking: bool | None = None
    notes: str | None = None


def tool_results(msg: Message) -> list[Message]:
    return [{"role": "tool", "content": '{"ok": true}'} for _ in msg.get("tool_calls") or []]


PYREPR = (
    "Non-string values are rendered with Jinja's |string filter, i.e. Python str(): True, False, and dicts/lists "
    "with single quotes. The schema types say boolean/object/array, so a correct parser returns JSON values. The "
    "original vLLM seed_oss parser contributed with the model (vllm#23241, ByteDance) lowercases booleans and falls "
    "back to ast.literal_eval for objects and arrays: https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/"
    "entrypoints/openai/tool_parsers/seed_oss_tool_parser.py#L180-L206"
)

CASES: list[Case] = [
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
    ),
    Case(
        "call-order-vs-tool-order",
        ["parallel-calls", "reasoning", "x-call-index-vs-tool-index"],
        assistant(
            reasoning="Weather, then search, then weather again.",
            calls=[
                call("get_weather", {"city": "Oslo"}),
                call("search_web", {"query": "Oslo museums", "max_results": 3}),
                call("get_weather", {"city": "Bergen"}),
            ],
        ),
        notes="The same tool is called twice around another one; call indices must be 0,1,2 in emission order.",
    ),
    Case(
        "text-before-call",
        ["single-call", "text-before-call", "reasoning"],
        assistant(
            content="I'll look that up for you.",
            reasoning="Need live data, so call the tool.",
            calls=[call("get_weather", {"city": "Berlin"})],
        ),
        notes="The template writes the content between newlines before the first <seed:tool_call>; the correct "
        "content is the message text without those separators.",
    ),
    Case(
        "no-call",
        ["no-call", "reasoning"],
        assistant(content="Paris is the capital of France.", reasoning="A simple fact; no tool is needed."),
        user="What is the capital of France?",
        notes="The template puts a newline between </seed:think> and the answer; the correct content is the answer "
        "without it.",
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
                        "path": "docs/seed.md",
                        "content": "Calls are wrapped in <seed:tool_call> and </seed:tool_call>, start with "
                        "<function=NAME>, and thoughts sit in <seed:think> ... </seed:think>.",
                    },
                )
            ],
        ),
        notes="The string value contains <seed:tool_call>, </seed:tool_call>, <seed:think> and </seed:think> "
        "(tokenized as the marker tokens) and the text <function=NAME>. It contains none of the markers that can "
        "end a value (</parameter>, <parameter=, </function>), so the value is delimited unambiguously.",
    ),
    Case(
        "typed-numbers",
        ["single-call", "numeric-arguments", "x-typed-values", "reasoning"],
        assistant(
            reasoning="Set the alarm for 7:05 at 80% volume.",
            calls=[call("set_alarm", {"hour": 7, "minute": 5, "volume": 0.8})],
        ),
        notes="Integers and numbers are written as bare text (7, 5, 0.8); only the tool schema gives their types.",
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
        "number-looking-string",
        ["single-call", "x-number-looking-string", "reasoning"],
        assistant(
            reasoning="Look up the order id exactly as given.",
            calls=[call("lookup_order", {"order_id": "123_456", "reference": "0042"})],
        ),
        notes="String values are raw, so 123_456 and 0042 must stay strings (the schema says string).",
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
        "boolean-python-repr",
        ["single-call", "x-typed-values", "x-python-repr", "reasoning"],
        assistant(
            reasoning="A repeating alarm at 6:30.",
            calls=[call("set_alarm", {"hour": 6, "minute": 30, "repeat": True})],
        ),
        notes=PYREPR,
    ),
    Case(
        "nested-python-repr",
        ["single-call", "nested-json", "x-typed-values", "x-python-repr", "reasoning"],
        assistant(
            reasoning="Apply the dark theme, tag it, and do it for real.",
            calls=[
                call(
                    "update_settings",
                    {
                        "settings": {"theme": "dark", "notifications": {"email": True, "sms": False}},
                        "tags": ["ui", "night"],
                        "dry_run": False,
                    },
                )
            ],
        ),
        notes=PYREPR,
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
        "trailing-newline-value",
        ["single-call", "x-significant-whitespace", "reasoning"],
        assistant(
            reasoning="Write the two-line script with its final newline.",
            calls=[call("write_file", {"path": "run.sh", "content": "#!/bin/sh\necho hi\n"})],
        ),
        notes="Seed-OSS writes values with no wrapping newlines (<parameter=k>value</parameter>, as the template and "
        "its system-prompt example show), so a newline right before </parameter> is part of the value. llama.cpp's "
        "own Seed-OSS test expects it kept (tests/test-chat.cpp L3491-L3516 at a25c9865); stripping it, as the Qwen3 "
        "grammar does for Qwen's newline-wrapped values, changes the file.",
    ),
    Case(
        "leading-space-value",
        ["single-call", "x-significant-whitespace", "reasoning"],
        assistant(
            reasoning="Keep the indentation exactly.",
            calls=[call("write_file", {"path": "pad.txt", "content": "  two  spaces"})],
        ),
        notes="Values have no wrapping whitespace in Seed-OSS, so leading spaces inside <parameter=...> are data.",
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
        "multi-turn-second-call",
        ["single-call", "multi-turn", "reasoning"],
        assistant(
            reasoning="Now convert the temperature.",
            calls=[call("convert_units", {"value": 21.5, "from_unit": "celsius", "to_unit": "fahrenheit"})],
        ),
        context=[
            {"role": "user", "content": "Weather in Rome, in Fahrenheit please."},
            assistant(reasoning="Get the weather first.", calls=[call("get_weather", {"city": "Rome"})]),
            {"role": "tool", "content": '{"temp_c": 21.5}'},
        ],
        notes="The earlier turn's reasoning is replayed in the history inside <seed:think>...</seed:think>.",
    ),
    Case(
        "budget-reflect-in-reasoning",
        ["single-call", "reasoning", "x-thinking-budget"],
        assistant(reasoning=BUDGET_REASONING, calls=[call("get_weather", {"city": "Paris", "unit": "celsius"})]),
        user="What's the weather in Paris?",
        template_kwargs={"thinking_budget": 512},
        thinking=True,
        notes="Rendered with thinking_budget=512, which adds the budget system message to the prompt. With a budget "
        "the model writes <seed:cot_budget_reflect>...</seed:cot_budget_reflect> notes inside its reasoning (model "
        "card, 'Thinking Budget'); they are part of reasoning_content, as vLLM's own seed_oss test asserts "
        "(tests/parser/engine/test_seed_oss.py L207-L235 at v0.30.0). The reflect text follows the model card's "
        "example; it is not a recorded generation.",
    ),
]


# --------------------------------------------------------------------------- build


def render_case(ld: Loaded, case: Case) -> Rendered:
    context = case.context or [{"role": "user", "content": case.user}]
    return render(
        ld.tok,
        context,
        case.msg,
        tools=TOOLS,
        stop_ids=ld.stop_ids,
        followups=tool_results(case.msg),
        template_kwargs=case.template_kwargs,
    )


def default_gen_prompt(ld: Loaded) -> str:
    probe = [{"role": "user", "content": "x"}]
    full = ld.tok.apply_chat_template(probe, add_generation_prompt=True, tokenize=False)
    bare = ld.tok.apply_chat_template(probe, add_generation_prompt=False, tokenize=False)
    return str(full)[len(str(bare)) :]


def build_renders(ld: Loaded) -> tuple[list[dict[str, Any]], dict[str, Rendered]]:
    default_gp = default_gen_prompt(ld)
    records: list[dict[str, Any]] = []
    renders: dict[str, Rendered] = {}
    for case in CASES:
        r = render_case(ld, case)
        renders[case.name] = r
        records.append(
            Record(
                id=f"{SLUG}/{case.name}",
                family=SLUG,
                models=[REPO],
                provenance=ld.provenance(),
                tools=TOOLS,
                raw_output=r.raw_output,
                output_token_ids=r.output_token_ids,
                tokenizer=TOKENIZER_PIN,
                generation_prompt=r.generation_prompt if r.generation_prompt != default_gp else None,
                thinking=case.thinking,
                expected=expected_from(case.msg),
                tags=case.tags,
                notes=case.notes,
            ).to_dict()
        )
    return records, renders


def by_name(name: str) -> Case:
    return next(c for c in CASES if c.name == name)


ACCEPT_ALL = ["no_tool_calls", "content_passthrough", "exception"]


def build_truncated(ld: Loaded, renders: dict[str, Rendered]) -> list[dict[str, Any]]:
    """Token-prefix cuts of renders: what ``max_tokens`` produces."""
    out: list[dict[str, Any]] = []

    def rec(name: str, source: str, raw: str, ids: list[int], tags: list[str], note: str, **kw: Any) -> None:
        out.append(
            Record(
                id=f"{SLUG}/{name}",
                family=SLUG,
                models=[REPO],
                provenance=ld.provenance(),
                tools=TOOLS,
                raw_output=raw,
                output_token_ids=ids,
                tokenizer=TOKENIZER_PIN,
                tags=["truncated", *tags],
                notes=f"Token prefix of {SLUG}/{source}. {note}".strip(),
                **kw,
            ).to_dict()
        )

    single = renders["single-call"]
    reasoning = by_name("single-call").msg["reasoning_content"]

    raw, ids = cut_before(single, ld.tok, " in Celsius")
    assert raw.startswith("<seed:think>")
    rec(
        "truncated-in-reasoning",
        "single-call",
        raw,
        ids,
        ["reasoning"],
        "The model opened <seed:think> itself, so the unterminated text after it is reasoning, not content.",
        expected={"content": None, "reasoning_content": raw[len("<seed:think>") :], "tool_calls": []},
    )
    raw, ids = cut_before(renders["trailing-newline-value"], ld.tok, "echo hi")
    assert raw.endswith("<parameter=content>#!/bin/sh\n"), raw
    rec(
        "truncated-in-parameter-value",
        "trailing-newline-value",
        raw,
        ids,
        ["reasoning"],
        "A cut '#!/bin/sh\\n' must not be returned as the file content.",
        expected_error={
            "reason": "max_tokens hit inside the content value, before </parameter>; the call never closes.",
            "accept": ACCEPT_ALL,
        },
    )
    raw, ids = cut_before(single, ld.tok, "</function>")
    rec(
        "truncated-before-close-function",
        "single-call",
        raw,
        ids,
        ["reasoning"],
        "Every </parameter> is present, but the call is not closed.",
        expected_error={
            "reason": "max_tokens hit after the last </parameter>; </function> and </seed:tool_call> are missing.",
            "accept": ACCEPT_ALL,
        },
    )
    raw, ids = cut_before(single, ld.tok, "</seed:tool_call>")
    rec(
        "truncated-before-close-tool-call",
        "single-call",
        raw,
        ids,
        ["reasoning"],
        "As for qwen3-xml (llamacpp-q35-reasoning-then-call-cut-before-outer-close), the call's own close is the outer "
        "</seed:tool_call>, so a call with </function> but without it is unfinished.",
        expected_error={
            "reason": "max_tokens hit after </function>, before the outer close </seed:tool_call>.",
            "accept": ACCEPT_ALL,
        },
    )
    raw, ids = cut_before(single, ld.tok, "<function=")
    assert raw.endswith("<seed:tool_call>\n") or raw.endswith("<seed:tool_call>"), raw
    rec(
        "truncated-after-tool-call-open",
        "single-call",
        raw,
        ids,
        ["reasoning"],
        "",
        expected_error={
            "reason": "max_tokens hit right after <seed:tool_call>, before the function name.",
            "accept": ACCEPT_ALL,
        },
    )

    parallel = renders["parallel-same-tool"]
    pcase = by_name("parallel-same-tool")
    raw, ids = cut_before(parallel, ld.tok, "Tokyo")
    first = expected_from(pcase.msg)
    first["tool_calls"] = first["tool_calls"][:1]
    rec(
        "truncated-second-parallel-call",
        "parallel-same-tool",
        raw,
        ids,
        ["parallel-calls", "reasoning"],
        "The first call is closed by its own </seed:tool_call>, so it is complete and must be kept; the unfinished "
        "second call must not be returned, and its text is not content (docs/formats/recovery.md section 2).",
        expected=first,
    )
    assert reasoning and first["reasoning_content"]
    return out


MARKERS = [
    "<seed:tool_call>",
    "</seed:tool_call>",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
    "<seed:think>",
    "</seed:think>",
    "<seed:cot_budget_reflect>",
    "</seed:cot_budget_reflect>",
    "<seed:bos>",
    "<seed:eos>",
    "<seed:pad>",
]


def family_json(ld: Loaded) -> dict[str, Any]:
    return {
        "slug": SLUG,
        "name": "ByteDance Seed-OSS",
        "spec_version": "0.2",
        "has_reasoning": True,
        "markers": MARKERS,
        "reference_models": [
            {
                "repo": REPO,
                "revision": REVISION,
                "variant": "seed-oss",
                "default_generation_prompt": default_gen_prompt(ld),
                "stop_tokens": [ld.tok.convert_ids_to_tokens(i) for i in ld.stop_ids],
                "tokenizer_mode": "hf",
            }
        ],
        "format_notes": "docs/formats/seed-oss.md",
        "notes": "Qwen3-Coder-style XML wrapped in <seed:tool_call>...</seed:tool_call>, with <seed:think> reasoning "
        "that the model opens itself (the generation prompt pre-fills nothing unless thinking_budget=0). Values are "
        "written without wrapping newlines; non-string values are Python str() of the value, so a parser needs the "
        "tool schema. <seed:cot_budget_reflect> notes belong to the reasoning. Stop token from generation_config.json. "
        "Generator: scripts/fixtures/seed-oss/build.py.",
    }


def main() -> None:
    ld = load()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "family.json").write_text(json.dumps(family_json(ld), ensure_ascii=False, indent=2) + "\n")
    recs, renders = build_renders(ld)
    write_jsonl(OUT / "rendered.jsonl", recs)
    write_jsonl(OUT / "truncated.jsonl", build_truncated(ld, renders))
    imported_recs = [
        item.record(SLUG, REPO, TOKENIZER_PIN, encode_raw(ld.tok, item.raw)).to_dict() for item in imported.ITEMS
    ]
    write_jsonl(OUT / "imported.jsonl", imported_recs)

    everything = [json.loads(line) for p in sorted(OUT.glob("*.jsonl")) for line in p.read_text().splitlines()]
    validate_records(everything)
    print(f"validated {len(everything)} {SLUG} fixtures")


if __name__ == "__main__":
    main()

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
"""Build the ``nemotron`` fixture corpus (NVIDIA Nemotron 3 and 3.5).

Every fixture written here is a ``template_render``: an assistant message rendered
through the OFFICIAL ``chat_template.jinja`` of the reference model at a pinned
revision with ``apply_chat_template(tokenize=True)`` (``_shared.core.render``: the
generation-prompt ids must be a prefix of the full render, and the output is cut at
the first stop id). Truncated fixtures are token prefixes of such renders, which is
what ``max_tokens`` produces.

Two template variants (docs/formats/nemotron.md):

* ``v3``: Nemotron 3 Nano / Super. Reasoning renders as ``<think>\\nR\\n</think>\\n``.
* ``v35``: Nemotron 3.5 Lightning (and Nemotron 3 Ultra, whose template has the same
  history serialization). Reasoning renders as ``<think>\\nR</think>\\n``.

Each variant lists sibling repos; a sibling is added to ``models`` only if its own
template and tokenizer render byte-identical output ids for every case of the variant.

Stop ids: ``generation_config.json`` ``eos_token_id`` plus the tokenizer's
``eos_token`` (``<|im_end|>``). Nemotron-3-Nano-4B's generation config lists only
``</s>`` (2), while its tokenizer's eos token, and every other reference's generation
config, include ``<|im_end|>`` (11), the token that ends every assistant turn.

Only tokenizer, template and config files are downloaded, never weights. Run from
the repo root (deterministic; re-running leaves ``git diff`` clean)::

    uv run --script scripts/fixtures/nemotron/build.py
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

SLUG = "nemotron"
OUT = REPO_ROOT / "fixtures" / SLUG
GENERATOR = f"scripts/fixtures/{SLUG}/build.py"

NVIDIA_OML = "LicenseRef-nvidia-nemotron-open-model-license"
OPENMDW = "LicenseRef-OpenMDW-1.1"


# --------------------------------------------------------------------------- models


@dataclass(frozen=True)
class Model:
    key: str
    repo: str
    revision: str
    license: str
    variant: str
    tag: str
    """Fixture tag naming the reference template (tags allow no dots)."""
    siblings: tuple[tuple[str, str], ...] = ()
    """(repo, revision) whose rendered output is checked to be identical, then listed in ``models``."""


NANO = Model(
    "v3",
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
    NVIDIA_OML,
    "nemotron-3",
    "x-nemotron3",
    siblings=(
        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "bf77c3174f68ad409e1c2aa60daeb46e32d1c606"),
        ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "2dc98e2afe4face0e4ce40972a915c45368bd34a"),
    ),
)
LIGHTNING = Model(
    "v35",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    "a9904d24bcc1d289a1950fa9d2b978c47cf903b9",
    OPENMDW,
    "nemotron-3.5",
    "x-nemotron35",
    siblings=(("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", "77df655d5e9f8362164ed14dd8b48f8bce657498"),),
)
MODELS = (NANO, LIGHTNING)


@dataclass
class Loaded:
    model: Model
    tok: Any
    stop_ids: list[int]
    template_sha256: str
    siblings: list[tuple[str, Any, list[int]]]

    @property
    def tokenizer_pin(self) -> dict[str, str]:
        return {"repo": self.model.repo, "revision": self.model.revision, "mode": "hf"}

    def template_url(self) -> str:
        return f"https://huggingface.co/{self.model.repo}/blob/{self.model.revision}/chat_template.jinja"

    def provenance(self) -> Provenance:
        return Provenance(
            "template_render",
            self.template_url(),
            self.model.revision,
            self.model.license,
            GENERATOR,
            self.template_sha256,
        )


def stop_ids(repo: str, revision: str, tok: Any) -> list[int]:
    gen = json.loads(Path(hf_hub_download(repo, "generation_config.json", revision=revision)).read_text())
    eos = gen["eos_token_id"]
    ids = [int(s) for s in (eos if isinstance(eos, list) else [eos])]
    if tok.eos_token_id not in ids:
        ids.append(int(tok.eos_token_id))
    return ids


def load_tok(repo: str, revision: str) -> tuple[Any, list[int]]:
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    return tok, stop_ids(repo, revision, tok)


def load(model: Model) -> Loaded:
    tok, stops = load_tok(model.repo, model.revision)
    sha = sha256_file(hf_hub_download(model.repo, "chat_template.jinja", revision=model.revision))
    sibs = [(repo, *load_tok(repo, rev)) for repo, rev in model.siblings]
    return Loaded(model, tok, stops, sha, sibs)


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
T_CONVERT = tool(
    "convert_units",
    "Convert a value between units.",
    {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}},
    ["value", "from_unit", "to_unit"],
)
TOOLS: list[Tool] = [T_WEATHER, T_SEARCH, T_TIME, T_WRITE, T_TIMER, T_EVENT, T_CONVERT]


# --------------------------------------------------------------------------- cases


@dataclass
class Case:
    name: str
    tags: list[str]
    msg: Message
    variants: tuple[str, ...]
    thinking: bool | None = None
    """``None``: the template default (thinking on); ``False``: ``enable_thinking=False``."""
    context: list[Message] = field(default_factory=list)
    notes: str | None = None
    user: str = "Please help."


def tool_results(msg: Message) -> list[Message]:
    return [{"role": "tool", "content": '{"ok": true}'} for _ in msg.get("tool_calls") or []]


BOTH = ("v3", "v35")
NOTE_PREFILL = "The generation prompt pre-fills '<think>\\n', so raw_output starts inside the reasoning."
CASES: list[Case] = [
    Case(
        "single-call",
        ["single-call", "reasoning", "reasoning-prefilled"],
        assistant(
            reasoning="The user wants the current weather in Paris, in Celsius.",
            calls=[call("get_weather", {"city": "Paris", "unit": "celsius"})],
        ),
        BOTH,
        user="What's the weather in Paris?",
        notes=NOTE_PREFILL,
    ),
    Case(
        "parallel-calls",
        ["parallel-calls", "reasoning", "reasoning-prefilled", "unicode", "nested-json", "numeric-arguments"],
        assistant(
            reasoning="Check the weather, then search for cafés.",
            calls=[
                call("get_weather", {"city": "Zürich", "unit": "celsius"}),
                call(
                    "search_web",
                    {
                        "query": 'café "best"',
                        "filters": {"site": "example.com", "max_results": 5, "tags": ["coffee", "wifi"]},
                    },
                ),
            ],
        ),
        BOTH,
        notes="Objects and arrays render as JSON (tojson); strings render raw, quotes included.",
    ),
    Case(
        "parallel-same-tool",
        ["parallel-calls", "reasoning", "reasoning-prefilled"],
        assistant(
            reasoning="Three cities, so three independent weather lookups.",
            calls=[
                call("get_weather", {"city": "Paris"}),
                call("get_weather", {"city": "Tokyo"}),
                call("get_weather", {"city": "Lima", "unit": "fahrenheit"}),
            ],
        ),
        ("v3",),
    ),
    Case(
        "no-call",
        ["no-call", "reasoning", "reasoning-prefilled"],
        assistant(content="Paris is the capital of France.", reasoning="A simple fact; no tool is needed."),
        BOTH,
        user="What is the capital of France?",
    ),
    Case(
        "content-before-call",
        ["single-call", "text-before-call", "reasoning", "reasoning-prefilled"],
        assistant(
            content="I'll look that up for you.",
            reasoning="Need live data, so call the tool.",
            calls=[call("get_weather", {"city": "Berlin"})],
        ),
        BOTH,
        notes="The template writes the content, then '\\n', then <tool_call>; the correct content is the message "
        "text without that separator.",
    ),
    Case(
        "marker-in-arguments",
        ["single-call", "marker-in-arguments", "reasoning", "reasoning-prefilled"],
        assistant(
            reasoning="Write the format note to the docs file.",
            calls=[
                call(
                    "write_file",
                    {
                        "path": "docs/format.md",
                        "content": "Wrap each call in <tool_call> and </tool_call>, with <function=NAME> and "
                        "<parameter=KEY> inside; reasoning goes in <think>.",
                    },
                )
            ],
        ),
        BOTH,
        notes="The string value contains <tool_call>, </tool_call> and <think> (tokenized as the marker tokens) and "
        "the text markers <function= and <parameter=, none at a line start. It does not contain </parameter>, and a "
        "value always ends with '\\n</parameter>', so only one parse is consistent with the template.",
    ),
    Case(
        "empty-arguments",
        ["single-call", "empty-arguments", "reasoning", "reasoning-prefilled"],
        assistant(reasoning="Just fetch the time.", calls=[call("get_time", {})]),
        ("v3",),
    ),
    Case(
        "python-literals",
        [
            "single-call",
            "numeric-arguments",
            "x-python-literals",
            "x-schema-coercion",
            "reasoning",
            "reasoning-prefilled",
        ],
        assistant(
            reasoning="Create the standup with no reminder.",
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
        BOTH,
        notes="Scalars render with Jinja '| string', so false and null appear as the Python literals False and "
        "None, and 15 as bare text. Only the tool schema (boolean, integer|null, integer) says what they are; "
        "the correct parse is the JSON-typed value.",
    ),
    Case(
        "schema-coercion",
        ["single-call", "numeric-arguments", "x-schema-coercion", "reasoning", "reasoning-prefilled"],
        assistant(
            reasoning="Three second timer labelled 3.",
            calls=[call("set_timer", {"seconds": 3, "label": "3"})],
        ),
        ("v3",),
        notes="Both values render as the bare text 3; only the tool schema tells the integer from the string.",
    ),
    Case(
        "multiline-string",
        [
            "single-call",
            "string-escapes",
            "x-multiline-string",
            "x-whitespace-significant-string",
            "reasoning",
            "reasoning-prefilled",
        ],
        assistant(
            reasoning="Save the note verbatim.",
            calls=[
                call(
                    "write_file",
                    {
                        "path": "C:\\Users\\ana\\notes.txt",
                        "content": 'line 1\n\tline 2 with "quotes" and a backslash \\ end\n  ',
                    },
                )
            ],
        ),
        ("v3",),
        notes="String values are written raw (real newlines, tab, quotes, backslashes) between the wrapping "
        "newlines '<parameter=content>\\n' and '\\n</parameter>'. Exactly one leading and one trailing newline are "
        "markup; the trailing '\\n  ' belongs to the value.",
    ),
    Case(
        "unicode",
        ["single-call", "unicode", "reasoning", "reasoning-prefilled"],
        assistant(
            reasoning="用户想搜索东京的天气。🌦️",
            calls=[call("search_web", {"query": "東京の天気 ☀️ — «prévisions» für Zürich 🇨🇭"})],
        ),
        ("v3",),
    ),
    Case(
        "thinking-disabled",
        ["single-call"],
        assistant(calls=[call("get_weather", {"city": "Berlin", "unit": "celsius"})]),
        BOTH,
        thinking=False,
        notes="enable_thinking=false: the generation prompt pre-fills an empty '<think></think>', and the output "
        "starts with the '\\n' the template writes before <tool_call>.",
    ),
    Case(
        "thinking-disabled-no-call",
        ["no-call"],
        assistant(content="Hello! How can I help you today?"),
        ("v3",),
        thinking=False,
        user="Hi!",
        notes="enable_thinking=false and plain content: there is no reasoning, so a parser must not return the "
        "content as reasoning.",
    ),
    Case(
        "multi-turn-second-call",
        ["single-call", "multi-turn", "reasoning", "numeric-arguments", "reasoning-prefilled"],
        assistant(
            reasoning="Now convert the temperature.",
            calls=[call("convert_units", {"value": 21.5, "from_unit": "celsius", "to_unit": "fahrenheit"})],
        ),
        ("v3",),
        context=[
            {"role": "user", "content": "Weather in Rome, in Fahrenheit please."},
            assistant(reasoning="Get the weather first.", calls=[call("get_weather", {"city": "Rome"})]),
            {"role": "tool", "content": '{"temp_c": 21.5}'},
        ],
    ),
]


# --------------------------------------------------------------------------- build


def template_kwargs(case: Case) -> dict[str, Any]:
    return {} if case.thinking is None else {"enable_thinking": case.thinking}


def render_case(ld: Loaded, case: Case, tok: Any | None = None, stops: list[int] | None = None) -> Rendered:
    context = case.context or [{"role": "user", "content": case.user}]
    return render(
        tok or ld.tok,
        context,
        case.msg,
        tools=TOOLS,
        stop_ids=stops or ld.stop_ids,
        followups=tool_results(case.msg),
        template_kwargs=template_kwargs(case),
    )


def models_for(ld: Loaded, cases: list[Case]) -> list[str]:
    """The reference repo plus every sibling whose render (ids and text) is identical for all cases."""
    out = [ld.model.repo]
    for repo, stok, sstops in ld.siblings:
        same = True
        for c in cases:
            a, b = render_case(ld, c), render_case(ld, c, stok, sstops)
            same = same and (a.raw_output, a.output_token_ids) == (b.raw_output, b.output_token_ids)
        if same:
            out.append(repo)
        else:
            print(f"  sibling {repo} renders differently; not listed")
    return out


def default_gen_prompt(tok: Any, **kw: Any) -> str:
    probe = [{"role": "user", "content": "x"}]
    full = tok.apply_chat_template(probe, add_generation_prompt=True, tokenize=False, **kw)
    bare = tok.apply_chat_template(probe, add_generation_prompt=False, tokenize=False, **kw)
    return str(full)[len(str(bare)) :]


def build_renders(ld: Loaded) -> tuple[list[dict[str, Any]], dict[str, Rendered]]:
    cases = [c for c in CASES if ld.model.key in c.variants]
    models = models_for(ld, cases)
    default_gp = default_gen_prompt(ld.tok)
    records: list[dict[str, Any]] = []
    renders: dict[str, Rendered] = {}
    for case in cases:
        r = render_case(ld, case)
        renders[case.name] = r
        records.append(
            Record(
                id=f"{SLUG}/{ld.model.key}-{case.name}",
                family=SLUG,
                models=models,
                provenance=ld.provenance(),
                tools=TOOLS,
                raw_output=r.raw_output,
                output_token_ids=r.output_token_ids,
                tokenizer=ld.tokenizer_pin,
                generation_prompt=r.generation_prompt if r.generation_prompt != default_gp else None,
                thinking=case.thinking,
                expected=expected_from(case.msg),
                tags=[*case.tags, ld.model.tag],
                notes=case.notes,
            ).to_dict()
        )
    return records, renders


def by_name(name: str) -> Case:
    return next(c for c in CASES if c.name == name)


ACCEPT_ALL = ["no_tool_calls", "content_passthrough", "exception"]


def build_truncated(lds: dict[str, Loaded], renders: dict[str, dict[str, Rendered]]) -> list[dict[str, Any]]:
    """Token-prefix cuts of renders: what ``max_tokens`` produces (docs/formats/recovery.md, section 2)."""
    out: list[dict[str, Any]] = []

    def rec(ld: Loaded, name: str, source: str, marker: str, occurrence: int = 1, **kw: Any) -> str:
        case = by_name(source)
        r = renders[ld.model.key][source]
        raw, ids = cut_before(r, ld.tok, marker, occurrence)
        expected = kw.pop("expected", None)
        out.append(
            Record(
                id=f"{SLUG}/{ld.model.key}-{name}",
                family=SLUG,
                models=[ld.model.repo],
                provenance=ld.provenance(),
                tools=TOOLS,
                raw_output=raw,
                output_token_ids=ids,
                tokenizer=ld.tokenizer_pin,
                generation_prompt=r.generation_prompt if case.thinking is False else None,
                thinking=case.thinking,
                expected=expected(raw) if callable(expected) else expected,
                notes=f"Token prefix of {SLUG}/{ld.model.key}-{source}, cut before {marker!r}"
                + (f" (occurrence {occurrence})" if occurrence > 1 else "")
                + ". "
                + kw.pop("note", ""),
                **kw,
            ).to_dict()
        )
        return raw

    nano, light = lds["v3"], lds["v35"]
    rec(
        nano,
        "truncated-in-reasoning",
        "single-call",
        " in Celsius",
        expected=lambda raw: {"content": None, "reasoning_content": raw, "tool_calls": []},
        tags=["truncated", "reasoning", "reasoning-prefilled", "x-nemotron3"],
        note="The generation prompt opened <think>, so the unterminated text is reasoning, not content.",
    )
    rec(
        nano,
        "truncated-in-parameter-value",
        "single-call",
        "\n</parameter>\n</function>",
        expected_error={
            "reason": "max_tokens hit right after the last parameter value; </parameter>, </function> and "
            "</tool_call> are missing.",
            "accept": ACCEPT_ALL,
        },
        tags=["truncated", "reasoning", "reasoning-prefilled", "x-nemotron3"],
    )
    rec(
        nano,
        "truncated-before-close-function",
        "thinking-disabled",
        "</function>",
        expected_error={
            "reason": "max_tokens hit after the last </parameter>; </function> and </tool_call> are missing.",
            "accept": ACCEPT_ALL,
        },
        tags=["truncated", "x-nemotron3"],
        note="A parser must not emit a call that never closed.",
    )
    rec(
        nano,
        "truncated-before-close-tool-call",
        "thinking-disabled",
        "</tool_call>",
        expected_error={
            "reason": "max_tokens hit after </function>; the closing </tool_call> is missing.",
            "accept": ACCEPT_ALL,
        },
        tags=["truncated", "x-nemotron3"],
        note="All arguments and </function> are present, but the call is still unfinished.",
    )
    reasoning = by_name("parallel-same-tool").msg["reasoning_content"]
    rec(
        nano,
        "truncated-second-parallel-call",
        "parallel-same-tool",
        "Tokyo",
        expected={
            "content": None,
            "reasoning_content": reasoning,
            "tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}],
        },
        tags=["truncated", "parallel-calls", "reasoning", "reasoning-prefilled", "x-nemotron3"],
        note="The first call closed with </tool_call> before the cut, so it must be kept; the unfinished second "
        "call must not be returned, and its text is not content (recovery.md, section 2).",
    )
    rec(
        light,
        "truncated-in-parameter-value",
        "single-call",
        "\n</parameter>\n</function>",
        expected_error={
            "reason": "max_tokens hit right after the last parameter value; </parameter>, </function> and "
            "</tool_call> are missing.",
            "accept": ACCEPT_ALL,
        },
        tags=["truncated", "reasoning", "reasoning-prefilled", "x-nemotron35"],
    )
    return out


MARKERS = [
    "<tool_call>",
    "</tool_call>",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
    "<think>",
    "</think>",
    "<tool_response>",
    "</tool_response>",
    "<|im_start|>",
    "<|im_end|>",
    "</s>",
]


def family_json(lds: dict[str, Loaded]) -> dict[str, Any]:
    refs = []
    for ld in lds.values():
        toks = {ld.model.repo: (ld.tok, ld.stop_ids)} | {repo: (tok, stops) for repo, tok, stops in ld.siblings}
        for repo, revision in ((ld.model.repo, ld.model.revision), *ld.model.siblings):
            tok, stops = toks[repo]
            refs.append(
                {
                    "repo": repo,
                    "revision": revision,
                    "variant": ld.model.variant,
                    "default_generation_prompt": default_gen_prompt(tok),
                    "stop_tokens": [tok.convert_ids_to_tokens(i) for i in stops],
                    "tokenizer_mode": "hf",
                }
            )
    return {
        "slug": SLUG,
        "name": "NVIDIA Nemotron 3 and 3.5",
        "spec_version": "0.2",
        "has_reasoning": True,
        "markers": MARKERS,
        "reference_models": refs,
        "format_notes": "docs/formats/nemotron.md",
        "notes": "Tool calls use the Qwen3-Coder XML format (<tool_call>\\n<function=NAME>\\n<parameter=K>\\nV\\n"
        "</parameter>...); string values are raw and scalars render with Jinja '| string' (True/None), so a parser "
        "needs the tool schema. Thinking is on by default: the generation prompt pre-fills '<think>\\n'; "
        "enable_thinking=false pre-fills '<think></think>'. Variants differ only in how reasoning closes: "
        "nemotron-3 '\\n</think>\\n', nemotron-3.5 '</think>\\n'. Stop tokens: generation_config.json eos_token_id "
        "plus the tokenizer's eos_token <|im_end|> (Nano-4B's generation config lists only </s>). Tags "
        "x-nemotron3/x-nemotron35 name the reference template. Generator: scripts/fixtures/nemotron/build.py.",
    }


def main() -> None:
    lds = {m.key: load(m) for m in MODELS}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "family.json").write_text(json.dumps(family_json(lds), ensure_ascii=False, indent=2) + "\n")
    renders: dict[str, dict[str, Rendered]] = {}
    for key, ld in lds.items():
        recs, renders[key] = build_renders(ld)
        write_jsonl(OUT / f"{key}-render.jsonl", recs)
    write_jsonl(OUT / "truncated.jsonl", build_truncated(lds, renders))

    everything = [json.loads(line) for p in sorted(OUT.glob("*.jsonl")) for line in p.read_text().splitlines()]
    validate_records(everything)
    print(f"validated {len(everything)} {SLUG} fixtures")


if __name__ == "__main__":
    main()

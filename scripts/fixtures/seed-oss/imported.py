"""Seed-OSS fixtures copied from llama.cpp's test suite.

Each item quotes the raw model output verbatim from its source (line-anchored at the
pinned commit) and states the parse a correct parser must produce. ``build.py`` adds
token ids by encoding the raw text with the reference tokenizer (``<seed:think>`` and
``<seed:tool_call>`` written literally map to their added-token ids, as the model emits
them) and checks that the encoding round-trips.

Source: llama.cpp (MIT) ``tests/test-chat.cpp`` at a25c9865fe03c954c93fd755b5d79ae86ba99750,
the ``ByteDance-Seed-OSS`` block (L3433-L3538), which tests the parser llama.cpp derives
from ``models/templates/ByteDance-Seed-OSS.jinja`` (byte-identical to the Hugging Face
template at the fixtures' pinned revision).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from _shared.core import Provenance, Record, Tool, tool

LLAMACPP_SHA = "a25c9865fe03c954c93fd755b5d79ae86ba99750"
LLAMACPP_ATTR = "Copyright (c) 2023-2026 The ggml authors (MIT)."
TEST = "tests/test-chat.cpp"


def llamacpp(lines: str) -> Provenance:
    return Provenance(
        "engine_test",
        f"https://github.com/ggml-org/llama.cpp/blob/{LLAMACPP_SHA}/{TEST}#{lines}",
        LLAMACPP_SHA,
        "MIT",
        "scripts/fixtures/seed-oss/imported.py",
        attribution=LLAMACPP_ATTR,
    )


@dataclass
class Item:
    name: str
    raw: str
    tools: list[Tool]
    provenance: Provenance
    tags: list[str]
    expected: dict[str, Any]
    notes: str | None = None

    def record(self, slug: str, repo: str, tokenizer: dict[str, str], ids: list[int]) -> Record:
        return Record(
            id=f"{slug}/{self.name}",
            family=slug,
            models=[repo],
            provenance=self.provenance,
            tools=self.tools,
            raw_output=self.raw,
            output_token_ids=ids,
            tokenizer=tokenizer,
            expected=self.expected,
            tags=self.tags,
            notes=self.notes,
        )


def calls(*pairs: tuple[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"name": n, "arguments": a} for n, a in pairs]


# Tools exactly as declared in tests/test-chat.cpp (L425-L456, L541-L577).
SPECIAL = tool("special_function", "I'm special", {"arg1": {"type": "integer", "description": "The arg."}}, ["arg1"])
SPECIAL_OPT = tool(
    "special_function_with_opt",
    "I'm special but have optional stuff",
    {
        "arg1": {"type": "integer", "description": "The arg."},
        "arg2": {"type": "integer", "description": "The optional arg."},
    },
    ["arg1"],
)
TODO_LIST = tool(
    "todo_list",
    "Create or update the todo list",
    {"todos": {"type": "array", "description": "List of TODO list items"}},
    ["todos"],
)
EDIT = tool(
    "edit",
    "Edit file",
    {
        "filename": {"type": "string", "description": "Path of file to edit"},
        "oldString": {"type": "string", "description": "String to replace"},
        "newString": {"type": "string", "description": "New (replacement) value"},
    },
    ["filename", "oldString", "newString"],
)

ITEMS: list[Item] = [
    Item(
        "llamacpp-reasoning-then-content",
        "<seed:think>I'm thinking about the answer</seed:think>\nHello, world!",
        [],
        llamacpp("L3439-L3442"),
        ["no-call", "reasoning"],
        {"content": "Hello, world!", "reasoning_content": "I'm thinking about the answer", "tool_calls": []},
        notes="llama.cpp expects the newline after </seed:think> to be dropped from content. No tools are offered.",
    ),
    Item(
        "llamacpp-parallel-two-functions",
        "<seed:tool_call>\n<function=special_function>\n<parameter=arg1>1</parameter>\n</function>\n"
        "</seed:tool_call>\n<seed:tool_call>\n<function=special_function_with_opt>\n<parameter=arg1>1</parameter>\n"
        "<parameter=arg2>2</parameter>\n</function>\n</seed:tool_call>",
        [SPECIAL, SPECIAL_OPT],
        llamacpp("L3454-L3474"),
        ["parallel-calls", "numeric-arguments"],
        {
            "content": None,
            "reasoning_content": None,
            "tool_calls": calls(
                ("special_function", {"arg1": 1}), ("special_function_with_opt", {"arg1": 1, "arg2": 2})
            ),
        },
        notes="No reasoning block: the output starts with the call. The schema types arg1/arg2 as integers.",
    ),
    Item(
        "llamacpp-json-array-value",
        '<seed:tool_call>\n<function=todo_list>\n<parameter=todos>[{"item": "Check stuff", "selected": false}, '
        '{"item": "Prepare stuff", "selected": true}]</parameter>\n</function>\n</seed:tool_call>',
        [TODO_LIST],
        llamacpp("L3476-L3488"),
        ["single-call", "nested-json"],
        {
            "content": None,
            "reasoning_content": None,
            "tool_calls": calls(
                (
                    "todo_list",
                    {
                        "todos": [
                            {"item": "Check stuff", "selected": False},
                            {"item": "Prepare stuff", "selected": True},
                        ]
                    },
                )
            ),
        },
        notes="An array parameter written as JSON (lower-case true/false), unlike the template's Python str() form "
        "(see seed-oss/nested-python-repr). The schema says array, so the value is parsed.",
    ),
    Item(
        "llamacpp-quotes-and-trailing-newlines",
        "<seed:tool_call>\n<function=edit>\n<parameter=filename>foo.cpp</parameter>\n"
        '<parameter=oldString>def foo(arg = "14"):\n    return arg + "bar"\n</parameter>\n'
        '<parameter=newString>def foo(arg = "15"):\n    pass\n</parameter>\n'
        "</function>\n</seed:tool_call>",
        [EDIT],
        llamacpp("L3490-L3516"),
        ["single-call", "string-escapes", "x-significant-whitespace"],
        {
            "content": None,
            "reasoning_content": None,
            "tool_calls": calls(
                (
                    "edit",
                    {
                        "filename": "foo.cpp",
                        "oldString": 'def foo(arg = "14"):\n    return arg + "bar"\n',
                        "newString": 'def foo(arg = "15"):\n    pass\n',
                    },
                )
            ),
        },
        notes="Both code values end with a newline right before </parameter>; llama.cpp expects it kept, because "
        "Seed-OSS values have no wrapping newlines.",
    ),
]

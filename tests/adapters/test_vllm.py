"""Tests for the vLLM adapter.

* The fast tests need no engine: the parser map, ``supports``, the
  engine-free part of ``parser_config``, and stream accumulation.
* The ``@pytest.mark.engine("vllm")`` tests drive the real worker
  (``python -m canitoolcall.adapters.worker vllm``) in ``.venvs/vllm``. They
  are skipped when that venv is missing (see ``scripts/engines/vllm.sh``).

The engine tests replay the core sample, one corpus fixture per family (when
the corpus has one), and a tool-call turn per family rendered at test time
through the model's official chat template or encoder at a pinned revision.
Only families where the spike found that a history render equals the
generated format are rendered (see docs/formats/ and spec/README.md,
"History rendering is not generation").
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from canitoolcall.adapters import PYTHON_ENV, engine_python
from canitoolcall.adapters.base import ReplayInput
from canitoolcall.adapters.vllm import (
    PARSERS,
    VllmAdapter,
    _StreamAccumulator,
    chat_template_kwargs_for,
    resolve_parsers,
)
from canitoolcall.chunking import DEFAULT_STRATEGIES
from canitoolcall.fixtures import (
    TokenizerPin,
    default_fixtures_dir,
    load_families,
    load_fixtures,
    read_jsonl,
    repo_root,
)

FAMILY_SLUGS = (
    "qwen3-hermes",
    "qwen3-xml",
    "gpt-oss",
    "deepseek",
    "kimi",
    "glm",
    "minimax",
    "llama",
    "mistral",
    "gemma4",
)

# --------------------------------------------------------------------------- fast (no engine)


@pytest.mark.parametrize(
    ("family", "model", "tool", "reasoning"),
    [
        ("qwen3-hermes", "Qwen/Qwen3-0.6B", "hermes", "qwen3"),
        ("qwen3-hermes", "Qwen/Qwen3-8B", "hermes", "qwen3"),
        ("qwen3-hermes", "Qwen/Qwen3-4B-Instruct-2507", "hermes", None),
        ("qwen3-hermes", "Qwen/Qwen3-4B-Thinking-2507", "hermes", "deepseek_r1"),
        ("qwen3-hermes", "Qwen/Qwen2.5-7B-Instruct", "hermes", None),
        ("qwen3-xml", "Qwen/Qwen3-Coder-30B-A3B-Instruct", "qwen3_coder", None),
        ("qwen3-xml", "Qwen/Qwen3.5-9B", "qwen3_coder", "qwen3"),
        ("gpt-oss", "openai/gpt-oss-20b", "openai", "openai_gptoss"),
        ("deepseek", "deepseek-ai/DeepSeek-R1-0528", "deepseek_v3", "deepseek_r1"),
        ("deepseek", "deepseek-ai/DeepSeek-V3.1", "deepseek_v31", "deepseek_v3"),
        ("deepseek", "deepseek-ai/DeepSeek-V3.2", "deepseek_v32", "deepseek_v3"),
        ("deepseek", "deepseek-ai/DeepSeek-V4-Flash", "deepseek_v4", "deepseek_v4"),
        ("deepseek", "deepseek-ai/DeepSeek-V4.1-Flash", "deepseek_v41", "deepseek_v41"),
        ("kimi", "moonshotai/Kimi-K2-Instruct-0905", "kimi_k2", None),
        ("kimi", "moonshotai/Kimi-K2.6", "kimi_k2", "kimi_k2"),
        ("kimi", "moonshotai/Kimi-K3", "kimi_k3", "kimi_k3"),
        ("glm", "zai-org/GLM-4.5-Air", "glm45", "glm45"),
        ("glm", "zai-org/GLM-4.6", "glm45", "glm45"),
        ("glm", "zai-org/GLM-4.7", "glm47", "glm47"),
        ("glm", "zai-org/GLM-5.3", "glm47", "glm47"),
        ("minimax", "MiniMaxAI/MiniMax-M2.7", "minimax_m2", "minimax_m2"),
        ("minimax", "MiniMaxAI/MiniMax-M2", "minimax_m2", "minimax_m2"),
        ("llama", "meta-llama/Llama-3.1-8B-Instruct", "llama3_json", None),
        ("llama", "meta-llama/Llama-4-Scout-17B-16E-Instruct", "llama4_pythonic", None),
        ("mistral", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "mistral", None),
        ("mistral", "mistralai/Magistral-Small-2509", "mistral", "mistral"),
        ("mistral", "mistralai/Mistral-Medium-3.5-128B", "mistral", "mistral"),
        ("mistral", "mistralai/Ministral-3-14B-Reasoning-2512", "mistral", "mistral"),
        ("mistral", "mistralai/Ministral-3-14B-Instruct-2512", "mistral", None),
        ("mistral", "mistralai/Magistral-Small-2506", "mistral", "mistral"),
        ("mistral", "mistralai/Devstral-Small-2507", "mistral", None),
        ("gemma4", "google/gemma-4-31B-it", "gemma4", "gemma4"),
    ],
)
def test_parser_map(family: str, model: str, tool: str, reasoning: str | None) -> None:
    rule = resolve_parsers(family, model)
    assert rule is not None and rule.choice is not None
    assert (rule.choice.tool_parser, rule.choice.reasoning_parser) == (tool, reasoning)
    assert VllmAdapter().supports(family, model)


def test_parser_map_covers_known_families_only() -> None:
    assert set(PARSERS) <= set(FAMILY_SLUGS)


@pytest.mark.parametrize(
    ("family", "model"),
    [
        ("no-such-family", "Qwen/Qwen3-0.6B"),
        ("llama", "meta-llama/Llama-2-7b-chat-hf"),
        ("gemma4", "google/gemma-3-27b-it"),
        ("minimax", "MiniMaxAI/MiniMax-M3"),
    ],
)
def test_unsupported_is_honest(family: str, model: str) -> None:
    sup = VllmAdapter().supports(family, model)
    assert not sup
    assert sup.reason and "vLLM 0.30.0" in sup.reason


def test_module_does_not_import_vllm() -> None:
    code = "import sys, canitoolcall.adapters.vllm as m; m.VllmAdapter(); assert 'vllm' not in sys.modules"
    root = repo_root()
    assert root is not None
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    subprocess.run([sys.executable, "-c", code], check=True, env=env, timeout=60)


def test_chat_template_kwargs() -> None:
    assert chat_template_kwargs_for(None) == {}
    assert chat_template_kwargs_for(False) == {"enable_thinking": False, "thinking": False}


def test_static_parser_config() -> None:
    raw = ReplayInput(
        fixture_id="deepseek/x",
        family="deepseek",
        model="deepseek-ai/DeepSeek-V3.1",
        text="",
        token_ids=(1, 2),
        tokenizer=TokenizerPin("deepseek-ai/DeepSeek-V3.1", "c0781d039fb7a1ba2abc4add0bdc293e92d2b8db"),
        thinking=True,
    )
    cfg = VllmAdapter().static_parser_config(raw)
    assert cfg["tool_parser"] == "deepseek_v31" and cfg["reasoning_parser"] == "deepseek_v3"
    assert cfg["tokenizer"] == {
        "repo": "deepseek-ai/DeepSeek-V3.1",
        "revision": "c0781d039fb7a1ba2abc4add0bdc293e92d2b8db",
        "requested_mode": "auto",
    }
    assert cfg["chat_template_kwargs"] == {"enable_thinking": True, "thinking": True}
    assert cfg["units_source"] == "fixture.output_token_ids"
    assert cfg["stop_token_in_final_delta"] is False
    json.dumps(cfg)  # recorded in results: must be JSON-serializable


def test_parser_config_without_engine() -> None:
    import importlib.util

    if importlib.util.find_spec("vllm") is not None:
        pytest.skip("vllm is importable here; the engine tests cover the resolved config")
    raw = ReplayInput(
        fixture_id="mistral/x",
        family="mistral",
        model="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        text="",
        tokenizer_mode="mistral",
    )
    cfg = VllmAdapter().parser_config(raw)
    assert cfg["resolved"] is None
    assert cfg["tool_parser"] == "mistral" and cfg["reasoning_parser"] is None
    assert cfg["tokenizer"]["requested_mode"] == "mistral" and cfg["tokenizer"]["revision"] is None
    assert cfg["units_source"] == "engine tokenizer encode"


def _delta(content: str | None = None, reasoning: str | None = None, calls: list[Any] | None = None) -> Any:
    return SimpleNamespace(content=content, reasoning=reasoning, tool_calls=calls or [])


def _call(index: int, name: str | None = None, arguments: str | None = None) -> Any:
    return SimpleNamespace(index=index, function=SimpleNamespace(name=name, arguments=arguments))


def test_stream_accumulator_openai_style() -> None:
    acc = _StreamAccumulator()
    acc.add(None)
    acc.add(_delta(reasoning="thin"))
    acc.add(_delta(reasoning="king", content=""))
    acc.add(_delta(calls=[_call(0, "get_weather", '{"city": '), _call(1, "search")]))
    acc.add(_delta(calls=[_call(0, None, '"Paris"}'), _call(1, None, "{}")]))
    acc.add(_delta(content="done"))
    res = acc.result()
    assert res.reasoning_content == "thinking" and res.content == "done"
    assert [(c.name, c.arguments_raw) for c in res.tool_calls] == [
        ("get_weather", '{"city": "Paris"}'),
        ("search", "{}"),
    ]
    # A repeated name is concatenated, as openai-python's accumulate_delta does.
    acc.add(_delta(calls=[_call(1, "search")]))
    assert acc.result(exception="X: y").tool_calls[1].name == "searchsearch"
    assert acc.result(exception="X: y").exception == "X: y"


def test_empty_stream_accumulator() -> None:
    res = _StreamAccumulator().result()
    assert res.content is None and res.reasoning_content is None and res.tool_calls == ()


# --------------------------------------------------------------------------- engine (real worker)

STRATEGIES = [s.id for s in DEFAULT_STRATEGIES]


def _vllm_python() -> Path:
    root = repo_root()
    py = engine_python("vllm", root)
    if not os.environ.get(PYTHON_ENV.format(ENGINE="VLLM")) and py == Path(sys.executable):
        pytest.skip("vllm env not set up (run scripts/engines/vllm.sh)")
    return py


def _hf_cached(repo: str, revision: str | None) -> bool:
    """Whether ``repo@revision`` has tokenizer files in the local HF cache (config/tokenizer only)."""
    if revision is None:
        return False
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    snapshot = Path(os.environ.get("HF_HUB_CACHE", home / "hub")) / f"models--{repo.replace('/', '--')}"
    snapshot = snapshot / "snapshots" / revision
    return any((snapshot / name).is_file() for name in ("tokenizer_config.json", "tekken.json"))


def _engine_env(pins: Iterable[tuple[str, str | None]]) -> dict[str, str]:
    """Env for engine subprocesses: offline when every pinned tokenizer is cached (DESIGN.md rule 10)."""
    root = repo_root()
    assert root is not None
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    if all(_hf_cached(repo, rev) for repo, rev in pins):
        env["HF_HUB_OFFLINE"] = "1"
    return env


class Worker:
    """Minimal JSON-lines client for the adapter worker (protocol v1)."""

    def __init__(self, python: Path, env: dict[str, str]) -> None:
        self.proc = subprocess.Popen(
            [str(python), "-m", "canitoolcall.adapters.worker", "vllm"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )

    def call(self, msg: dict[str, Any]) -> dict[str, Any]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        assert line, "worker exited"
        reply: dict[str, Any] = json.loads(line)
        return reply

    def replay(self, record: dict[str, Any], family: dict[str, Any] | None) -> dict[str, Any]:
        reply = self.call({"op": "replay", "fixture": record, "family": family, "strategies": STRATEGIES})
        assert reply["ok"], reply.get("error")
        return reply

    def close(self) -> None:
        try:
            self.call({"op": "shutdown"})
        finally:
            self.proc.wait(timeout=60)


SAMPLE_PIN = ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca")


@pytest.fixture(scope="module")
def worker() -> Iterator[Worker]:
    pins = [SAMPLE_PIN, *((c["repo"], c["revision"]) for c in RENDER_CASES)]
    w = Worker(_vllm_python(), _engine_env(pins))
    yield w
    w.close()


def _calls(result: dict[str, Any]) -> list[tuple[str, Any]]:
    return [(c["name"], json.loads(c["arguments_raw"])) for c in result["tool_calls"]]


def _check_replay_shape(reply: dict[str, Any], family: str, model: str) -> None:
    """Harness-level invariants (the engine's parse itself may legitimately be wrong)."""
    assert reply["supported"], reply.get("reason")
    rule = resolve_parsers(family, model)
    assert rule is not None and rule.choice is not None
    cfg = reply["parser_config"]
    assert (cfg["tool_parser"], cfg["reasoning_parser"]) == (rule.choice.tool_parser, rule.choice.reasoning_parser)
    assert cfg["tokenizer"]["resolved_revision"]
    assert cfg["template"]["source"]
    assert set(reply["streams"]) | set(reply.get("skipped", {})) == set(STRATEGIES)
    assert "special" in reply["streams"], reply.get("skipped")
    for res in [reply["nonstream"], *reply["streams"].values()]:
        assert set(res) == {"content", "reasoning_content", "tool_calls", "exception"}


@pytest.mark.engine("vllm")
def test_hello(worker: Worker) -> None:
    hello = worker.call({"op": "hello"})
    assert hello["ok"] and hello["engine"] == "vllm"
    assert hello["version"] == VllmAdapter.pinned_version
    assert hello["details"]["transformers"]


@pytest.mark.engine("vllm")
def test_parser_names_registered_in_engine() -> None:
    """Every name in PARSERS is registered in the installed vLLM."""
    root = repo_root()
    assert root is not None
    code = (
        "import json\n"
        "from vllm.tool_parsers import ToolParserManager as T\n"
        "from vllm.reasoning import ReasoningParserManager as R\n"
        "print(json.dumps([sorted(T.list_registered()), sorted(R.list_registered())]))\n"
    )
    out = subprocess.run(
        [str(_vllm_python()), "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    ).stdout
    tools, reasonings = json.loads(out.strip().splitlines()[-1])
    for rules in PARSERS.values():
        for rule in rules:
            assert rule.choice is not None
            assert rule.choice.tool_parser in tools, rule
            assert rule.choice.reasoning_parser is None or rule.choice.reasoning_parser in reasonings, rule


@pytest.mark.engine("vllm")
def test_core_sample(worker: Worker, sample_fixtures_dir: Path) -> None:
    ((_, rec),) = list(read_jsonl(sample_fixtures_dir / "qwen3-hermes" / "sample.jsonl"))
    fam = load_families(sample_fixtures_dir)["qwen3-hermes"].to_dict()
    reply = worker.replay(rec, fam)
    _check_replay_shape(reply, "qwen3-hermes", rec["models"][0])
    expected = [(c["name"], c["arguments"]) for c in rec["expected"]["tool_calls"]]
    assert _calls(reply["nonstream"]) == expected
    for sid, res in reply["streams"].items():
        assert _calls(res) == expected, sid
    cfg = reply["parser_config"]
    assert cfg["tokenizer"]["revision"] == rec["tokenizer"]["revision"]
    assert cfg["generation_prompt_match"] is True
    assert cfg["template"]["sha256"] == rec["provenance"]["template_sha256"]


@pytest.mark.engine("vllm")
@pytest.mark.parametrize("family", FAMILY_SLUGS)
def test_corpus_fixture_per_family(family: str) -> None:
    """Replay the first supported corpus fixture of each family through the real worker."""
    root = default_fixtures_dir()
    if not (root / family / "family.json").is_file():
        pytest.skip(f"no corpus for {family} yet")
    fam = load_families(root)[family]
    adapter = VllmAdapter()
    fx = next(
        (f for f in load_fixtures([root / family]) if f.expected and adapter.supports(family, f.reference_model)),
        None,
    )
    if fx is None:
        pytest.skip(f"no supported corpus fixture with an expected parse for {family}")
    pin = (fx.tokenizer.repo, fx.tokenizer.revision) if fx.tokenizer else (fx.reference_model, None)
    worker = Worker(_vllm_python(), _engine_env([pin]))
    try:
        reply = worker.replay(fx.to_dict(), fam.to_dict())
    finally:
        worker.close()
    _check_replay_shape(reply, family, fx.reference_model)


# A tool-call turn rendered at test time through each model's official template
# (tokenize=True, prompt ids sliced off, cut at the first stop id), exactly as
# spec/README.md prescribes for template_render fixtures. Pinned revisions.
RENDER_CASES: list[dict[str, Any]] = [
    {
        "family": "qwen3-xml",
        "repo": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "revision": "b2cff646eb4bb1d68355c01b18ae02e7cf42d120",
        "stops": ["<|im_end|>"],
    },
    {
        "family": "glm",
        "repo": "zai-org/GLM-4.5-Air",
        "revision": "a24ceef6ce4f3536971efe9b778bdaa1bab18daa",
        "stops": ["<|user|>", "<|observation|>", "<|endoftext|>"],
        "reasoning": True,
    },
    {
        "family": "deepseek",
        "repo": "deepseek-ai/DeepSeek-V3.1",
        "revision": "c0781d039fb7a1ba2abc4add0bdc293e92d2b8db",
        "stops": ["<\uff5cend\u2581of\u2581sentence\uff5c>"],  # <｜end▁of▁sentence｜>  # noqa: RUF003
        "thinking": False,
    },
    {
        "family": "kimi",
        "repo": "moonshotai/Kimi-K2-Instruct",
        "revision": "fd1984e2b7a3350dbf7305fe73a4ede25c14de50",
        "stops": ["<|im_end|>"],
        "call_id": "functions.get_weather:0",
    },
    {
        "family": "gemma4",
        "repo": "google/gemma-4-E2B-it",
        "revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "stops": ["<turn|>", "<|tool_response>"],
        "reasoning": True,
        "thinking": True,
    },
    {
        # mistral_common v13 (Magistral) renders [TOOL_CALLS]name[ARGS]{...} with no
        # [CALL_ID], which is the generated format (v11 history renders add [CALL_ID]).
        "family": "mistral",
        "repo": "mistralai/Magistral-Small-2509",
        "revision": "a31cc96ab10cf19bc42c628fedf1e359e0853c49",
        "stops": ["</s>"],
        "mode": "mistral",
    },
]
RENDER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
RENDER_ARGS = {"city": "Zürich", "unit": "°C 🌤"}
RENDER_REASONING = "The user wants the weather; call the tool."

RENDER_SCRIPT = r"""
import json, sys
case, tools, args, reasoning_text = json.loads(sys.stdin.read())
if case.get("mode") == "mistral":
    from vllm.tokenizers import get_tokenizer
    tok = get_tokenizer(case["repo"], revision=case["revision"], tokenizer_mode="mistral")
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(case["repo"], revision=case["revision"], trust_remote_code=True)
kw = {} if case.get("thinking") is None else {"enable_thinking": case["thinking"], "thinking": case["thinking"]}
def ids(x):
    return list(x["input_ids"] if hasattr(x, "keys") else x)
user = {"role": "user", "content": "What is the weather in Zürich?"}
err = None
for a in (args, json.dumps(args, ensure_ascii=False)):  # some templates take dict args, some JSON text
    fn = {"name": "get_weather", "arguments": a}
    call = {"id": case.get("call_id", "a1b2c3d4e"), "type": "function", "function": fn}
    asst = {"role": "assistant", "content": "", "tool_calls": [call]}
    if case.get("reasoning"):
        asst["reasoning_content"] = reasoning_text
    tail = [{"role": "tool", "tool_call_id": call["id"], "name": "get_weather", "content": "sunny"}]
    try:
        prefix = ids(tok.apply_chat_template([user], tools=tools, add_generation_prompt=True, tokenize=True, **kw))
        full = ids(tok.apply_chat_template([user, asst, *tail], tools=tools, tokenize=True, **kw))
        break
    except Exception as e:
        err = e
else:
    raise SystemExit(f"render failed: {err}")
assert full[: len(prefix)] == prefix, "prompt is not a prefix of the rendered conversation"
out = full[len(prefix):]
stop_ids = {tok.convert_tokens_to_ids(s) for s in case["stops"]}
out = out[: next((i for i, t in enumerate(out) if t in stop_ids), len(out))]
print(json.dumps({"ids": out, "raw": tok.decode(out, skip_special_tokens=False)}, ensure_ascii=False))
"""


def _render(case: dict[str, Any]) -> dict[str, Any]:
    proc = subprocess.run(
        [str(_vllm_python()), "-c", RENDER_SCRIPT],
        input=json.dumps([case, RENDER_TOOLS, RENDER_ARGS, RENDER_REASONING]),
        capture_output=True,
        text=True,
        timeout=600,
        env=_engine_env([(case["repo"], case["revision"])]),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    rendered: dict[str, Any] = json.loads(proc.stdout.strip().splitlines()[-1])
    return rendered


@pytest.mark.engine("vllm")
@pytest.mark.parametrize("case", RENDER_CASES, ids=[c["family"] for c in RENDER_CASES])
def test_rendered_tool_call_per_family(worker: Worker, case: dict[str, Any]) -> None:
    rendered = _render(case)
    family, repo = case["family"], case["repo"]
    record: dict[str, Any] = {
        "id": f"{family}/test-render",
        "family": family,
        "models": [repo],
        "spec_version": "0.1",
        "provenance": {
            "kind": "template_render",
            "source_url": f"https://huggingface.co/{repo}/tree/{case['revision']}",
            "revision": case["revision"],
            "license": "NOASSERTION",
            "generator": "tests/adapters/test_vllm.py",
        },
        "tools": RENDER_TOOLS,
        "raw_output": rendered["raw"],
        "output_token_ids": rendered["ids"],
        "tokenizer": {"repo": repo, "revision": case["revision"], "mode": case.get("mode", "hf")},
        "expected": {
            "content": None,
            "reasoning_content": RENDER_REASONING if case.get("reasoning") else None,
            "tool_calls": [{"name": "get_weather", "arguments": RENDER_ARGS}],
        },
        "tags": [],
    }
    if case.get("thinking") is not None:
        record["thinking"] = case["thinking"]
    reply = worker.replay(record, None)
    _check_replay_shape(reply, family, repo)
    expected = [("get_weather", RENDER_ARGS)]
    # Verified by the engine spike: every family parses these renders correctly
    # when not streaming, and when streaming one token per delta.
    assert reply["nonstream"]["exception"] is None
    assert _calls(reply["nonstream"]) == expected, reply["nonstream"]
    assert _calls(reply["streams"]["token"]) == expected, reply["streams"]["token"]
    if case.get("reasoning"):
        assert (reply["nonstream"]["reasoning_content"] or "").strip() == RENDER_REASONING
    assert reply["parser_config"]["tokenizer"]["mode"] == case.get("mode", "hf")

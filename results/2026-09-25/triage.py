"""Classify every failing case of this results snapshot into findings; writes triage.jsonl.

Run from the repo root:  python3 results/2026-09-25/triage.py
Each failing case is assigned to the first finding whose selector matches, so the
order below matters: policy / not-a-bug categories first, then engine findings,
then a per-engine catch-all labelled ``untriaged_discrepancy``.
"""
import collections
import glob
import gzip
import json
import re
from pathlib import Path

SNAP = str(Path(__file__).resolve().parent)
ROOT = Path(__file__).resolve().parents[2]
FX = {}
for f in glob.glob(str(ROOT / "fixtures" / "*" / "*.jsonl")):
    for l in open(f):
        d = json.loads(l); FX[d["id"]] = d

runs = {}
for f in sorted(glob.glob(f"{SNAP}/*.json.gz")):
    with gzip.open(f, "rt") as fh:
        d = json.loads(fh.read())
    runs[d["engine"]["name"]] = d
    # Strategies this engine never produces (run.synthetic_strategies) never count.
    syn = set(d["run"].get("synthetic_strategies") or ())
    for c in d["cases"]:
        c["_synthetic"] = syn

def fails(c):
    out = collections.defaultdict(set)
    for r in c["checks"]:
        if r["status"] == "fail" and not r["strategy"].startswith("char") and r["strategy"] not in c["_synthetic"]:
            out[r["check"]].add(r["strategy"])
    return out

def strat_fail(c):
    s = set()
    for k, v in fails(c).items():
        s |= v
    return s

def model(fid): return FX[fid]["models"][0]
def tags(fid): return set(FX[fid]["tags"])
def is_err(fid): return "expected_error" in FX[fid]

def returned_calls(c):
    return any(r["check"] == "expected_error" and r["status"] == "fail" and "returned" in (r["detail"] or "") for r in c["checks"])

def only_one_special(c):
    s = strat_fail(c) - {"*"}
    return bool(s) and s <= {"one", "special"}

def nonstream_ok(c):
    return "nonstream" not in strat_fail(c)

def marker(fid):
    if "stop-at-open-marker" in fid:
        return False
    return bool({"marker-in-arguments", "marker-in-reasoning"} & tags(fid)) or "marker" in fid or "token-in-arguments" in fid

NOT_GENERATED_CALL_ID = {"mistral/llamacpp-v11-call-id", "mistral/llamacpp-v11-call-id-parallel"}
INTERIOR_STOP = {"gpt-oss/vllm-sequential-calls", "gpt-oss/vllm-call-then-final"}
CONTESTED = {"deepseek/sglang-v4-self-closing-invoke"}
NO_NEWLINE = {"qwen3-hermes/vllm-no-newlines-no-spaces", "qwen3-hermes/vllm-text-then-call-no-separator", "qwen3-hermes/vllm-two-calls-no-separator", "qwen3-hermes/vllm-content-and-call-single-chunk", "qwen3-hermes/sglang-text-before-call-with-space"}
HISTORY_TEXT_AFTER = {"gemma4/text-after-call", "gemma4/text-after-call-thinking", "glm/glm45-ollama-content-after-call"}

# (id, engine, classification, summary, upstream, direct_repro, notes, selector)
F = []
def finding(fid, engine, cls, summary, sel, upstream=None, direct=None, notes=None):
    F.append(dict(id=fid, engine=engine, classification=cls, summary=summary, upstream=upstream, direct_repro=direct,
                  notes=notes, sel=sel))

# ---- not bugs / policy (checked first) ----
for eng in runs:
    finding(f"{eng}-truncated-call-returned", eng, "truncation_policy",
            "Output cut by max_tokens inside a call comes back as a (partial) tool call. Fails by the spec's truncation policy (spec/README.md), not a mis-parse of complete output.",
            lambda c, fid: ("truncated" in tags(fid) or "truncated" in fid) and (not is_err(fid) or returned_calls(c)) and ("second-parallel" in fid or returned_calls(c)))
    if eng != "llamacpp":  # llama.cpp rejects both shapes the same way: see llamacpp-mistral-bos-rejects-calls
        finding(f"{eng}-call-id-format-unverified", eng, "format_unverified",
                "Mistral v11 [CALL_ID] outputs come from llama.cpp's tests. mistral-common's generation grammar has no [CALL_ID], but it only constrains guided decoding, and llama.cpp's tests assume the model emits it. Which form Mistral-Small-3.2 generates freely is unverified until a recorded generation settles it; not counted as a bug either way.",
                lambda c, fid: fid in NOT_GENERATED_CALL_ID)
    finding(f"{eng}-contested-format", eng, "contested_fixture",
            "DSML self-closing invoke: SGLang says the model emits it, DeepSeek's encoder never does and its parser rejects it.",
            lambda c, fid: fid in CONTESTED)
    finding(f"{eng}-history-render-text-after-call", eng, "uncertain_history_render",
            "Text after a call exists only in a history render (or an Ollama test); whether the model generates it is unverified, so dropping it is not counted as a bug.",
            lambda c, fid: fid in HISTORY_TEXT_AFTER)
for eng in ("llamacpp", "ollama"):
    finding(f"{eng}-interior-stop-token", eng, "not_applicable_stop_applied",
            "Fixture has an interior <|call|> that only reaches a parser when the stop token is ignored (vLLM #50690). llama-server stops at it, so only the first call is produced.",
            lambda c, fid: fid in INTERIOR_STOP)
finding("vllm-kimi-drops-text-after-section", "vllm", "intended_engine_behaviour",
        "kimi_k2 drops text after <|tool_calls_section_end|>; vLLM's own test asserts this. The fixture keeps it as content.",
        lambda c, fid: fid == "kimi/k2i-vllm-content-after-tool-section")
finding("vllm-gemma4-lenient-no-brace", "vllm", "intended_engine_behaviour",
        "gemma4 returns a call named 'bad_func no brace' for a call without an argument object; vLLM's own test asserts this leniency.",
        lambda c, fid: fid == "gemma4/vllm-malformed-no-brace")
for eng in runs:
    finding(f"{eng}-hermes-no-newline-variant", eng, "not_generated_by_model",
            "'<tool_call>{...}' without the newline Qwen3's official template always emits (a hermes-format engine-test variant). SGLang's qwen25 detector and llama.cpp's template-derived parser require the newline; not counted as a bug for Qwen3.",
            lambda c, fid: fid in NO_NEWLINE)

# ---- genuine engine bugs ----
finding("vllm-deepseek-v3-stream-split", "vllm", "engine_bug",
        "deepseek_v3/deepseek_v31 streaming loses the whole call (or its preceding content) when a delta carries more than one structural token, e.g. the whole output in one delta or special-token-boundary deltas; non-streaming is correct.",
        lambda c, fid: c["family"] == "deepseek" and re.search(r"DeepSeek-V3(\.1)?(-0324)?$|DeepSeek-V3$", model(fid)) and nonstream_ok(c) and not marker(fid) and not is_err(fid))
finding("vllm-llama3-json-one-delta", "vllm", "engine_bug_known_upstream",
        "llama3_json drops the call when the whole call arrives in one streaming delta.",
        lambda c, fid: c["family"] == "llama" and only_one_special(c), upstream="https://github.com/vllm-project/vllm/issues/48294")
finding("vllm-llama3-json-content-swallowed", "vllm", "engine_bug",
        "llama3_json streaming swallows content that starts with '{' but is not a call (a JSON answer, '{}'), and loses empty arguments; non-streaming returns it as content.",
        lambda c, fid: c["family"] == "llama" and nonstream_ok(c) and not only_one_special(c))
finding("vllm-gemma4-untyped-nested-strings", "vllm", "engine_bug",
        "gemma4 returns numbers/booleans/null nested in objects or arrays that the tool schema does not type as strings (25 -> \"25\", true -> \"true\"). Gemma marks strings with <|\"|>, so this loses the emitted type; vLLM's tests assert string output for schema-less values (design choice). SGLang and Ollama keep the types.",
        lambda c, fid: c["family"] == "gemma4" and fid in {"gemma4/nested-objects-and-arrays", "gemma4/edge-values-key-with-space", "gemma4/sglang-nested-array-with-spaces", "gemma4/sglang-nested-object", "gemma4/multi-turn-second-call", "gemma4/parallel-two-calls-thinking", "gemma4/bug-ollama-18390-key-with-spaces"},
        direct="repro/vllm_gemma4_nested_numbers.py")
finding("vllm-gemma4-text-around-call", "vllm", "engine_bug",
        "gemma4 drops content that follows a call (SGLang's engine test expects it as content).",
        lambda c, fid: fid == "gemma4/sglang-text-around-call")
finding("vllm-marker-text-in-arguments", "vllm", "engine_bug",
        "Format-marker text inside an argument string or reasoning (e.g. '</tool_call>', '<arg_key>', '[TOOL_CALLS]', '</function>') truncates the argument, splits it into phantom calls, or turns the whole call into content (hermes, glm45/glm47, kimi_k2, kimi_k3, gemma4, llama3_json, mistral, qwen3_coder, deepseek_v31).",
        lambda c, fid: marker(fid) or fid in {"kimi/k26-marker-terminator-in-string", "kimi/k3-control-marker-text-in-value", "mistral/llamacpp-ministral3-marker-in-reasoning"},
        direct="repro/vllm_hermes_marker_in_arguments.py")
finding("vllm-kimi-k3-truncated-reasoning-as-content", "vllm", "engine_bug_known_upstream",
        "kimi_k3 non-streaming returns unterminated reasoning (think channel opened by the prompt) as content; streaming returns it as reasoning.",
        lambda c, fid: fid in {"kimi/k3-truncated-in-reasoning", "kimi/k3-bug-truncated-reasoning-recorded"},
        upstream="https://github.com/vllm-project/vllm/issues/57353")
finding("vllm-kimi-k3-response-only", "vllm", "engine_bug_known_upstream",
        "kimi_k3 streaming puts a response-only completion (thinking disabled) into reasoning_content with the XTML markup leaked.",
        lambda c, fid: fid == "kimi/k3-bug-response-only-completion",
        upstream="https://github.com/vllm-project/vllm/issues/57688")
finding("vllm-llama4-pythonic-underscore", "vllm", "engine_bug_known_upstream",
        "llama4_pythonic non-streaming rejects an argument name starting with '_' and returns the call as content; streaming parses it.",
        lambda c, fid: fid == "llama/l4-bug-leading-underscore-identifier", upstream="https://github.com/vllm-project/vllm/issues/56840")
finding("vllm-llama3-text-before-python-tag", "vllm", "engine_bug",
        "llama3_json non-streaming drops text before <|python_tag|> (SGLang's test keeps it as content); streaming returns the text and the call JSON as content with no tool call.",
        lambda c, fid: fid == "llama/l3-sglang-text-before-python-tag")
finding("vllm-hermes-stream-incomplete-json", "vllm", "engine_bug",
        "hermes streaming returns a call for '<tool_call>' JSON missing its closing brace and </tool_call>; non-streaming (as vLLM's test asserts) returns no call.",
        lambda c, fid: fid == "qwen3-hermes/vllm-invalid-json-missing-brace")
finding("vllm-missing-close-drops-argument", "vllm", "engine_bug_known_upstream",
        "A missing closing tag before the call end drops the last argument (glm47 without the last </arg_value>; qwen3_coder without </parameter> gives {}).",
        lambda c, fid: fid in {"glm/glm47-missing-last-close-arg-value", "qwen3-xml/bug-missing-close-parameter-before-function"},
        upstream="https://github.com/vllm-project/vllm/issues/57699",
        notes="GLM part: vllm#57826's thread (2026-09-23) says it was already fixed on main by #45701 (merged 2026-06-16, before v0.30.0), but vLLM 0.30.0 still drops the value: vllm/parser/glm47_moe.py L56-66 gates _PARTIAL_ARG_RE behind 'if partial:', and Glm47MoeModelToolParser.extract_tool_calls returns {\"city\": \"Berlin\"} without unit. Settle the conflicting comment before reporting. The glm fixture's raw_output is derived (tag x-derived), not quoted from the issue.")
finding("vllm-qwen3-coder-text-after-call", "vllm", "engine_bug_known_upstream",
        "qwen3_coder drops text after a call.", lambda c, fid: fid == "qwen3-xml/bug-coder-text-after-call",
        upstream="https://github.com/vllm-project/vllm/issues/56263",
        notes="vllm#56263 reports this class (non-streaming drops post-tool-call text) for deepseekv3 and hermes; qwen3_coder is not named there, so add it as a comment. The fixture comes from the same bug in SGLang (sgl-project/sglang#40739).")
finding("vllm-harmony-header-leaks", "vllm", "engine_bug",
        "gpt-oss: output cut inside a call header leaks the header markup into content; a stray 'commentary to=assistant' header yields a call named 'assistant<|channel|>analysis'.",
        lambda c, fid: fid in {"gpt-oss/harmony-truncated-in-header", "gpt-oss/llamacpp-stray-commentary-header"})
# ---- vLLM malformed/truncated recovery contracts (issue #10) ----
finding("vllm-harmony-recovery-newline", "vllm", "untriaged_discrepancy",
        "gpt-oss: malformed-header recovery differs by an inter-message newline between non-streaming and streaming.",
        lambda c, fid: fid == "gpt-oss/vllm-malformed-headers",
        notes="The fixture accepts both recovery outputs; only stream_equals_nonstream fails. Non-streaming joins completed content messages with a newline; all eight tested stream chunkings recover the same messages without it. At ced6857, tests/parser/test_harmony.py L443-457 asserts the non-streaming join, while L552-577 checks individual deltas, including the last one, not the fully reconstructed streaming content. The intended cross-mode contract is therefore unresolved, not asserted by those tests. They include a terminal token removed by this fixture; the earlier pinned replay confirms the same difference without it. No conformance check is waived.")
finding("vllm-mistral-truncated-empty-call", "vllm", "truncation_policy",
        "mistral: a truncated call opener leaves an empty streaming call; non-streaming returns no call.",
        lambda c, fid: fid == "mistral/v13think-stop-at-open-marker",
        notes="Generation ends immediately after [TOOL_CALLS], with no call name or arguments. Non-streaming preserves the preceding text/reasoning with no call; all eight tested stream chunkings leave one empty-name, empty-argument call as the adapter accumulates parser events. The fixture is tagged truncated: this is a partial-call outcome on cut-off output, not a mis-parse of a complete call. The earlier replay generated no new model output and executed no tools.")
finding("vllm-harmony-malformed-header-rejection", "vllm", "untriaged_discrepancy",
        "gpt-oss: every parsing mode rejects the garbled commentary header with the same HarmonyError.",
        lambda c, fid: fid == "gpt-oss/bug-garbled-channel-commentary-question",
        notes="Non-streaming and all eight tested stream chunkings raise the same error. The failure is that expected_error permits only no_tool_calls/content_passthrough, not exceptions; it is not a stream/non-stream disagreement. Keep unresolved until maintainers decide the malformed-header recovery contract. Adding exception would change the fixture's contract. No HTTP response or production failure was measured.")
finding("vllm-deepseek-v3-malformed-call-fragments", "vllm", "untriaged_discrepancy",
        "deepseek_v3: malformed JSON produces invalid call fragments or no call, depending on the streaming split.",
        lambda c, fid: fid == "deepseek/vllm-v3-malformed-missing-brace",
        notes="With the JSON closing brace missing, non-streaming and some stream groupings emit get_weather with invalid arguments; one/special emit no call and other random groupings emit shorter fragments. This is the only within-stream split-dependent case among these nine; non-streaming also violates the fixture's no-call expectation. At ced6857, tests/tool_parsers/test_deepseekv3_tool_parser.py includes this exact input and marks non-streaming test_malformed_input as xfail. That test body in common_tests.py only requires extraction not to raise, rather than a particular recovered parse. Keep the recovery contract unresolved; do not equate it with a complete-output split-loss bug.")
finding("vllm-malformed-content-recovery-contract", "vllm", "untriaged_discrepancy",
        "Five malformed/truncated inputs return no calls but recover different text across streaming and non-streaming.",
        lambda c, fid: fid in {"deepseek/vllm-v3-malformed-missing-call-tokens", "mistral/vllm-v3-malformed-not-json", "qwen3-hermes/bug-sglang-30480-truncated-at-opener", "qwen3-hermes/sglang-malformed-json-in-tags", "qwen3-hermes/truncated-after-open-tag"},
        notes="Both modes return no calls and no exceptions, so each individual expected_error outcome is permitted. Only stream_equals_nonstream fails: fallback text is kept or dropped differently across modes, while all eight stream chunkings agree within each fixture. Maintainers must decide whether malformed-input text recovery must match across modes; keep all five unresolved, not automatically accepted. At ced6857, tests/parser/mistral/test_tool_calls.py L508-517 asserts the Mistral non-streaming fallback only, not the streaming contract.")
finding("vllm-stream-nonstream-other", "vllm", "untriaged_discrepancy",
        "Other streaming/non-streaming disagreements on malformed or unusual output (content whitespace, content kept vs dropped).",
        lambda c, fid: True)

# ---- genuine engine bugs: SGLang ----
finding("sglang-one-delta-args-lost", "sglang", "engine_bug",
        "Streaming loses arguments (''/'{}'), later parallel calls or preceding content only when a delta carries a whole call or several structural tokens (one delta, special-token-boundary deltas); per-token and small random deltas and non-streaming are correct (qwen25, deepseekv3, deepseekv31, glm45/glm47, llama3, mistral).",
        lambda c, fid: nonstream_ok(c) and only_one_special(c) and not marker(fid),
        direct="repro/sglang_qwen25_one_delta.py")
finding("sglang-stream-split-loss", "sglang", "engine_bug",
        "Streaming loses arguments, parallel calls or content for multi-token deltas beyond the single-delta case (random 1-8 token groups also fail); non-streaming is correct (qwen25, deepseekv3/v31, llama3, mistral, gemma4, glm45).",
        lambda c, fid: nonstream_ok(c) and not marker(fid) and c["family"] != "gpt-oss")
finding("sglang-gpt-oss-role-header-recipient", "sglang", "engine_bug",
        "ONE gap, many fixtures: the gpt-oss detector does not recognise a recipient in the role header ('<|start|>assistant to=functions.X<|channel|>commentary'), which the Harmony spec allows ('The recipient might be defined in the role or channel section of the header', docs/format.md) and which openai-harmony and the HF template render for history. Generations put the recipient after the channel, which SGLang parses. Every fixture here is tagged x-recipient-in-role; count it as one finding.",
        lambda c, fid: c["family"] == "gpt-oss" and "x-recipient-in-role" in tags(fid),
        notes="Most of these fixtures are history renders (openai-harmony 0.0.8, the HF template). Severity depends on whether any real client or model sends this form to the parser.")
finding("sglang-gpt-oss-header-forms", "sglang", "engine_bug",
        "gpt-oss detector only recognises '<|start|>assistant<|channel|>commentary to=...<|constrain|>json'. A call as the first message (no '<|start|>assistant' prefix), or no '<|constrain|>json', comes back as content with the markup leaked.",
        lambda c, fid: c["family"] == "gpt-oss")
finding("sglang-glm45-think-leak", "sglang", "engine_bug",
        "glm45 non-streaming reasoning leaks '\\n<think>' into reasoning_content.",
        lambda c, fid: c["family"] == "glm" and "GLM-4.5" in model(fid) and not marker(fid))
finding("sglang-gemma4-values", "sglang", "engine_bug",
        "gemma4: null becomes the string \"null\", keys with spaces / nested objects and arrays are mangled, text after a call is dropped in non-streaming, parallel calls lost in streaming.",
        lambda c, fid: c["family"] == "gemma4")
finding("sglang-dsml-string-values", "sglang", "engine_bug",
        "DeepSeek V3.2/V4 DSML: raw string values lose trailing newlines / are cut at '<' characters.",
        lambda c, fid: fid in {"deepseek/v32-unescaped-string-value", "deepseek/v4-unescaped-string-value"})
finding("sglang-marker-text-in-arguments", "sglang", "engine_bug",
        "Format-marker text inside argument strings or reasoning breaks the parse (qwen25, qwen3_coder, glm47, deepseekv31, kimi_k3, gemma4, llama32, mistral).",
        lambda c, fid: marker(fid) or fid in {"kimi/k26-marker-terminator-in-string", "kimi/k3-control-marker-text-in-value"})
finding("sglang-other", "sglang", "untriaged_discrepancy",
        "Other SGLang discrepancies (Mistral [THINK] reasoning not separated, Llama JSON content handling, Qwen3-Coder variants, Kimi content after section).",
        lambda c, fid: True)

finding("llamacpp-deepseek-v3-preserved-tokens", "llamacpp", "engine_bug",
        "With the official HF DeepSeek-V3/V3.1 chat template (as embedded in a converted GGUF), the autoparser derives malformed preserved_tokens (multi-token strings such as 'function<｜tool▁sep｜>'), so <｜tool▁sep｜> is never rendered and every call fails with 'does not match the expected peg-native format'.",
        lambda c, fid: c["family"] == "deepseek" and re.search(r"DeepSeek-V3(\.1)?(-0324)?$", model(fid)) is not None,
        notes="Only with the HF template. llama.cpp's own tests (test-chat.cpp L3909-3981) use its rewritten copy models/templates/deepseek-ai-DeepSeek-V3.1.jinja (sha256 d9f5f351..., 3211 bytes vs the HF template's 45690185..., 2779 bytes), with which the parser works (parser_config.template_alternatives records that outcome per fixture). Popular GGUFs (Hub API GET, 2026-09-25): bartowski/deepseek-ai_DeepSeek-V3.1-GGUF@a7ccff77 embeds the HF template (sha256 45690185...); unsloth/DeepSeek-V3.1-GGUF@feb73eff embeds its own modified template (eccaf95e..., 3096 bytes), not tested here. Re-check both before reporting upstream.")
finding("llamacpp-mistral-bos-rejects-calls", "llamacpp", "engine_bug_candidate",
        "CANDIDATE, unconfirmed: Mistral-Small-3.2 (v11) and Mistral-7B-v0.3 (v3) tool calls are rejected ('does not match the expected peg-native format') when the vocab's BOS is passed to common_chat_templates_init, as llama-server does; test-chat.cpp (model=nullptr) accepts them. Seen only by A/B in the harness, never against a live llama-server. For v11 it covers both shapes: the id-less mistral-common renders and llama.cpp's own [CALL_ID] test strings fail the same way, so it does not depend on which form the model generates (that is unverified, see *-call-id-format-unverified).",
        lambda c, fid: c["family"] == "mistral" and re.search(r"Mistral-Small-3\.2|Mistral-7B", model(fid)) is not None)
finding("llamacpp-kimi-k3-eog-leak", "llamacpp", "engine_bug",
        "Kimi K3: <|end_of_msg|> is both a preserved token and the end-of-generation token, so llama-server renders it and it leaks into content/reasoning.",
        lambda c, fid: c["family"] == "kimi" and "K3" in model(fid) and not marker(fid) and fid not in {"kimi/k3-control-marker-text-in-value"} and any("end_of_msg" in (r["detail"] or "") for r in c["checks"]))
finding("llamacpp-garbled-harmony-channel", "llamacpp", "engine_bug_known_upstream",
        "gpt-oss generations with a malformed Harmony channel name ('commentary?', '??', 'comment') raise 'does not match the expected peg-native format' and fail the whole turn instead of degrading to content.",
        lambda c, fid: c["family"] == "gpt-oss" and FX[fid]["provenance"]["kind"] == "bug_report" and any("peg-native format" in (r["detail"] or "") for r in c["checks"]),
        upstream="https://github.com/ggml-org/llama.cpp/issues/27720",
        notes="Closed by the maintainer on 2026-08-26 as not feasible to handle ('handling garbage output is not feasible from a parsing perspective'); the reporter's generations came from a client that dropped reasoning_content. Do not re-file; the matrix still shows it because a graceful fallback to content is what the fixtures expect.")
finding("llamacpp-rejects-realistic-output", "llamacpp", "engine_bug",
        "Official-template renders and bug-report outputs raise 'does not match the expected peg-native format' instead of parsing or degrading to content (a plain Llama 3 JSON answer, content that starts with a JSON object, Kimi K3 XTML attribute escaping).",
        lambda c, fid: any("peg-native format" in (r["detail"] or "") for r in c["checks"]) and FX[fid]["provenance"]["kind"] != "engine_test")
finding("llamacpp-strict-grammar-variants", "llamacpp", "strict_grammar_variant",
        "Engine-test variants outside the official template's grammar (other engines' test strings: Kimi K2 id/noise variants, GLM-4.5 newline format, Mistral v3 JSON variants, Llama 3 'arguments' key) raise 'does not match the expected peg-native format'. llama.cpp's parser is derived from the template, so rejecting them is strict rather than wrong; raising instead of returning content is still unfriendly.",
        lambda c, fid: any("peg-native format" in (r["detail"] or "") for r in c["checks"]))
finding("llamacpp-marker-text-in-arguments", "llamacpp", "engine_bug",
        "Format-marker text inside argument strings or reasoning breaks the parse.",
        lambda c, fid: marker(fid))
finding("llamacpp-other", "llamacpp", "untriaged_discrepancy",
        "Other llama.cpp discrepancies (text around Gemma calls, GLM-4.7 missing </arg_value> leaks '</tool_call>' into a value, Qwen3-Coder variants, a Kimi K3 tools section cut by max_tokens that non-streaming passes through as content while streaming drops it).",
        lambda c, fid: True)

finding("ollama-gemma4-keys-with-spaces", "ollama", "engine_bug_known_upstream",
        "gemma4: an object key containing spaces drops the whole call.",
        lambda c, fid: fid in {"gemma4/bug-ollama-18390-key-with-spaces", "gemma4/edge-values-key-with-space"},
        upstream="https://github.com/ollama/ollama/issues/18390")
finding("ollama-gemma4-string-placeholder", "ollama", "engine_bug_known_upstream",
        "gemma4: 45 string values followed by a string array drop the call (string-placeholder collision).",
        lambda c, fid: fid == "gemma4/many-strings-then-string-array", upstream="https://github.com/ollama/ollama/issues/18354")
finding("ollama-qwen3coder-int64-clamp", "ollama", "engine_bug_known_upstream",
        "qwen3-coder: a number outside int64 (1e20) is silently converted with Go's int64(f), whose result for out-of-range floats is implementation-defined: it saturates to 9223372036854775807 on arm64 (macOS snapshot) and wraps to -9223372036854775808 on amd64 (Linux CI nightly 36220399447). Either way the argument is corrupted.",
        lambda c, fid: fid == "qwen3-xml/bug-coder-number-outside-int64", upstream="https://github.com/ollama/ollama/issues/18421")
finding("ollama-glm47-trailing-whitespace", "ollama", "engine_bug",
        "glm-4.7: leading/trailing whitespace inside <arg_value> (data, per the template) is stripped ('  two  spaces\\n' -> '  two  spaces').",
        lambda c, fid: fid == "glm/glm47-significant-whitespace")
finding("ollama-marker-text-in-arguments", "ollama", "engine_bug",
        "Format-marker text inside argument strings raises or leaks into content (gemma4, glm-4.7, qwen3, qwen3-coder XML parser, harmony).",
        lambda c, fid: marker(fid) or "tool-call-token-in-arguments" in fid)
finding("ollama-other", "ollama", "untriaged_discrepancy",
        "Other Ollama discrepancies (strict XML on missing </parameter>, bare <function=...> without <tool_call>, [THINK] in one delta, harmony builtin recipient, malformed JSON raising).",
        lambda c, fid: True)

finding("transformers-gemma4-response-template", "transformers", "engine_bug",
        "Gemma 4 response_template (tokenizer.parse_response) raises 'json: could not parse after dialect transforms' for tool names with '-' or '.', object keys with spaces, and marker text in strings.",
        lambda c, fid: any("dialect transforms" in (r["detail"] or "") for r in c["checks"]),
        direct="repro/transformers_gemma4_hyphenated_name.py")
finding("transformers-other", "transformers", "untriaged_discrepancy",
        "Other transformers discrepancies (text before a call dropped in non-streaming, malformed call raises).",
        lambda c, fid: True)

assigned = collections.defaultdict(list)
for eng, d in runs.items():
    for c in d["cases"]:
        if c["status"] != "fail":
            continue
        fid = c["fixture_id"]
        for f in F:
            if f["engine"] == eng and f["sel"](c, fid):
                assigned[f["id"]].append(fid); break
        else:
            print("UNASSIGNED", eng, fid)

out = []
total = collections.Counter()
for f in F:
    ids = assigned.get(f["id"], [])
    if not ids:
        continue
    total[(f["engine"], f["classification"])] += len(ids)
    rep = ids[0]
    out.append({
        "id": f["id"], "engine": f["engine"], "engine_version": runs[f["engine"]]["engine"]["version"],
        "classification": f["classification"], "summary": f["summary"], "upstream": f["upstream"],
        "fixtures": ids,
        "repro": f"uv run canitoolcall run --engine {f['engine']} --id {rep} --observed all --env HF_HUB_OFFLINE=1",
        "direct_repro": f["direct_repro"],
        **({"notes": f["notes"]} if f["notes"] else {}),
    })
with open(f"{SNAP}/triage.jsonl", "w") as fh:
    for o in out:
        fh.write(json.dumps(o, ensure_ascii=False) + "\n")
for k, v in sorted(total.items()):
    print(k, v)
for o in out:
    print(f'{o["engine"]:12} {o["classification"]:28} {len(o["fixtures"]):3}  {o["id"]}')

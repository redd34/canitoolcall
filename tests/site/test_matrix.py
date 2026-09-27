"""Tests for the static matrix site (``canitoolcall.matrix``).

The results files here are written by :class:`RunResults` inside each test.
They are test inputs for the site builder, not published results.
"""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

import pytest

from canitoolcall.cli import EXIT_ERROR, EXIT_OK, main
from canitoolcall.fixtures import fixtures_digest, load_fixtures
from canitoolcall.matrix import (
    Matrix,
    build_matrix,
    case_status,
    cell_path,
    check_stats,
    load_results,
    render_site,
    repro_command,
)
from canitoolcall.results import (
    CaseResult,
    CheckResult,
    EngineInfo,
    Observation,
    ParsedToolCall,
    ParseResult,
    RunInfo,
    RunResults,
    Status,
)

SAMPLE_ID = "qwen3-hermes/sample-parallel-reasoning"
GOOD = ParseResult(
    content=None,
    reasoning_content="I should call the tools.",
    tool_calls=(
        ParsedToolCall("get_weather", '{"city": "Zürich", "unit": "c"}'),
        ParsedToolCall("search", '{"query": "café \\"best\\"", "filters": {"tags": ["a", "b"], "max": 3}}'),
    ),
)
LOST_ARGS = ParseResult(
    content=None,
    reasoning_content="I should call the tools.",
    tool_calls=(ParsedToolCall("get_weather", ""), GOOD.tool_calls[1]),
)


def _check(name: str, strategy: str, status: Status, detail: str | None = None) -> CheckResult:
    return CheckResult(name, strategy, status, detail)


def _passing_checks(strategies: tuple[str, ...] = ("nonstream", "one", "token")) -> tuple[CheckResult, ...]:
    return tuple(_check(n, s, Status.PASS) for s in strategies for n in ("expected_match", "arguments_json"))


def _run(
    engine: str = "vllm",
    version: str = "0.30.0",
    finished: str = "2026-09-25T10:00:00Z",
    cases: tuple[CaseResult, ...] = (),
    digest: str = "0" * 64,
    strategies: tuple[str, ...] = ("one", "token"),
) -> RunResults:
    return RunResults(
        canitoolcall_version="0.1.0.dev0",
        engine=EngineInfo(engine, version, "abc123def4567890", {"transformers": "5.17.0"}),
        run=RunInfo("2026-09-25T09:59:00Z", finished, "linux-aarch64", "3.12.13", digest, strategies, "soft-v1"),
        cases=cases,
    )


def _failing_case(fixture_id: str = SAMPLE_ID, family: str = "qwen3-hermes") -> CaseResult:
    checks = (
        *_passing_checks(("nonstream", "one")),
        _check("expected_match", "token", Status.FAIL, "tool_calls[0].arguments differ"),
        _check("arguments_json", "token", Status.FAIL, "tool_calls[0]: '' is not a JSON object"),
        _check("split_invariance", "token", Status.FAIL, "token differs from one"),
    )
    return CaseResult(
        fixture_id,
        family,
        Status.FAIL,
        checks,
        parser_config={"tool_parser": "hermes", "reasoning_parser": "qwen3"},
        skipped_strategies={"special": "adapter does not expose special token ids"},
        observed=Observation(nonstream=GOOD, streams={"one": GOOD, "token": LOST_ARGS}),
    )


def _sample_runs() -> list[RunResults]:
    vllm_cases = (
        _failing_case(),
        CaseResult("qwen3-hermes/other", "qwen3-hermes", Status.PASS, _passing_checks()),
        CaseResult(
            "qwen3-hermes/soft",
            "qwen3-hermes",
            Status.SOFT_PASS,
            (*_passing_checks(), _check("stream_equals_nonstream", "token", Status.SOFT_PASS, "content: '\\n'")),
        ),
        CaseResult("deepseek/v4", "deepseek", Status.UNSUPPORTED, reason="no parser for DeepSeek-V4.1"),
    )
    sglang_cases = (
        CaseResult(SAMPLE_ID, "qwen3-hermes", Status.PASS, _passing_checks()),
        CaseResult("qwen3-hermes/other", "qwen3-hermes", Status.ERROR, harness_error="worker timed out"),
    )
    return [_run(cases=vllm_cases), _run("sglang", "0.5.20", cases=sglang_cases)]


def _write(tmp_path: Path, runs: list[RunResults]) -> Path:
    results = tmp_path / "results"
    for run in runs:
        run.write(results / run.default_filename())
    results.mkdir(exist_ok=True)
    return results


# --------------------------------------------------------------------------- aggregation


def test_build_matrix_cells_and_counts() -> None:
    m = build_matrix(_sample_runs())
    assert m.families == ("deepseek", "qwen3-hermes")
    assert m.engines == (("sglang", "0.5.20"), ("vllm", "0.30.0"))
    cell = m.cell("qwen3-hermes", "vllm", "0.30.0")
    assert cell is not None
    assert cell.status is Status.FAIL
    assert cell.counts == {"pass": 1, "soft_pass": 1, "fail": 1, "error": 0, "unsupported": 0}
    assert cell.pass_rate == pytest.approx(1 / 3)
    assert cell.run_at == "2026-09-25T10:00:00Z"
    weak = {k.check: (k.counts["pass"], k.applied) for k in cell.weak_checks}
    assert weak == {
        "expected_match": (2, 3),
        "arguments_json": (2, 3),
        "split_invariance": (0, 1),
        "stream_equals_nonstream": (0, 1),
    }

    ds = m.cell("deepseek", "vllm", "0.30.0")
    assert ds is not None and ds.status is Status.UNSUPPORTED and ds.pass_rate is None
    assert m.cell("deepseek", "sglang", "0.5.20") is None
    sg = m.cell("qwen3-hermes", "sglang", "0.5.20")
    assert sg is not None and sg.status is Status.ERROR


def test_latest_run_wins_and_versions_sort_naturally() -> None:
    old = _run(version="0.10.0", finished="2026-09-20T00:00:00Z", cases=(_failing_case(),))
    new = _run(
        version="0.10.0",
        finished="2026-09-24T00:00:00+00:00",
        cases=(CaseResult(SAMPLE_ID, "qwen3-hermes", Status.PASS, _passing_checks()),),
    )
    older_version = _run(version="0.9.2", cases=(_failing_case(),))
    m = build_matrix([new, old, older_version])
    assert m.engines == (("vllm", "0.9.2"), ("vllm", "0.10.0"))
    cell = m.cell("qwen3-hermes", "vllm", "0.10.0")
    assert cell is not None and cell.status is Status.PASS
    assert m.run_for("vllm", "0.10.0") is new


def test_stress_strategies_never_count() -> None:
    case = CaseResult(
        SAMPLE_ID,
        "qwen3-hermes",
        Status.FAIL,
        (*_passing_checks(), _check("split_invariance", "char:1", Status.FAIL, "split special token")),
    )
    assert case_status(case) is Status.PASS
    m = build_matrix([_run(cases=(case,), strategies=("one", "token", "char:1"))])
    cell = m.cell("qwen3-hermes", "vllm", "0.30.0")
    assert cell is not None
    assert cell.status is Status.PASS
    assert cell.stress_failures == 1
    assert "split_invariance" not in {k.check for k in cell.checks}


def test_check_stats_take_worst_strategy_per_case() -> None:
    stats = {k.check: k for k in check_stats([_failing_case(), _failing_case()])}
    assert stats["expected_match"].counts["fail"] == 2
    assert stats["expected_match"].applied == 2
    assert stats["expected_match"].pass_rate == 0.0
    assert "parallel_order" not in stats


def test_repro_command_targets_failing_strategies() -> None:
    cmd = repro_command(_failing_case(), "vllm", "fixtures/qwen3-hermes/sample.jsonl")
    assert cmd == (
        "uv run canitoolcall run --engine vllm --fixtures fixtures/qwen3-hermes/sample.jsonl "
        f"--id {_failing_case().fixture_id} --strategy token --observed all"
    )
    only_nonstream = CaseResult("f/x", "f", Status.FAIL, (_check("expected_match", "nonstream", Status.FAIL),))
    assert repro_command(only_nonstream, "sglang", None).endswith("--family f --id f/x --strategy one --observed all")
    split = CaseResult("f/x", "f", Status.FAIL, (_check("split_invariance", "*", Status.FAIL),))
    assert "--strategy *" not in repro_command(split, "vllm", None)


# --------------------------------------------------------------------------- loading


def test_load_results_roundtrip_and_errors(tmp_path: Path) -> None:
    results = _write(tmp_path, _sample_runs())
    runs = load_results(results)
    assert [r.engine.name for r in runs] == ["sglang", "vllm"]
    assert runs == sorted(_sample_runs(), key=lambda r: r.default_filename())

    (results / "broken.json").write_text('{"schema_version": "0.1"}', encoding="utf-8")
    with pytest.raises(ValueError, match=r"broken\.json: not a valid results file"):
        load_results(results)
    (results / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_results(results)
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path / "missing")


# --------------------------------------------------------------------------- rendering


class _Html(HTMLParser):
    """Collects tags and checks that non-void elements are balanced."""

    VOID = frozenset({"meta", "link", "br", "hr", "img", "input", "wbr", "source", "col", "area", "base"})

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}>, open: {self.stack[-3:]}")
        else:
            self.stack.pop()


def _parse(path: Path) -> _Html:
    p = _Html()
    p.feed(path.read_text(encoding="utf-8"))
    p.close()
    assert not p.errors, p.errors
    assert not p.stack, f"unclosed: {p.stack}"
    return p


def _assert_accessible(page: _Html) -> None:
    tags = [t for t, _ in page.tags]
    html = next(a for t, a in page.tags if t == "html")
    assert html.get("lang") == "en"
    assert tags.count("main") == 1
    assert tags.count("h1") == 1
    assert tags.count("table") == tags.count("caption")
    assert all(a.get("scope") in ("col", "row") for t, a in page.tags if t == "th")


def test_gzipped_results_load_and_publish_as_json(tmp_path: Path) -> None:
    import gzip

    results = _write(tmp_path, _sample_runs())
    plain = results / "vllm-0.30.0.json"
    gz = results / "vllm-0.30.0.json.gz"
    gz.write_bytes(gzip.compress(plain.read_bytes()))
    plain.unlink()
    assert sorted(r.engine.name for r in load_results(results)) == ["sglang", "vllm"]
    out = tmp_path / "site"
    html = render_site(results, out).read_text(encoding="utf-8")
    assert 'href="data/vllm-0.30.0.json"' in html
    assert json.loads((out / "data" / "vllm-0.30.0.json").read_text(encoding="utf-8"))["engine"]["name"] == "vllm"
    gz.write_bytes(b"not gzip")
    with pytest.raises(ValueError, match=r"vllm-0\.30\.0\.json\.gz"):
        load_results(results)


def test_render_site(tmp_path: Path, sample_fixtures_dir: Path) -> None:
    results = _write(tmp_path, _sample_runs())
    out = tmp_path / "site"
    index = render_site(results, out, fixtures_dir=sample_fixtures_dir)
    assert index == out / "index.html"

    html = index.read_text(encoding="utf-8")
    _assert_accessible(_parse(index))
    for text in ("vllm", "0.30.0", "sglang", "0.5.20", "2026-09-25", "Qwen3 (Hermes-style JSON)", "33%"):
        assert text in html
    assert "split_invariance</code> 0/1" in html
    assert 'href="data/vllm-0.30.0.json"' in html
    assert (out / "assets" / "style.css").is_file()
    assert (out / "data" / "vllm-0.30.0.json").read_bytes() == (results / "vllm-0.30.0.json").read_bytes()
    assert (out / ".nojekyll").is_file()
    assert "Built with Llama" in html  # Llama Community License 1.b.i(B)
    # Footer: source repo, issue tracker and CONTRIBUTING (issue #1)
    assert "https://github.com/redd34/canitoolcall" in html
    assert "https://github.com/redd34/canitoolcall/issues/new/choose" in html
    assert "https://github.com/redd34/canitoolcall/blob/main/CONTRIBUTING.md" in html

    data = json.loads((out / "matrix.json").read_text(encoding="utf-8"))
    cells = {(c["family"], c["engine"]): c for c in data["cells"]}
    assert cells[("qwen3-hermes", "vllm")]["status"] == "fail"
    assert cells[("deepseek", "vllm")]["status"] == "unsupported"

    page = out / cell_path("vllm", "0.30.0", "qwen3-hermes")
    assert page.is_file()
    assert cells[("qwen3-hermes", "vllm")]["page"] == "cells/vllm/0.30.0/qwen3-hermes.html"
    parsed = _parse(page)
    _assert_accessible(parsed)
    hrefs = [a.get("href") for t, a in parsed.tags if t == "link"]
    assert hrefs == ["../../../assets/style.css"]
    cell_html = page.read_text(encoding="utf-8")
    assert "https://github.com/redd34/canitoolcall" in cell_html
    assert SAMPLE_ID in cell_html
    repro = f"--fixtures fixtures/qwen3-hermes/sample.jsonl --id {SAMPLE_ID} --strategy token --observed all"
    assert repro in cell_html
    assert "tool_calls[0]: &#39;&#39; is not a JSON object" in cell_html
    assert '<span class="d-add">+' in cell_html  # diff of expected vs the lossy token stream
    assert "&lt;arguments_raw, not valid JSON&gt;" in cell_html
    assert "Matches the expected parse." in cell_html  # nonstream and one agree with the fixture
    assert "&lt;tool_call&gt;" in cell_html  # raw output from the corpus, escaped
    assert "the fixture corpus has changed since this run" in cell_html
    assert "hermes" in cell_html and "qwen3" in cell_html  # parser config
    assert "qwen3-hermes/soft" in cell_html
    assert "qwen3-hermes/other" not in cell_html  # strict passes are not listed

    ds = (out / cell_path("vllm", "0.30.0", "deepseek")).read_text(encoding="utf-8")
    assert "no parser for DeepSeek-V4.1" in ds
    sg = (out / cell_path("sglang", "0.5.20", "qwen3-hermes")).read_text(encoding="utf-8")
    assert "worker timed out" in sg
    assert "<code>special</code> skipped for 1 fixture(s): adapter does not expose special token ids" in cell_html


def test_digest_match_is_reported(tmp_path: Path, sample_fixtures_dir: Path) -> None:
    digest = fixtures_digest(load_fixtures([sample_fixtures_dir]))
    results = _write(tmp_path, [_run(cases=(_failing_case(),), digest=digest)])
    render_site(results, tmp_path / "site", fixtures_dir=sample_fixtures_dir)
    page = (tmp_path / "site" / cell_path("vllm", "0.30.0", "qwen3-hermes")).read_text(encoding="utf-8")
    assert "matches the run" in page


def test_render_without_corpus_and_escapes(tmp_path: Path) -> None:
    evil = CaseResult(
        "qwen3-hermes/x",
        "qwen3-hermes",
        Status.FAIL,
        (_check("no_leakage", "nonstream", Status.FAIL, "<script>alert(1)</script><tool_call> leaked"),),
    )
    results = _write(tmp_path, [_run(version="1.0+cpu/x", cases=(evil,))])
    out = tmp_path / "site"
    render_site(results, out, fixtures_dir=tmp_path / "nowhere")
    page = out / cell_path("vllm", "1.0+cpu/x", "qwen3-hermes")
    assert page.parent.name == "1.0_cpu_x"
    text = page.read_text(encoding="utf-8")
    assert "<script>" not in text
    assert "&lt;tool_call&gt; leaked" in text
    assert "fixture corpus &#39;nowhere&#39; not found" in text
    assert str(tmp_path) not in text  # no absolute local paths in published pages
    assert "does not include the observed parses" in text
    _parse(page)


def test_render_empty_results(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    index = render_site(results, tmp_path / "site")
    text = index.read_text(encoding="utf-8")
    assert "No results yet" in text
    _assert_accessible(_parse(index))


def test_results_schema_rejects_unsafe_fixture_ids(tmp_path: Path) -> None:
    bad = CaseResult("qwen3-hermes/$(touch x)", "qwen3-hermes", Status.FAIL, ())
    results = _write(tmp_path, [_run(cases=(bad,))])
    with pytest.raises(ValueError, match="fixture_id"):
        render_site(results, tmp_path / "site")


def test_cli_matrix_empty_results_dir_hints_at_snapshots(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    results = tmp_path / "results"
    _write(results / "x", _sample_runs()).rename(results / "2026-01-01")
    (results / "x").rmdir()
    assert main(["matrix", "--results", str(results), "--out", str(tmp_path / "o")]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert "no results files" in err and "2026-01-01" in err


def test_repro_command_quotes_untrusted_values() -> None:
    case = CaseResult(
        "qwen3-hermes/a", "qwen3-hermes", Status.FAIL, (_check("expected_match", "rand:1:8", Status.FAIL),)
    )
    cmd = repro_command(case, "vllm; rm -rf ~", "fixtures/q h/x.jsonl")
    assert "'vllm; rm -rf ~'" in cmd and "'fixtures/q h/x.jsonl'" in cmd
    # One-token-per-step engines never fall back to the multi-token "one" strategy.
    ok = CaseResult(
        "qwen3-hermes/a", "qwen3-hermes", Status.FAIL, (_check("expected_match", "nonstream", Status.FAIL),)
    )
    assert "--strategy token" in repro_command(ok, "ollama", None, ("one", "special"))


def test_non_http_provenance_url_is_not_a_link(tmp_path: Path) -> None:
    from canitoolcall.matrix import _http_url

    assert _http_url("javascript:alert(document.domain)") is None
    assert _http_url("https://github.com/x") == "https://github.com/x"
    assert _http_url("http:no-host") is None


def test_cli_matrix(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    results = _write(tmp_path, _sample_runs())
    assert main(["matrix", "--results", str(results), "--out", str(tmp_path / "o")]) == EXIT_OK
    assert "index.html" in capsys.readouterr().out


def test_missing_templates(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no site templates"):
        render_site(tmp_path, tmp_path / "o", tmp_path / "no-templates")


def test_matrix_cell_lookup_on_empty() -> None:
    assert Matrix((), (), ()).cell("x", "vllm", "1") is None


def test_run_synthetic_strategies_do_not_count(tmp_path: Path) -> None:
    from dataclasses import replace

    from canitoolcall.matrix import build_cell

    checks = (*_passing_checks(("nonstream", "token")), _check("expected_match", "one", Status.FAIL, "lost"))
    case = CaseResult(SAMPLE_ID, "qwen3-hermes", Status.PASS, checks)
    one_token = _run("ollama", "7af39318", cases=(case,))
    one_token = replace(one_token, run=replace(one_token.run, synthetic_strategies=("one",)))
    cell = build_cell("qwen3-hermes", one_token)
    assert cell.status is Status.PASS and cell.stress_failures == 1
    assert cell.multi_token_synthetic == ("one",)
    assert case_status(case, ("one",)) is Status.PASS
    page = render_site(_write(tmp_path, [one_token]), tmp_path / "site")
    cell_page = (page.parent / cell_path("ollama", "7af39318", "qwen3-hermes")).read_text(encoding="utf-8")
    assert "streams one token per event" in cell_page

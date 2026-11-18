from __future__ import annotations

import sys
from pathlib import Path

import pytest
from core_corpus import write_corpus

from canitoolcall.cli import EXIT_ERROR, EXIT_FAILURES, EXIT_OK, main
from canitoolcall.results import RunResults


def test_validate_ok(sample_fixtures_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", str(sample_fixtures_dir)]) == EXIT_OK
    assert "1 file(s) checked, 0 issue(s)" in capsys.readouterr().out


def test_validate_bad(tmp_path: Path) -> None:
    d = tmp_path / "fam"
    d.mkdir()
    (d / "x.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
    assert main(["validate", str(tmp_path)]) == EXIT_FAILURES


def test_engines_lists_all(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["engines"]) == EXIT_OK
    out = capsys.readouterr().out
    for engine in ("vllm", "sglang", "llamacpp", "ollama", "transformers"):
        assert engine in out


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert capsys.readouterr().out.startswith("canitoolcall ")


def test_matrix_delegates_to_matrix_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import canitoolcall.matrix

    seen: dict[str, object] = {}

    def fake_render(
        results: Path,
        out: Path,
        templates: Path | None = None,
        *,
        fixtures_dir: Path | None = None,
        site_url: str | None = None,
    ) -> Path:
        seen.update(results=results, out=out, templates=templates, fixtures_dir=fixtures_dir, site_url=site_url)
        return out / "index.html"

    monkeypatch.setattr(canitoolcall.matrix, "render_site", fake_render)
    rc = main(["matrix", "--results", str(tmp_path / "r"), "--out", str(tmp_path / "o")])
    assert rc == EXIT_OK
    assert seen == {
        "results": tmp_path / "r",
        "out": tmp_path / "o",
        "templates": None,
        "fixtures_dir": None,
        "site_url": None,
    }
    assert "index.html" in capsys.readouterr().out
    rc = main(["matrix", "--results", str(tmp_path / "r"), "--out", str(tmp_path / "o"), "--fixtures", "fx"])
    assert rc == EXIT_OK and seen["fixtures_dir"] == Path("fx")


def test_matrix_missing_or_invalid_results_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["matrix", "--results", str(tmp_path / "nope"), "--out", str(tmp_path / "o")]) == EXIT_ERROR
    assert "canitoolcall matrix:" in capsys.readouterr().err
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "x.json").write_text('{"not": "a results file"}', encoding="utf-8")
    assert main(["matrix", "--results", str(bad), "--out", str(tmp_path / "o")]) == EXIT_ERROR
    assert "canitoolcall matrix:" in capsys.readouterr().err


def test_validate_missing_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", str(tmp_path / "nope")]) == EXIT_ERROR
    assert "no such path" in capsys.readouterr().err


# --------------------------------------------------------------------------- run


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return write_corpus(tmp_path / "fixtures")


def _run_args(corpus: Path, out: Path, spec: str, env: dict[str, str], *extra: str) -> list[str]:
    args = ["run", "--engine", spec, "--fixtures", str(corpus), "--python", sys.executable, "--out", str(out)]
    args += [a for k, v in env.items() for a in ("--env", f"{k}={v}")]
    return [*args, *extra]


def test_run_passes(
    tmp_path: Path,
    corpus: Path,
    reference_adapter_spec: str,
    reference_env: dict[str, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(_run_args(corpus, tmp_path / "res", reference_adapter_spec, reference_env, "--jobs", "2"))
    out = capsys.readouterr().out
    assert rc == EXIT_OK, out
    assert (tmp_path / "res" / "reference-1.0.json").is_file()
    assert "reference 1.0: 4 case(s)" in out
    assert "pass=3" in out and "unsupported=1" in out


def test_run_failures_exit_1(
    tmp_path: Path,
    corpus: Path,
    reference_adapter_spec: str,
    reference_env: dict[str, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = {**reference_env, "REFERENCE_ADAPTER_BUG": "leak"}
    rc = main(
        _run_args(corpus, tmp_path / "res", reference_adapter_spec, env, "--strategy", "one", "--observed", "all")
    )
    assert rc == EXIT_FAILURES
    loaded = RunResults.load(tmp_path / "res" / "reference-1.0.json")
    assert loaded.run.strategies == ("one",)
    assert loaded.summary()["totals"]["fail"] >= 1


def test_run_filters(
    tmp_path: Path,
    corpus: Path,
    reference_adapter_spec: str,
    reference_env: dict[str, str],
) -> None:
    args = _run_args(corpus, tmp_path / "res", reference_adapter_spec, reference_env)
    assert main([*args, "--tag", "truncated", "--id", "qwen3-hermes/*", "--family", "qwen3-hermes"]) == EXIT_OK
    loaded = RunResults.load(tmp_path / "res" / "reference-1.0.json")
    assert [c.fixture_id for c in loaded.cases] == ["qwen3-hermes/derived-truncated"]


def test_run_usage_errors_exit_2(
    tmp_path: Path,
    corpus: Path,
    reference_adapter_spec: str,
    reference_env: dict[str, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = _run_args(corpus, tmp_path / "res", reference_adapter_spec, reference_env)
    assert main([*base, "--family", "nope"]) == EXIT_ERROR
    assert "no fixtures selected" in capsys.readouterr().err
    assert main(_run_args(corpus, tmp_path / "res", "missing_mod:Adapter", reference_env)) == EXIT_ERROR
    assert "canitoolcall run:" in capsys.readouterr().err
    assert main([*base, "--strategy", "bogus:1"]) == EXIT_ERROR
    with pytest.raises(SystemExit) as e:
        main(["run", "--engine", "tgi"])
    assert e.value.code == 2
    with pytest.raises(SystemExit):
        main([*base, "--jobs", "0"])
    with pytest.raises(SystemExit):
        main([*base, "--env", "NOEQUALS"])
    assert not (tmp_path / "res").exists()


def test_run_rejects_invalid_fixtures(
    tmp_path: Path,
    corpus: Path,
    reference_adapter_spec: str,
    reference_env: dict[str, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    (corpus / "qwen3-hermes" / "family.json").unlink()
    base = _run_args(corpus, tmp_path / "res", reference_adapter_spec, reference_env, "--family", "other")
    assert main(base) == EXIT_ERROR
    assert "missing family.json" in capsys.readouterr().err
    assert main([*base, "--no-validate"]) == EXIT_OK


def test_engines_shows_setup_status(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import canitoolcall.adapters as adapters

    monkeypatch.setattr("canitoolcall.fixtures.repo_root", lambda: tmp_path)
    monkeypatch.setenv("CANITOOLCALL_VLLM_PYTHON", sys.executable)
    for engine in adapters.ENGINES:
        if engine != "vllm":
            monkeypatch.delenv(f"CANITOOLCALL_{engine.upper()}_PYTHON", raising=False)
    assert main(["engines"]) == EXIT_OK
    lines = {line.split()[0]: line for line in capsys.readouterr().out.splitlines()[1:]}
    assert "ready" in lines["vllm"] and sys.executable in lines["vllm"]
    assert "not set up" in lines["sglang"] and "scripts/engines/sglang.sh" in lines["sglang"]


def test_run_unset_engine_is_one_line(
    tmp_path: Path, corpus: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "no-venv" / "bin" / "python"
    rc = main(["run", "--engine", "vllm", "--fixtures", str(corpus), "--python", str(missing), "--out", str(tmp_path)])
    assert rc == EXIT_ERROR
    err = capsys.readouterr().err
    assert "vllm is not set up" in err and "scripts/engines/vllm.sh" in err and "Traceback" not in err

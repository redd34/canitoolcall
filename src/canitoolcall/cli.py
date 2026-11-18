"""Command-line interface: ``canitoolcall {run,probe,matrix,lookup,validate,engines}``.

* ``run --engine E``        offline suite: replay fixtures through engine E's parsers
* ``probe --base-url --model``  live check of an OpenAI-compatible endpoint
* ``matrix``               build the static matrix site from ``results/*.json``
* ``lookup NAME``          does family or model NAME parse on an engine? (reads matrix.json)
* ``validate [PATH...]``   validate fixtures and family.json against the spec
* ``engines``              list known engines and the interpreter each would use

Exit codes: 0 success; 1 conformance failures (run/probe) or invalid fixtures
(validate); 2 usage or harness error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from canitoolcall import __version__

EXIT_OK = 0
EXIT_FAILURES = 1
EXIT_ERROR = 2

DEFAULT_API_KEY_ENV = "CANITOOLCALL_API_KEY"
"""Default ``--api-key-env``: tool-specific, so an exported OPENAI_API_KEY is never sent by accident."""


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse tree (exposed for tests and docs)."""
    from canitoolcall.adapters import ENGINES

    p = argparse.ArgumentParser(
        prog="canitoolcall",
        description="Conformance suite for tool-call and reasoning parsers across inference engines.",
    )
    p.add_argument("--version", action="version", version=f"canitoolcall {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    r = sub.add_parser("run", help="replay fixtures through an engine's parsers (offline)")
    r.add_argument(
        "--engine",
        required=True,
        type=_engine_arg,
        metavar="ENGINE",
        help=f"one of {', '.join(sorted(ENGINES))}, or an adapter spec package.module:Class",
    )
    r.add_argument(
        "--fixtures",
        nargs="*",
        type=Path,
        default=[],
        metavar="PATH",
        help="fixture files/dirs (default: the fixtures root)",
    )
    r.add_argument("--family", action="append", default=[], metavar="SLUG", help="only this family (repeatable)")
    r.add_argument(
        "--strategy",
        action="append",
        default=[],
        metavar="ID",
        help="chunking strategy id, e.g. one, token, rand:1:8 (repeatable; default set if omitted)",
    )
    r.add_argument(
        "--python",
        type=Path,
        default=None,
        help="engine interpreter (default: $CANITOOLCALL_<ENGINE>_PYTHON or .venvs/<engine>)",
    )
    r.add_argument("--out", type=Path, default=Path("results"), help="results directory (default: results/)")
    r.add_argument(
        "--observed",
        choices=["all", "failures", "none"],
        default="failures",
        help="which cases keep the observed parses in the results file",
    )
    r.add_argument("--timeout", type=float, default=120.0, help="per-fixture worker timeout in seconds")
    r.add_argument("--tag", action="append", default=[], metavar="TAG", help="only fixtures with this tag (repeatable)")
    r.add_argument(
        "--id", action="append", default=[], metavar="PATTERN", help="only fixture ids matching this glob (repeatable)"
    )
    r.add_argument("--jobs", "-j", type=_positive_int, default=1, help="worker processes to run in parallel")
    r.add_argument(
        "--env",
        action="append",
        default=[],
        type=_env_arg,
        metavar="KEY=VALUE",
        help="extra environment for the worker, e.g. HF_HUB_OFFLINE=1 (repeatable)",
    )
    r.add_argument("--no-validate", action="store_true", help="skip validating the fixtures against the spec")

    pr = sub.add_parser("probe", help="check a live OpenAI-compatible endpoint")
    pr.add_argument(
        "--base-url",
        required=True,
        type=_base_url_arg,
        help="http(s) URL of the API root, e.g. http://localhost:8000/v1",
    )
    pr.add_argument("--model", required=True, help="model name to send in each request")
    pr.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        metavar="VAR",
        help=(
            f"environment variable holding the API key (default: {DEFAULT_API_KEY_ENV}; unset means no key). "
            "Keys are never passed on the command line; pass --api-key-env OPENAI_API_KEY explicitly to send that key"
        ),
    )
    pr.add_argument(
        "--allow-insecure",
        action="store_true",
        help="allow sending an API key over plain http:// to a non-loopback host",
    )
    pr.add_argument("--no-stream", action="store_true", help="skip the streaming variants")
    pr.add_argument("--timeout", type=float, default=60.0, help="per-request timeout in seconds (default 60)")
    pr.add_argument("--concurrency", type=_positive_int, default=4, help="parallel requests (default 4)")
    pr.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="PATH",
        help="also write the report as JSON ('-' writes JSON to stdout and the table to stderr)",
    )

    lk = sub.add_parser("lookup", help="does family X parse on engine Y? (reads the published matrix.json)")
    lk.add_argument("name", help="family slug, family name or model name (e.g. qwen3-hermes, Qwen/Qwen3.5-9B)")
    lk.add_argument("--engine", default=None, help="only this engine (vllm, sglang, ...)")
    lk.add_argument(
        "--matrix",
        default=None,
        metavar="PATH_OR_URL",
        help="matrix.json to read (default: the published https://redd34.github.io/canitoolcall/matrix.json)",
    )
    lk.add_argument(
        "--json",
        action="store_true",
        help="print the answer as JSON. The exit code is 0 whenever the lookup works, even if the status is "
        "fail: scripts should read the status from this output",
    )

    m = sub.add_parser("matrix", help="build the static matrix site from results files")
    m.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="directory of <engine>-<version>.json[.gz] results files (default: results/)",
    )
    m.add_argument("--out", type=Path, default=Path("site/_build"), help="output directory (default: site/_build)")
    m.add_argument(
        "--templates", type=Path, default=None, help="Jinja template directory (default: the bundled templates)"
    )
    m.add_argument(
        "--fixtures",
        type=Path,
        default=None,
        help=(
            "fixture corpus for drill-down pages (default: $CANITOOLCALL_FIXTURES, else the checkout's "
            "fixtures/, else the corpus bundled with the package)"
        ),
    )
    m.add_argument(
        "--site-url",
        default=None,
        help="absolute URL the site will be published at; adds canonical and preview links and a sitemap.xml",
    )

    v = sub.add_parser("validate", help="validate fixtures against the spec")
    v.add_argument("paths", nargs="*", type=Path, help="fixture files/dirs (default: the fixtures root)")

    sub.add_parser("engines", help="list known engines and their interpreters")
    return p


def _engine_arg(value: str) -> str:
    from canitoolcall.adapters import ENGINES, is_adapter_spec

    if value in ENGINES or is_adapter_spec(value):
        return value
    raise argparse.ArgumentTypeError(f"unknown engine {value!r} (choose from {', '.join(sorted(ENGINES))})")


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def _base_url_arg(value: str) -> str:
    from canitoolcall.probe import check_base_url

    try:
        return check_base_url(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _env_arg(value: str) -> tuple[str, str]:
    key, sep, val = value.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {value!r}")
    return key, val


def cmd_run(args: argparse.Namespace) -> int:
    from canitoolcall.chunking import parse_strategies
    from canitoolcall.runner import FixtureValidationError, RunConfig, WorkerError, run

    try:
        cfg = RunConfig(
            engine=args.engine,
            fixtures=tuple(args.fixtures),
            families=tuple(args.family),
            strategies=parse_strategies(args.strategy),
            python=args.python,
            out_dir=args.out,
            include_observed=args.observed,
            timeout_s=args.timeout,
            env=dict(args.env),
            tags=tuple(args.tag),
            ids=tuple(args.id),
            jobs=args.jobs,
            validate=not args.no_validate,
        )
        results = run(cfg)
    except WorkerError as e:
        if e.unavailable:
            from canitoolcall.adapters import PYTHON_ENV, is_adapter_spec

            hint = (
                ""
                if is_adapter_spec(args.engine)
                else f": run `bash scripts/engines/{args.engine}.sh` or set "
                f"{PYTHON_ENV.format(ENGINE=args.engine.upper())} to an interpreter that has it"
            )
            print(f"canitoolcall run: {args.engine} is not set up{hint}", file=sys.stderr)
            print(f"  ({str(e).strip().splitlines()[-1]})", file=sys.stderr)
        else:
            print(f"canitoolcall run: {e}", file=sys.stderr)
        return EXIT_ERROR
    except (FixtureValidationError, ValueError, FileNotFoundError) as e:
        print(f"canitoolcall run: {e}", file=sys.stderr)
        return EXIT_ERROR
    path = results.write(cfg.out_dir / results.default_filename())
    summary = results.summary()
    totals = summary["totals"]
    print(f"{results.engine.name} {results.engine.version}: {len(results.cases)} case(s) -> {path}")
    for fam, counts in summary["by_family"].items():
        print(f"  {fam:14} " + "  ".join(f"{k}={v}" for k, v in counts.items() if v))
    print("  total          " + "  ".join(f"{k}={v}" for k, v in totals.items()))
    return EXIT_FAILURES if totals["fail"] or totals["error"] else EXIT_OK


def cmd_probe(args: argparse.Namespace) -> int:
    import os

    from canitoolcall.probe import ProbeRequestError, check_reachable, insecure_key_transport, probe, safe_url

    api_key = os.environ.get(args.api_key_env) or None
    if api_key and insecure_key_transport(args.base_url) and not args.allow_insecure:
        print(
            f"canitoolcall probe: refusing to send the key in ${args.api_key_env} over plain http to "
            f"{safe_url(args.base_url)}; use https, unset the variable, or pass --allow-insecure",
            file=sys.stderr,
        )
        return EXIT_ERROR
    try:
        check_reachable(args.base_url, timeout_s=min(args.timeout, 10.0))
    except ProbeRequestError as e:
        print(f"canitoolcall probe: {e}", file=sys.stderr)
        return EXIT_ERROR
    report = probe(
        args.base_url,
        args.model,
        api_key=api_key,
        stream_modes=(False,) if args.no_stream else (False, True),
        timeout_s=args.timeout,
        concurrency=args.concurrency,
    )
    to_stdout = args.json is not None and str(args.json) == "-"
    print(report.render_text(), file=sys.stderr if to_stdout else sys.stdout)
    if args.json is not None:
        blob = json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n"
        if to_stdout:
            sys.stdout.write(blob)
        else:
            args.json.write_text(blob, encoding="utf-8")
    return EXIT_FAILURES if any(o.status in ("fail", "error") for o in report.outcomes) else EXIT_OK


def cmd_matrix(args: argparse.Namespace) -> int:
    from canitoolcall.matrix import no_results_hint, render_site, results_files

    if args.results.is_dir() and not results_files(args.results):
        print(f"canitoolcall matrix: {no_results_hint(args.results)}", file=sys.stderr)
        return EXIT_ERROR
    try:
        index = render_site(args.results, args.out, args.templates, fixtures_dir=args.fixtures, site_url=args.site_url)
    except (FileNotFoundError, ValueError) as e:
        print(f"canitoolcall matrix: {e}", file=sys.stderr)
        return EXIT_ERROR
    print(f"wrote {index}")
    return EXIT_OK


def cmd_lookup(args: argparse.Namespace) -> int:
    from canitoolcall.lookup import DEFAULT_MATRIX, LookupFailure, format_table, load_matrix, lookup, page_base

    source = args.matrix or DEFAULT_MATRIX
    try:
        data = load_matrix(source)  # rejects a malformed URL before page_base sees it
        result = lookup(data, args.name, args.engine, page_base(source))
    except LookupFailure as e:
        print(f"canitoolcall lookup: {e}", file=sys.stderr)
        return EXIT_ERROR
    print(json.dumps(result, indent=2) if args.json else format_table(result))
    return EXIT_OK


def cmd_validate(args: argparse.Namespace) -> int:
    from canitoolcall.fixtures import iter_fixture_files, validate

    paths = args.paths or None
    missing = [p for p in paths or [] if not p.exists()]
    if missing:
        print(f"canitoolcall validate: no such path: {', '.join(map(str, missing))}", file=sys.stderr)
        return EXIT_ERROR
    files = list(iter_fixture_files(paths))
    issues = validate(paths)
    for issue in issues:
        print(issue, file=sys.stderr)
    print(f"{len(files)} file(s) checked, {len(issues)} issue(s)")
    return EXIT_FAILURES if issues else EXIT_OK


def cmd_engines(args: argparse.Namespace) -> int:
    from canitoolcall.adapters import ENGINES, adapter_class, engine_setup

    print(f"{'ENGINE':13} {'STATUS':13} {'PINNED':42} PYTHON")
    for name in sorted(ENGINES):
        cls = adapter_class(name)
        setup = engine_setup(name)
        status = "ready" if setup.configured else "not set up"
        python = str(setup.python) if setup.configured else f"(run scripts/engines/{name}.sh)"
        print(f"{name:13} {status:13} {cls.pinned_version:42} {python}")
    return EXIT_OK


COMMANDS = {
    "run": cmd_run,
    "probe": cmd_probe,
    "matrix": cmd_matrix,
    "lookup": cmd_lookup,
    "validate": cmd_validate,
    "engines": cmd_engines,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point."""
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except NotImplementedError as e:
        print(f"canitoolcall {args.command}: not implemented yet ({e or 'stub'})", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())

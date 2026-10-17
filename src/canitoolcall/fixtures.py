"""Fixture and family loading, validation and digests.

Fixtures live in ``fixtures/<family>/*.jsonl`` (one JSON object per line) next
to ``fixtures/<family>/family.json``. The on-disk format is defined by
``spec/fixture.schema.json`` and ``spec/family.schema.json``.

This module imports with the standard library only. ``jsonschema`` is imported
lazily by :func:`validate` (it runs in the main dev env, never in engine venvs).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from canitoolcall import SPEC_VERSION

ProvenanceKind = Literal["template_render", "engine_test", "bug_report", "recorded", "spec_example"]
ErrorOutcome = Literal["no_tool_calls", "content_passthrough", "exception"]

FIXTURES_ENV = "CANITOOLCALL_FIXTURES"
"""Environment variable that overrides the fixtures root directory."""

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent.parent


# --------------------------------------------------------------------------- paths


def repo_root() -> Path | None:
    """Return the source checkout root when running from a checkout, else None."""
    return _REPO_ROOT if (_REPO_ROOT / "pyproject.toml").is_file() else None


def default_fixtures_dir() -> Path:
    """Resolve the fixtures root: ``$CANITOOLCALL_FIXTURES``, the checkout's
    ``fixtures/``, or the copy shipped in the wheel (``canitoolcall/_data``)."""
    env = os.environ.get(FIXTURES_ENV)
    if env:
        return Path(env)
    root = repo_root()
    if root is not None and (root / "fixtures").is_dir():
        return root / "fixtures"
    return _PKG_DIR / "_data" / "fixtures"


def spec_dir() -> Path:
    """Directory holding the JSON Schemas (checkout ``spec/`` or wheel data)."""
    root = repo_root()
    if root is not None and (root / "spec").is_dir():
        return root / "spec"
    return _PKG_DIR / "_data" / "spec"


# --------------------------------------------------------------------------- models


@dataclass(frozen=True)
class Provenance:
    """Where a fixture's raw output comes from. Mandatory on every fixture."""

    kind: ProvenanceKind
    source_url: str
    revision: str
    license: str
    generator: str | None = None
    template_sha256: str | None = None
    attribution: str | None = None


@dataclass(frozen=True)
class TokenizerPin:
    """Tokenizer that produced ``Fixture.output_token_ids``."""

    repo: str
    revision: str
    mode: Literal["hf", "mistral"] = "hf"


@dataclass(frozen=True)
class ExpectedToolCall:
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class Expected:
    """The correct parse. ``None`` means absent; loaders map ``""`` to ``None``."""

    content: str | None
    reasoning_content: str | None
    tool_calls: tuple[ExpectedToolCall, ...]


@dataclass(frozen=True)
class ExpectedError:
    """Graceful-failure expectation for truncated/malformed outputs."""

    reason: str
    accept: tuple[ErrorOutcome, ...]


@dataclass(frozen=True)
class Finish:
    """Recorded generation termination, not the family's possible stop tokens.

    Stop-without-token semantics are deferred; explicit stops name the token.
    """

    reason: Literal["stop", "length"]
    token: str | None

    def __post_init__(self) -> None:
        if self.reason not in ("stop", "length"):
            raise ValueError("finish.reason must be stop or length")
        if self.reason == "length":
            if self.token is not None:
                raise ValueError("finish length must have token null")
        elif not isinstance(self.token, str) or not self.token:
            raise ValueError("finish stop requires a non-empty token")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Finish:
        if not isinstance(value, Mapping) or set(value) != {"reason", "token"}:
            raise ValueError("finish must contain exactly reason and token")
        return cls(reason=value["reason"], token=value["token"])

    def to_dict(self) -> dict[str, str | None]:
        return {"reason": self.reason, "token": self.token}


def validate_replay_conditions(
    assumes: tuple[str, ...] | None, finish: Finish | None, truncated: bool, text: str
) -> None:
    """Validate metadata even in stdlib-only workers that bypass JSON Schema."""
    if assumes is not None:
        if not isinstance(assumes, tuple) or any(item != "stop_token_ignored" for item in assumes):
            raise ValueError("assumes permits only stop_token_ignored")
        if len(set(assumes)) != len(assumes):
            raise ValueError("assumes entries must be unique")
    if finish is not None:
        if not isinstance(finish, Finish):
            raise ValueError("finish must be a Finish value")
        if finish.reason == "stop" and truncated:
            raise ValueError("finish stop contradicts the truncated tag")
        if finish.token is not None and text.endswith(finish.token):
            raise ValueError("raw_output must end before finish.token")


@dataclass(frozen=True)
class Fixture:
    """One fixture record. See spec/README.md for field semantics."""

    id: str
    family: str
    models: tuple[str, ...]
    spec_version: str
    provenance: Provenance
    tools: tuple[Mapping[str, Any], ...]
    raw_output: str
    tags: tuple[str, ...]
    expected: Expected | None = None
    expected_error: ExpectedError | None = None
    output_token_ids: tuple[int, ...] | None = None
    tokenizer: TokenizerPin | None = None
    generation_prompt: str | None = None
    thinking: bool | None = None
    notes: str | None = None
    source: Path | None = field(default=None, compare=False)
    """File the record was loaded from (not part of the record)."""
    line: int | None = field(default=None, compare=False)
    """1-based line number in ``source`` (not part of the record)."""

    assumes: tuple[str, ...] | None = None
    """Optional replay preconditions. None preserves an absent JSON field."""
    finish: Finish | None = None

    def __post_init__(self) -> None:
        validate_replay_conditions(self.assumes, self.finish, "truncated" in self.tags, self.raw_output)

    @property
    def reference_model(self) -> str:
        """``models[0]``: the model whose tokenizer/template adapters use."""
        return self.models[0]

    @property
    def truncated(self) -> bool:
        """True for explicit length termination, else the legacy ``truncated`` tag.

        The model emitted no stop token, so adapters must replay the output
        with ``finish_reason: "length"`` and never append one.
        """
        return self.finish.reason == "length" if self.finish is not None else "truncated" in self.tags

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], source: Path | None = None, line: int | None = None) -> Fixture:
        """Build a Fixture from a decoded JSONL record (assumed schema-valid)."""
        exp = d.get("expected")
        err = d.get("expected_error")
        tok = d.get("tokenizer")
        ids = d.get("output_token_ids")
        if "assumes" in d and not isinstance(d["assumes"], list):
            raise ValueError("assumes must be an array")
        return cls(
            id=d["id"],
            family=d["family"],
            models=tuple(d["models"]),
            spec_version=d["spec_version"],
            provenance=Provenance(**d["provenance"]),
            tools=tuple(d["tools"]),
            raw_output=d["raw_output"],
            tags=tuple(d["tags"]),
            expected=(
                Expected(
                    content=exp["content"] or None,
                    reasoning_content=exp["reasoning_content"] or None,
                    tool_calls=tuple(ExpectedToolCall(tc["name"], tc["arguments"]) for tc in exp["tool_calls"]),
                )
                if exp is not None
                else None
            ),
            expected_error=ExpectedError(err["reason"], tuple(err["accept"])) if err is not None else None,
            output_token_ids=tuple(ids) if ids is not None else None,
            tokenizer=TokenizerPin(**tok) if tok is not None else None,
            generation_prompt=d.get("generation_prompt"),
            thinking=d.get("thinking"),
            notes=d.get("notes"),
            source=source,
            line=line,
            assumes=tuple(d["assumes"]) if "assumes" in d else None,
            finish=Finish.from_dict(d["finish"]) if "finish" in d else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to the JSONL record shape (omits unset optional fields)."""
        prov = {k: v for k, v in vars(self.provenance).items() if v is not None}
        out: dict[str, Any] = {
            "id": self.id,
            "family": self.family,
            "models": list(self.models),
            "spec_version": self.spec_version,
            "provenance": prov,
            "tools": [dict(t) for t in self.tools],
            "raw_output": self.raw_output,
        }
        if self.output_token_ids is not None:
            out["output_token_ids"] = list(self.output_token_ids)
        if self.tokenizer is not None:
            out["tokenizer"] = vars(self.tokenizer).copy()
        if self.generation_prompt is not None:
            out["generation_prompt"] = self.generation_prompt
        if self.thinking is not None:
            out["thinking"] = self.thinking
        if self.assumes is not None:
            out["assumes"] = list(self.assumes)
        if self.finish is not None:
            out["finish"] = self.finish.to_dict()
        if self.expected is not None:
            out["expected"] = {
                "content": self.expected.content,
                "reasoning_content": self.expected.reasoning_content,
                "tool_calls": [{"name": tc.name, "arguments": dict(tc.arguments)} for tc in self.expected.tool_calls],
            }
        if self.expected_error is not None:
            out["expected_error"] = {
                "reason": self.expected_error.reason,
                "accept": list(self.expected_error.accept),
            }
        out["tags"] = list(self.tags)
        if self.notes is not None:
            out["notes"] = self.notes
        return out


@dataclass(frozen=True)
class ReferenceModel:
    repo: str
    revision: str
    variant: str | None = None
    default_generation_prompt: str | None = None
    stop_tokens: tuple[str, ...] = ()
    gated: bool = False
    tokenizer_mode: Literal["hf", "mistral"] = "hf"
    mirror: tuple[str, str] | None = None
    """``(repo, revision)`` of the ungated mirror fixtures pin as ``tokenizer`` (gated repos only)."""

    @classmethod
    def from_dict(cls, m: Mapping[str, Any]) -> ReferenceModel:
        mirror = m.get("mirror")
        return cls(
            **{
                **m,
                "stop_tokens": tuple(m.get("stop_tokens", ())),
                "mirror": (mirror["repo"], mirror["revision"]) if mirror else None,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in vars(self).items():
            if v is None:
                continue
            if k == "mirror":
                out[k] = {"repo": v[0], "revision": v[1]}
            else:
                out[k] = list(v) if isinstance(v, tuple) else v
        return out

    def allows_tokenizer(self, pin: TokenizerPin) -> bool:
        """Whether a fixture of this model may pin ``pin``: the model itself or its declared mirror."""
        return (pin.repo, pin.revision) in {(self.repo, self.revision), *([self.mirror] if self.mirror else [])}


@dataclass(frozen=True)
class Family:
    """``fixtures/<slug>/family.json``."""

    slug: str
    name: str
    spec_version: str
    has_reasoning: bool
    markers: tuple[str, ...]
    reference_models: tuple[ReferenceModel, ...]
    format_notes: str
    notes: str | None = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Family:
        return cls(
            slug=d["slug"],
            name=d["name"],
            spec_version=d["spec_version"],
            has_reasoning=d["has_reasoning"],
            markers=tuple(d["markers"]),
            reference_models=tuple(ReferenceModel.from_dict(m) for m in d["reference_models"]),
            format_notes=d["format_notes"],
            notes=d.get("notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "slug": self.slug,
            "name": self.name,
            "spec_version": self.spec_version,
            "has_reasoning": self.has_reasoning,
            "markers": list(self.markers),
            "reference_models": [m.to_dict() for m in self.reference_models],
            "format_notes": self.format_notes,
        }
        if self.notes is not None:
            out["notes"] = self.notes
        return out

    def reference_for(self, model: str) -> ReferenceModel | None:
        """The reference-model entry for ``model`` (exact repo match), if any."""
        return next((m for m in self.reference_models if m.repo == model), None)


# --------------------------------------------------------------------------- loading


def iter_fixture_files(paths: Iterable[Path] | None = None) -> Iterator[Path]:
    """Yield ``*.jsonl`` files under the given files/directories (sorted).

    With ``paths=None`` walks :func:`default_fixtures_dir`.
    """
    for p in paths if paths is not None else [default_fixtures_dir()]:
        p = Path(p)
        if p.is_dir():
            yield from sorted(p.rglob("*.jsonl"))
        elif p.suffix == ".jsonl":
            yield p


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, record)`` for each non-blank line of ``path``.

    Raises ``ValueError`` with the location on malformed JSON.
    """
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield n, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON: {e}") from e


def load_fixtures(
    paths: Iterable[Path] | None = None,
    families: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
) -> list[Fixture]:
    """Load fixtures (no schema validation; run :func:`validate` for that).

    Raises ``ValueError`` (with the file and line) for invalid JSON or a
    record that lacks required fields.

    Args:
        paths: files or directories; defaults to :func:`default_fixtures_dir`.
        families: keep only these family slugs.
        tags: keep only fixtures carrying at least one of these tags.
    """
    out: list[Fixture] = []
    for f in iter_fixture_files(paths):
        for n, rec in read_jsonl(f):
            try:
                fx = Fixture.from_dict(rec, source=f, line=n)
            except (KeyError, TypeError, AttributeError, ValueError) as e:
                raise ValueError(
                    f"{f}:{n}: malformed fixture ({type(e).__name__}: {e}); run canitoolcall validate"
                ) from e
            if families and fx.family not in families:
                continue
            if tags and not set(tags) & set(fx.tags):
                continue
            out.append(fx)
    return out


def load_family(slug: str, root: Path | None = None) -> Family:
    """Load ``<root>/<slug>/family.json``."""
    path = (root or default_fixtures_dir()) / slug / "family.json"
    return Family.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_families(root: Path | None = None) -> dict[str, Family]:
    """Load every ``family.json`` under the fixtures root, keyed by slug."""
    base = root or default_fixtures_dir()
    return {
        p.parent.name: Family.from_dict(json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(base.glob("*/family.json"))
    }


def fixtures_digest(fixtures: Iterable[Fixture]) -> str:
    """sha256 over sorted fixture ids and canonical JSON; recorded in results."""
    h = hashlib.sha256()
    for fx in sorted(fixtures, key=lambda f: f.id):
        h.update(json.dumps(fx.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# --------------------------------------------------------------------------- validation


@dataclass(frozen=True)
class ValidationIssue:
    path: Path
    line: int | None
    message: str

    def __str__(self) -> str:
        loc = f"{self.path}:{self.line}" if self.line is not None else str(self.path)
        return f"{loc}: {self.message}"


def _schema(name: str) -> dict[str, Any]:
    return json.loads((spec_dir() / name).read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def validate(paths: Iterable[Path] | None = None) -> list[ValidationIssue]:
    """Validate fixture files and their family.json against the spec.

    Checks, beyond JSON Schema:
      * ``id`` prefix == ``family`` == parent directory name
      * ids are unique across all files validated together
      * ``fixtures/<family>/family.json`` exists and is valid
      * ``tokenizer`` is present whenever ``output_token_ids`` is
      * every expected tool call names one of the offered tools
      * every tool's ``parameters`` is itself a valid JSON Schema
      * ``models[0]`` is listed in the family's ``reference_models``
      * a ``tokenizer`` pin names that reference model at its pinned revision,
        or its declared ``mirror`` (adapters load tokenizers from the pin, so a
        fixture must not be able to point them at an arbitrary Hub repo)
      * ``raw_output`` does not end with one of that model's stop tokens
      * ``spec_version`` major matches this package

    Returns a list of issues; empty means valid.
    """
    import jsonschema  # lazy: not needed inside engine venvs

    fx_validator = jsonschema.Draft202012Validator(
        _schema("fixture.schema.json"), format_checker=jsonschema.FormatChecker()
    )
    fam_validator = jsonschema.Draft202012Validator(_schema("family.schema.json"))
    issues: list[ValidationIssue] = []
    seen: dict[str, tuple[Path, int]] = {}
    families: dict[Path, Family | None] = {}

    for f in iter_fixture_files(paths):
        fam_file = f.parent / "family.json"
        if fam_file not in families:
            families[fam_file] = None
            if not fam_file.is_file():
                issues.append(ValidationIssue(fam_file, None, "missing family.json"))
            else:
                try:
                    fam = json.loads(fam_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    issues.append(ValidationIssue(fam_file, None, f"invalid JSON: {exc}"))
                    fam = None
                fam_errors = list(fam_validator.iter_errors(fam)) if fam is not None else []
                for e in fam_errors:
                    issues.append(ValidationIssue(fam_file, None, f"{e.json_path}: {e.message}"))
                if fam is not None and not fam_errors:
                    families[fam_file] = Family.from_dict(fam)
                    if fam.get("slug") != f.parent.name:
                        issues.append(ValidationIssue(fam_file, None, "slug must equal the directory name"))
        family = families[fam_file]
        try:
            records = list(read_jsonl(f))
        except ValueError as exc:
            issues.append(ValidationIssue(f, None, str(exc)))
            continue
        for n, rec in records:
            errs = sorted(fx_validator.iter_errors(rec), key=lambda e: list(e.path))
            for e in errs:
                issues.append(ValidationIssue(f, n, f"{e.json_path}: {e.message}"))
            if errs:
                continue
            try:
                Fixture.from_dict(rec)
            except (TypeError, ValueError) as exc:
                issues.append(ValidationIssue(f, n, str(exc)))
                continue
            fid, fam_slug = rec["id"], rec["family"]
            if fid.split("/", 1)[0] != fam_slug:
                issues.append(ValidationIssue(f, n, f"id {fid!r} must start with '{fam_slug}/'"))
            if f.parent.name != fam_slug:
                issues.append(ValidationIssue(f, n, f"family {fam_slug!r} must match directory {f.parent.name!r}"))
            if fid in seen:
                p0, n0 = seen[fid]
                issues.append(ValidationIssue(f, n, f"duplicate id {fid!r} (first at {p0}:{n0})"))
            else:
                seen[fid] = (f, n)
            if "output_token_ids" in rec and "tokenizer" not in rec:
                issues.append(ValidationIssue(f, n, "output_token_ids requires a tokenizer pin"))
            if rec["spec_version"].split(".")[0] != SPEC_VERSION.split(".")[0]:
                issues.append(ValidationIssue(f, n, f"unsupported spec_version {rec['spec_version']}"))
            offered = {t["function"]["name"] for t in rec["tools"]}
            for tc in (rec.get("expected") or {}).get("tool_calls", []):
                if tc["name"] not in offered:
                    issues.append(ValidationIssue(f, n, f"expected call {tc['name']!r} is not an offered tool"))
            for t in rec["tools"]:
                params = t["function"].get("parameters")
                if params is None:
                    continue
                try:
                    jsonschema.validators.validator_for(params, default=jsonschema.Draft202012Validator).check_schema(
                        params
                    )
                except jsonschema.SchemaError as exc:
                    msg = f"tool {t['function']['name']!r}: parameters is not a valid JSON Schema: {exc.message}"
                    issues.append(ValidationIssue(f, n, msg))
            if family is not None:
                ref = family.reference_for(rec["models"][0])
                if ref is None:
                    issues.append(
                        ValidationIssue(f, n, f"models[0] {rec['models'][0]!r} is not a reference_model in family.json")
                    )
                else:
                    pin = rec.get("tokenizer")
                    if pin and not ref.allows_tokenizer(TokenizerPin(pin["repo"], pin["revision"])):
                        msg = (
                            f"tokenizer {pin['repo']}@{pin['revision']} is neither {ref.repo}@{ref.revision} "
                            "nor its mirror in family.json"
                        )
                        issues.append(ValidationIssue(f, n, msg))
                    for stop in ref.stop_tokens:
                        if stop and rec["raw_output"].endswith(stop):
                            msg = f"raw_output must end before the stop token {stop!r}"
                            issues.append(ValidationIssue(f, n, msg))
    return issues

"""Redistribution obligations for third-party licenses quoted by the fixture corpus.

The Llama 3.3 and Llama 4 Community Licenses (clause 1.b.i) require anyone who
distributes Llama Materials to ship a copy of the agreement and to prominently
display "Built with Llama"; clause 1.b.iii requires a Notice line.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from canitoolcall.fixtures import load_fixtures, repo_root

ROOT = repo_root() or Path(__file__).resolve().parents[2]

LLAMA = {
    "LicenseRef-llama3.3-community": (
        "llama3.3-community.txt",
        "Llama 3.3 is licensed under the Llama 3.3 Community License",
    ),
    "LicenseRef-llama4-community": ("llama4-community.txt", "Llama 4 is licensed under the Llama 4 Community License"),
}


def _corpus_licenses() -> set[str]:
    fixtures_dir = ROOT / "fixtures"
    if not fixtures_dir.is_dir():
        pytest.skip("fixture corpus not available")
    return {fx.provenance.license for fx in load_fixtures([fixtures_dir])}


@pytest.mark.parametrize("license_id", sorted(LLAMA))
def test_llama_obligations_met(license_id: str) -> None:
    if license_id not in _corpus_licenses():
        pytest.skip(f"no fixture uses {license_id}")
    agreement, notice = LLAMA[license_id]
    assert (ROOT / "LICENSES" / agreement).is_file()
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert notice in notices
    assert "Built with Llama" in notices
    assert "Built with Llama" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Built with Llama" in (ROOT / "site" / "templates" / "base.html").read_text(encoding="utf-8")


NVIDIA = {
    # Both require a copy of the license with every distribution (NVIDIA Nemotron Open Model
    # License 3.a; OpenMDW-1.1, which also requires keeping the copyright notice).
    "LicenseRef-nvidia-nemotron-open-model-license": "nvidia-nemotron-open-model-license.txt",
    "LicenseRef-OpenMDW-1.1": "openmdw-1.1.txt",
}


@pytest.mark.parametrize("license_id", sorted(NVIDIA))
def test_nvidia_license_copies_shipped(license_id: str) -> None:
    if license_id not in _corpus_licenses():
        pytest.skip(f"no fixture uses {license_id}")
    text = NVIDIA[license_id]
    assert (ROOT / "LICENSES" / text).is_file()
    assert f"`LICENSES/{text}`" in (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

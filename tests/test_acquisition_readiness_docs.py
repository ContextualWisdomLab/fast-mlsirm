"""Protect current documentation from reviving the legacy deal-value gate."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    """Read one repository document as UTF-8 text."""
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_current_release_guides_use_price_neutral_readiness() -> None:
    """Current release guidance must name evidence completeness, not a price."""
    readme = _read("README.md")
    release_guide = _read("docs/release_acceptance.md")
    commercial_guide = _read("docs/commercial_readiness.md")
    enterprise_guide = _read("docs/enterprise_sales_readiness.md")

    assert "--require-acquisition-readiness" in release_guide
    assert "--require-acquisition-readiness" in commercial_guide
    assert "--require-acquisition-readiness" in enterprise_guide
    assert "scripts/build_acquisition_release.py" in readme
    assert "scripts/build_acquisition_release.py" in release_guide
    assert "python scripts/build_commercial_release.py" not in readme
    assert "python scripts/build_commercial_release.py" not in release_guide
    assert (
        "The configured evidence profile is complete and internally consistent"
        in release_guide
    )
    assert "does **not** prove a valuation" in release_guide
    assert "For KRW 2,000,000,000 enterprise sales review" not in release_guide
    assert "The KRW 2,000,000,000 sales-readiness standard" not in commercial_guide
    assert "For the KRW 2,000,000,000 product-readiness standard" not in enterprise_guide


def test_legacy_20b_documents_are_explicitly_compatibility_only() -> None:
    """The retained legacy bundle must not be mistaken for the current gate."""
    legacy_product = _read("docs/20b_product_readiness.md")
    enterprise_guide = _read("docs/enterprise_sales_readiness.md")

    assert legacy_product.startswith("# Legacy 20B compatibility evidence")
    assert "deprecated compatibility" in legacy_product
    assert "--require-20b-product" in legacy_product
    assert "Legacy 20B compatibility" in enterprise_guide
    assert "--require-20b-product" in enterprise_guide


def test_root_navigation_names_the_generic_readiness_contract() -> None:
    """Root navigation must lead buyers to the current neutral readiness gate."""
    readme = _read("README.md")
    governance = _read("docs/GOVERNANCE_INDEX.md")

    assert "Acquisition/commercial readiness" in readme
    assert "20B product readiness gate" not in readme
    assert "--require-20b-product" not in readme
    assert "Acquisition/commercial readiness" in governance
    assert "20B product narrative gates" not in governance

"""Repository contracts for the reviewed Rust compiler baseline."""

from __future__ import annotations

import tomllib
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_TOOLCHAIN = _ROOT / "rust-toolchain.toml"
_CI = _ROOT / ".github" / "workflows" / "ci.yml"
_STUDIES = _ROOT / ".github" / "workflows" / "statistical-studies.yml"
_DEPENDABOT = _ROOT / ".github" / "dependabot.yml"
_ACTION = "dtolnay/rust-toolchain@4be7066ada62dd38de10e7b70166bc74ed198c30"


def _dependabot_ecosystem_block(ecosystem: str) -> str:
    """Return exactly one Dependabot ecosystem block without borrowing sibling fields."""

    dependabot = _DEPENDABOT.read_text(encoding="utf-8")
    marker = f'  - package-ecosystem: "{ecosystem}"\n'
    assert dependabot.count(marker) == 1
    _, remainder = dependabot.split(marker, 1)
    return remainder.split("\n  - package-ecosystem:", 1)[0]


def test_local_rust_toolchain_is_exact_without_raising_public_crate_msrv() -> None:
    """Pin repository builds while leaving each published crate's MSRV unchanged."""

    manifest = tomllib.loads(_TOOLCHAIN.read_text(encoding="utf-8"))
    assert manifest["toolchain"] == {"channel": "1.97.1", "profile": "minimal"}

    for crate_manifest in (
        _ROOT / "crates" / "mlsirm-core" / "Cargo.toml",
        _ROOT / "crates" / "fast-mlsirm-py" / "Cargo.toml",
    ):
        crate = tomllib.loads(crate_manifest.read_text(encoding="utf-8"))
        assert "rust-version" not in crate["package"]


def test_every_product_and_statistical_rust_action_uses_1_97_1() -> None:
    """No Rust-backed verification lane may silently float to a new stable release."""

    expected_counts = ((_CI, 4), (_STUDIES, 4))
    for workflow_path, expected in expected_counts:
        workflow = workflow_path.read_text(encoding="utf-8")
        assert workflow.count(_ACTION) == expected
        assert workflow.count("toolchain: 1.97.1") == expected
        assert "toolchain: stable" not in workflow


def test_stable_compiler_updates_arrive_as_reviewable_pull_requests() -> None:
    """Dependabot tracks the root toolchain manifest with its own bounded settings."""

    block = _dependabot_ecosystem_block("rust-toolchain")
    assert '    directory: "/"' in block
    assert "    schedule:\n      interval: \"weekly\"" in block
    assert "    open-pull-requests-limit: 1" in block

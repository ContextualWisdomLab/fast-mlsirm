"""Repository contracts for the reviewed Rust compiler baseline."""

from __future__ import annotations

import tomllib
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_TOOLCHAIN = _ROOT / "rust-toolchain.toml"
_CI = _ROOT / ".github" / "workflows" / "ci.yml"
_STUDIES = _ROOT / ".github" / "workflows" / "statistical-studies.yml"
_DEPENDABOT = _ROOT / ".github" / "dependabot.yml"
_ACTION_PREFIX = "dtolnay/rust-toolchain@"
_ACTION = "dtolnay/rust-toolchain@4be7066ada62dd38de10e7b70166bc74ed198c30"


def _dependabot_ecosystem_block(ecosystem: str) -> str:
    """Return exactly one Dependabot ecosystem block without borrowing sibling fields."""

    dependabot = _DEPENDABOT.read_text(encoding="utf-8")
    marker = f'  - package-ecosystem: "{ecosystem}"\n'
    assert dependabot.count(marker) == 1
    _, remainder = dependabot.split(marker, 1)
    return remainder.split("\n  - package-ecosystem:", 1)[0]


def _rust_toolchain_steps(workflow: str) -> tuple[tuple[str, str | None], ...]:
    """Return Rust action references paired only with their own ``with.toolchain`` values."""

    lines = workflow.splitlines()
    steps: list[tuple[str, str | None]] = []
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith("- "):
            continue

        step_indent = len(line) - len(stripped)
        step_lines = [stripped[2:]]
        for candidate in lines[index + 1 :]:
            candidate_stripped = candidate.lstrip()
            candidate_indent = len(candidate) - len(candidate_stripped)
            if candidate_stripped and (
                candidate_indent < step_indent
                or (candidate_indent == step_indent and candidate_stripped.startswith("- "))
            ):
                break
            step_lines.append(candidate)

        action: str | None = None
        toolchain: str | None = None
        with_indent = step_indent + 2
        toolchain_indent = step_indent + 4
        inside_with = False
        for offset, candidate in enumerate(step_lines):
            candidate_stripped = candidate.lstrip()
            candidate_indent = (
                step_indent + 2
                if offset == 0
                else len(candidate) - len(candidate_stripped)
            )
            if (
                candidate_indent == with_indent
                and candidate_stripped.startswith("uses:")
            ):
                candidate_action = candidate_stripped.partition(":")[2].strip()
                if candidate_action.startswith(_ACTION_PREFIX):
                    action = candidate_action
                continue
            if candidate_indent == with_indent and candidate_stripped == "with:":
                inside_with = True
                continue
            if candidate_stripped and candidate_indent <= with_indent:
                inside_with = False
            if (
                inside_with
                and candidate_indent == toolchain_indent
                and candidate_stripped.startswith("toolchain:")
            ):
                toolchain = candidate_stripped.partition(":")[2].strip()
        if action is not None:
            steps.append((action, toolchain))
    return tuple(steps)


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

    # ci.yml: python-matrix, focal-gpu-native, rust, gpu-smoke, package,
    # focal-gpu-joint-bootstrap.
    # Statistical Studies의 기존 5개 lane과 별도 opt-in 기준 GPU 측정 lane.
    expected_counts = ((_CI, 6), (_STUDIES, 6))
    for workflow_path, expected in expected_counts:
        workflow = workflow_path.read_text(encoding="utf-8")
        steps = _rust_toolchain_steps(workflow)
        assert len(steps) == expected
        assert all(action == _ACTION for action, _ in steps)
        assert all(toolchain == "1.97.1" for _, toolchain in steps)
        assert "toolchain: stable" not in workflow


def test_rust_toolchain_parser_does_not_borrow_non_with_values() -> None:
    """An unrelated nested ``toolchain`` key cannot satisfy the action input contract."""

    workflow = f"""steps:
  - uses: {_ACTION}
    env:
      toolchain: 1.97.1
  - uses: {_ACTION}
    with:
      toolchain: 1.97.1
"""
    assert _rust_toolchain_steps(workflow) == ((_ACTION, None), (_ACTION, "1.97.1"))


def test_rust_toolchain_parser_finds_action_when_name_precedes_uses() -> None:
    """Step display names cannot hide a Rust action from toolchain enforcement."""

    workflow = f"""steps:
  - name: Install Rust
    uses: {_ACTION}
    with:
      toolchain: 1.97.1
"""
    assert _rust_toolchain_steps(workflow) == ((_ACTION, "1.97.1"),)


def test_stable_compiler_updates_arrive_as_reviewable_pull_requests() -> None:
    """Dependabot tracks the root toolchain manifest with its own bounded settings."""

    block = _dependabot_ecosystem_block("rust-toolchain")
    assert '    directory: "/"' in block
    assert "    schedule:\n      interval: \"weekly\"" in block
    assert "    cooldown:\n      default-days: 7" in block
    assert "    open-pull-requests-limit: 1" in block


# Rust tests measured above 60 s (up to 2,182 s) in a local non-coverage run
# on 2026-09-30; under coverage instrumentation on s1 amd64,
# bifactor_oakes_calibration exceeded 3 h 37 min.
_HEAVY_NUMERIC_RUST_TESTS = (
    ("tests/unit/bifactor_grm_tests.rs", "dense_quadrature_fit_never_claims_tolerance_at_start_slopes"),
    ("tests/unit/two_tier_grm_tests.rs", "arbitrary_quadrature_counts_above_the_old_fixed_table_are_accepted"),
    ("tests/unit/two_tier_grm_tests.rs", "focal_gaussian_recovers_declared_distribution"),
    ("crates/mlsirm-core/tests/bifactor_oakes_calibration.rs", "estimates_stabilize_as_grid_grows_within_supported_cap"),
    ("crates/mlsirm-core/tests/bifactor_oakes_calibration.rs", "se_matches_empirical_sd_over_simulation_replicates"),
    ("crates/mlsirm-core/tests/two_tier_grm_mirt_agreement.rs", "two_tier_grm_agrees_with_mirt_bfactor_two_tier_graded"),
    ("crates/mlsirm-core/tests/two_tier_grm_recovery.rs", "two_tier_grm_recovers_true_parameters_including_primary_correlation"),
    ("crates/mlsirm-core/tests/two_tier_oakes_mirt.rs", "rust_two_tier_oakes_se_matches_mirt_fixture"),
)
_HEAVY_NUMERIC_MARKER = '#[cfg_attr(coverage, ignore = "heavy-numeric:'


def test_heavy_numeric_rust_tests_skip_only_under_coverage() -> None:
    """Coverage runs skip heavy numeric tests; the plain ``rust`` job still runs them.

    cargo-llvm-cov enables ``cfg(coverage)`` (its README, ``--no-cfg-coverage``),
    so the marker removes these tests from instrumented runs only. Review
    sandboxes key on the ``heavy-numeric:`` reason text.
    """

    for relative, name in _HEAVY_NUMERIC_RUST_TESTS:
        lines = (_ROOT / relative).read_text(encoding="utf-8").splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith(f"fn {name}("))
        attributes = lines[max(0, index - 3) : index]
        assert any(line.startswith(_HEAVY_NUMERIC_MARKER) for line in attributes), (relative, name)
        assert not any(line.strip() == "#[ignore]" for line in attributes), (relative, name)
    rust_job = _CI.read_text(encoding="utf-8").split("\n  rust:\n", 1)[1].split("\n  gpu-smoke:\n", 1)[0]
    assert "cargo test --workspace" in rust_job
    assert "llvm-cov" not in rust_job

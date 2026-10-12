"""Source-wiring checks for L3 GPU-only manual evidence, without a native build.

These checks cover conditional-compilation and command wiring, not Rust type
checking, device execution, numerical parity, or scientific acceptance.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
GPU_ONLY_CFG = '#[cfg(all(feature = "gpu", not(coverage)))]'


@pytest.mark.parametrize(
    "function",
    [
        "measure_fixture",
        "max_count_abs_diff",
        "measure_concurrent_split_estep_vs_cpu_reference",
    ],
)
def test_manual_gpu_evidence_is_excluded_from_cpu_and_coverage_builds(
    function: str,
) -> None:
    """Match the GPU-only symbol availability condition at each evidence helper."""
    source = (ROOT / "tests/unit/bifactor_estep_split_tests.rs").read_text()
    match = re.search(
        rf"(?m)((?:(?:#\[[^\n]+\]|///[^\n]*)\n)*)fn {function}\(",
        source,
    )
    assert match is not None, f"evidence function {function} is missing"
    assert GPU_ONLY_CFG in match.group(1), (
        f"{function} must not compile when GPU symbols are unavailable; "
        "#[ignore] alone only prevents execution"
    )


def test_direct_rust_config_uses_the_existing_split_boundary_admission() -> None:
    """Require common validation before either GPU or CPU fallback execution."""
    source = (ROOT / "crates/mlsirm-core/src/bifactor_grm.rs").read_text()
    validation = source.split("pub(crate) fn validate(", 1)[1].split(
        "    let n_cells =", 1
    )[0]
    assert "if let BifactorDevice::Split { gpu_person_start } = cfg.device" in validation
    assert (
        'parse_bifactor_device("split", n_persons, Some(gpu_person_start))?;'
        in validation
    ), "direct Rust configs must use the same rejection rule as parsed configs"


def test_split_executor_does_not_silently_clamp_the_admitted_boundary() -> None:
    """Transport the validated caller boundary without a replacement value."""
    source = (ROOT / "crates/mlsirm-core/src/bifactor_grm.rs").read_text()
    executor = source.split("fn e_step_same_host_split(", 1)[1].split(
        "    let tables_wrapped", 1
    )[0]
    assert "let split = gpu_person_start;" in executor
    assert ".min(" not in executor and ".max(" not in executor


def test_documented_manual_command_selects_the_ignored_measurement() -> None:
    """Keep the present reproduction command aligned with the evidence script."""
    source = (
        ROOT / "docs/orchestration/bifactor-estep-split-overlap-2043-environment.txt"
    ).read_text()
    commands = [line for line in source.splitlines() if line.startswith("cargo test ")]
    assert len(commands) == 1
    assert "-- --ignored --nocapture" in commands[0], (
        "the direct invocation must execute the ignored measurement"
    )

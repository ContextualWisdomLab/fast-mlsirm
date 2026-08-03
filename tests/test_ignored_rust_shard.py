"""Tests for deterministic target-qualified ignored Rust test sharding."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_ignored_rust_shard.py"
_SPEC = importlib.util.spec_from_file_location("run_ignored_rust_shard", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
sharder = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = sharder
_SPEC.loader.exec_module(sharder)


def _metadata() -> str:
    """Return a compact workspace with duplicate test names across targets."""
    return json.dumps(
        {
            "workspace_members": [
                "path+file:///repo/crates/mlsirm-core#0.9.0",
                "path+file:///repo/crates/fast-mlsirm-py#0.9.0",
            ],
            "packages": [
                {
                    "id": "path+file:///repo/crates/mlsirm-core#0.9.0",
                    "name": "mlsirm-core",
                    "targets": [
                        {
                            "name": "mlsirm_core",
                            "kind": ["lib"],
                            "test": True,
                            "doctest": True,
                        },
                        {
                            "name": "literature_true_parameter_recovery",
                            "kind": ["test"],
                            "test": True,
                            "doctest": False,
                        },
                        {
                            "name": "helper",
                            "kind": ["bin"],
                            "test": False,
                            "doctest": False,
                        },
                    ],
                },
                {
                    "id": "path+file:///repo/crates/fast-mlsirm-py#0.9.0",
                    "name": "fast-mlsirm-py",
                    "targets": [
                        {
                            "name": "fast_mlsirm_py",
                            "kind": ["lib"],
                            "test": True,
                            "doctest": False,
                        }
                    ],
                },
            ],
        }
    )


def test_sharding_runner_contains_no_source_mutation_contract():
    """The ignored-test runner cannot rewrite statistical source before testing."""
    source = _SCRIPT.read_text(encoding="utf-8")
    assert "write_text(" not in source
    assert "apply_statistical_contract_overlay" not in source
    assert "_CDM_TEST_PATH" not in source


def test_parse_test_names_filters_noise_and_deduplicates_within_one_target():
    """Only one target's lines ending in ``: test`` become test names."""
    output = """
warning: ignored
alpha::one: test
beta: benchmark
alpha::one: test
zeta::two: test
2 tests, 0 benchmarks
"""
    assert sharder.parse_test_names(output) == ["alpha::one", "zeta::two"]


def test_workspace_targets_are_explicit_and_separately_governed_packages_are_excluded():
    """Cargo metadata becomes exact target selectors without PyO3 duplication."""
    targets = sharder.parse_workspace_targets(
        _metadata(),
        excluded_packages=["fast-mlsirm-py"],
    )
    assert targets == [
        sharder.CargoTarget("mlsirm-core", "doc", "mlsirm_core"),
        sharder.CargoTarget("mlsirm-core", "lib", "mlsirm_core"),
        sharder.CargoTarget(
            "mlsirm-core",
            "test",
            "literature_true_parameter_recovery",
        ),
    ]


def test_workspace_target_validation_rejects_stale_or_duplicate_exclusions():
    """Package exclusions must be unique exact workspace package names."""
    with pytest.raises(ValueError, match="must be unique"):
        sharder.parse_workspace_targets(
            _metadata(),
            excluded_packages=["fast-mlsirm-py", "fast-mlsirm-py"],
        )
    with pytest.raises(ValueError, match="missing-package"):
        sharder.parse_workspace_targets(
            _metadata(),
            excluded_packages=["missing-package"],
        )


def test_target_qualified_identifiers_preserve_same_named_tests():
    """Same-named tests in different targets remain distinct evidence entries."""
    lib_target = sharder.CargoTarget("mlsirm-core", "lib", "mlsirm_core")
    integration_target = sharder.CargoTarget(
        "mlsirm-core",
        "test",
        "literature_true_parameter_recovery",
    )
    left = f"{lib_target.identity}::shared_name"
    right = f"{integration_target.identity}::shared_name"
    assert left != right
    shards = sharder.partition_inventory([left, right], 1)
    assert set(shards[0]) == {left, right}


def test_round_robin_shards_are_exhaustive_disjoint_and_nonempty():
    """Every retained target-qualified test belongs to one stable shard."""
    names = [f"core/test/target_{index:02d}::test" for index in range(17)]
    shards = [set(values) for values in sharder.partition_inventory(names, 4)]
    union = set().union(*shards)
    assert union == set(names)
    assert all(shards)
    for left in range(len(shards)):
        for right in range(left + 1, len(shards)):
            assert shards[left].isdisjoint(shards[right])


def test_skip_requires_an_exact_target_qualified_inventory_identifier():
    """A function-only name cannot exclude same-named tests across targets."""
    names = [
        "core/lib/core::dedicated",
        "core/test/recovery::dedicated",
        "core/test/recovery::keep",
    ]
    with pytest.raises(ValueError, match="target-qualified identifier"):
        sharder.select_shard(names, 0, 1, ["dedicated"])
    selected = sharder.select_shard(
        names,
        0,
        1,
        ["core/lib/core::dedicated"],
    )
    assert selected == [
        "core/test/recovery::dedicated",
        "core/test/recovery::keep",
    ]


def test_skip_names_are_unique_and_each_matches_one_inventory_entry():
    """Duplicate or stale dedicated-test declarations fail before execution."""
    names = ["core/lib/core::one", "core/lib/core::two"]
    with pytest.raises(ValueError, match="must be unique"):
        sharder.validated_skip_set(
            names,
            ["core/lib/core::one", "core/lib/core::one"],
        )
    with pytest.raises(ValueError, match="core/lib/core::missing"):
        sharder.validated_skip_set(names, ["core/lib/core::missing"])
    assert sharder.validated_skip_set(
        names,
        ["core/lib/core::one"],
    ) == {"core/lib/core::one"}


def test_partition_rejects_duplicate_identifiers_and_silently_empty_shards():
    """Collapsed identities and matrix drift fail before any test execution."""
    with pytest.raises(ValueError, match="must be unique"):
        sharder.partition_inventory(["core/lib/core::one"] * 2, 1)
    with pytest.raises(ValueError, match="empty shards: 2, 3"):
        sharder.partition_inventory(["a", "b"], 4)


@pytest.mark.parametrize(
    ("shard", "count", "message"),
    [
        (0, 0, "at least 1"),
        (-1, 2, "0 <= shard"),
        (2, 2, "0 <= shard"),
    ],
)
def test_invalid_partition_parameters_fail(shard, count, message):
    """Invalid shard coordinates are rejected before any process execution."""
    with pytest.raises(ValueError, match=message):
        sharder.select_shard(["test"], shard, count)


def test_cargo_commands_are_target_exact_and_injection_safe():
    """Commands select one Cargo target and pass names as argument-vector fields."""
    target = sharder.CargoTarget(
        "mlsirm-core",
        "test",
        "literature_true_parameter_recovery",
    )
    assert sharder.cargo_metadata_command() == [
        "cargo",
        "metadata",
        "--no-deps",
        "--format-version",
        "1",
    ]
    assert sharder.cargo_list_command(target) == [
        "cargo",
        "test",
        "--release",
        "-p",
        "mlsirm-core",
        "--test",
        "literature_true_parameter_recovery",
        "--",
        "--ignored",
        "--list",
    ]
    name = "module::test name; echo unsafe"
    command = sharder.cargo_test_command(target, name)
    assert command[7] == name
    assert command[-4:] == [
        "--ignored",
        "--exact",
        "--nocapture",
        "--test-threads=1",
    ]


def test_library_and_doctest_commands_use_unambiguous_single_flag_selectors():
    """Library and doctest inventories never rely on flattened workspace output."""
    lib_target = sharder.CargoTarget("mlsirm-core", "lib", "mlsirm_core")
    doc_target = sharder.CargoTarget("mlsirm-core", "doc", "mlsirm_core")
    assert sharder.cargo_list_command(lib_target)[5:7] == ["--lib", "--"]
    assert sharder.cargo_list_command(doc_target)[5:7] == ["--doc", "--"]

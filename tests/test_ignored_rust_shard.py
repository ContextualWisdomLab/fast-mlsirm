"""Tests for deterministic ignored Rust test sharding."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_ignored_rust_shard.py"
_SPEC = importlib.util.spec_from_file_location("run_ignored_rust_shard", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
sharder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sharder)


def test_sharding_runner_contains_no_source_mutation_contract():
    """The ignored-test runner cannot rewrite statistical source before testing."""
    source = _SCRIPT.read_text(encoding="utf-8")
    assert "write_text(" not in source
    assert "apply_statistical_contract_overlay" not in source
    assert "_CDM_TEST_PATH" not in source


def test_parse_test_names_filters_noise_and_deduplicates():
    """Only Cargo lines ending in ``: test`` become inventory entries."""
    output = """
warning: ignored
alpha::one: test
beta: benchmark
alpha::one: test
zeta::two: test
2 tests, 0 benchmarks
"""
    assert sharder.parse_test_names(output) == ["alpha::one", "zeta::two"]


def test_round_robin_shards_are_exhaustive_disjoint_and_nonempty():
    """Every retained test belongs to exactly one stable nonempty shard."""
    names = [f"module::test_{index:02d}" for index in range(17)]
    shards = [set(values) for values in sharder.partition_inventory(names, 4)]
    union = set().union(*shards)
    assert union == set(names)
    assert all(shards)
    for left in range(len(shards)):
        for right in range(left + 1, len(shards)):
            assert shards[left].isdisjoint(shards[right])


def test_skip_requires_an_exact_fully_qualified_inventory_name():
    """A final path component cannot accidentally exclude multiple tests."""
    names = ["a::dedicated", "b::dedicated", "b::keep"]
    with pytest.raises(ValueError, match="not found by exact name: dedicated"):
        sharder.select_shard(names, 0, 1, ["dedicated"])
    selected = sharder.select_shard(names, 0, 1, ["a::dedicated"])
    assert selected == ["b::dedicated", "b::keep"]


def test_skip_names_are_unique_and_each_matches_one_inventory_entry():
    """Duplicate or stale dedicated-test declarations fail before execution."""
    names = ["module::one", "module::two"]
    with pytest.raises(ValueError, match="must be unique"):
        sharder.validated_skip_set(names, ["module::one", "module::one"])
    with pytest.raises(ValueError, match="module::missing"):
        sharder.validated_skip_set(names, ["module::missing"])
    assert sharder.validated_skip_set(names, ["module::one"]) == {"module::one"}


def test_partition_rejects_silently_empty_shards():
    """Matrix drift cannot emit a green shard with no evidentiary test."""
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


def test_cargo_commands_are_exact_and_injection_safe():
    """Commands are argument vectors rather than shell-expanded strings."""
    assert sharder.cargo_list_command() == [
        "cargo",
        "test",
        "--release",
        "--workspace",
        "--",
        "--ignored",
        "--list",
    ]
    name = "module::test name; echo unsafe"
    command = sharder.cargo_test_command(name)
    assert command[4] == name
    assert command[-4:] == ["--ignored", "--exact", "--nocapture", "--test-threads=1"]

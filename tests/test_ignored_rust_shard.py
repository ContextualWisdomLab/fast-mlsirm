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


def test_round_robin_shards_are_exhaustive_and_disjoint():
    """Every retained test belongs to exactly one stable shard."""
    names = [f"module::test_{index:02d}" for index in range(17)]
    shards = [set(sharder.select_shard(names, index, 4)) for index in range(4)]
    union = set().union(*shards)
    assert union == set(names)
    for left in range(len(shards)):
        for right in range(left + 1, len(shards)):
            assert shards[left].isdisjoint(shards[right])


def test_skip_matches_exact_or_final_component():
    """Dedicated tests can be excluded regardless of module qualification."""
    names = ["a::keep", "a::dedicated", "dedicated", "b::keep"]
    selected = sharder.select_shard(names, 0, 1, ["dedicated"])
    assert selected == ["a::keep", "b::keep"]


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

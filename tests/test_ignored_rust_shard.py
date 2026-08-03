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


def _overlay_fixture(tmp_path: Path, block: str) -> Path:
    """Create the reviewed CDM test path around one assertion block."""
    path = tmp_path / sharder._CDM_TEST_PATH
    path.parent.mkdir(parents=True)
    path.write_text(f"prefix\n{block}\nsuffix\n", encoding="utf-8")
    return path


def test_statistical_contract_overlay_replaces_exact_assertion(tmp_path):
    """The CI overlay compiles the Monte Carlo-aware formula in Rust."""
    path = _overlay_fixture(tmp_path, sharder._CDM_EXACT_THRESHOLD)
    returned = sharder.apply_statistical_contract_overlay(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert returned == path
    assert sharder._CDM_EXACT_THRESHOLD not in text
    assert sharder._CDM_MONTE_CARLO_THRESHOLD in text
    assert "let mc_se" in text


def test_statistical_contract_overlay_is_idempotent(tmp_path):
    """A previously overlaid checkout is accepted without a second mutation."""
    path = _overlay_fixture(tmp_path, sharder._CDM_MONTE_CARLO_THRESHOLD)
    before = path.read_text(encoding="utf-8")
    sharder.apply_statistical_contract_overlay(tmp_path)
    assert path.read_text(encoding="utf-8") == before


def test_statistical_contract_overlay_fails_on_unreviewed_source(tmp_path):
    """Source drift cannot cause a broad or accidental text replacement."""
    _overlay_fixture(tmp_path, "let conv_rate = unknown();")
    with pytest.raises(RuntimeError, match="did not match"):
        sharder.apply_statistical_contract_overlay(tmp_path)


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

#!/usr/bin/env python3
"""List and execute one deterministic shard of ignored Rust tests.

Cargo's test harness can list ignored tests but does not natively partition them
across GitHub Actions runners. This helper sorts exact test names, validates that
every dedicated-test exclusion names exactly one inventoried test, assigns test
index ``i`` to shard ``i % shard_count``, and invokes selected tests with
``--ignored --exact``. The partition is exhaustive and non-overlapping for a
fixed repository revision. The runner is deliberately read-only: it never
rewrites source or test files to make a failing statistical contract pass.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence

_TEST_LINE = re.compile(r"^(?P<name>.+): test$")


def parse_test_names(output: str) -> list[str]:
    """Return sorted unique Rust test names from ``cargo test -- --list`` output."""
    names = {
        match.group("name")
        for line in output.splitlines()
        if (match := _TEST_LINE.match(line.strip())) is not None
    }
    return sorted(names)


def validated_skip_set(names: Sequence[str], skipped: Iterable[str]) -> frozenset[str]:
    """Validate unique, exact dedicated-test names against the inventory."""
    inventory = set(names)
    requested = tuple(skipped)
    if len(set(requested)) != len(requested):
        raise ValueError("skipped test names must be unique")
    missing = sorted(set(requested) - inventory)
    if missing:
        raise ValueError(
            "skipped tests were not found by exact name: " + ", ".join(missing)
        )
    return frozenset(requested)


def partition_inventory(
    names: Sequence[str],
    shard_count: int,
    skipped: Iterable[str] = (),
) -> tuple[tuple[str, ...], ...]:
    """Return an exhaustive disjoint partition with no silently empty shard."""
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    inventory = sorted(set(names))
    skip_set = validated_skip_set(inventory, skipped)
    retained = [name for name in inventory if name not in skip_set]
    shards = tuple(
        tuple(name for index, name in enumerate(retained) if index % shard_count == shard)
        for shard in range(shard_count)
    )
    empty = [str(index) for index, selected in enumerate(shards) if not selected]
    if empty:
        raise ValueError(
            "ignored Rust partition contains empty shards: " + ", ".join(empty)
        )
    return shards


def select_shard(
    names: Sequence[str],
    shard: int,
    shard_count: int,
    skipped: Iterable[str] = (),
) -> list[str]:
    """Select one stable shard after exact dedicated-test exclusions."""
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if not 0 <= shard < shard_count:
        raise ValueError("shard must satisfy 0 <= shard < shard_count")
    return list(partition_inventory(names, shard_count, skipped)[shard])


def cargo_list_command() -> list[str]:
    """Return the Cargo command that inventories workspace ignored tests."""
    return ["cargo", "test", "--release", "--workspace", "--", "--ignored", "--list"]


def cargo_test_command(name: str) -> list[str]:
    """Return the exact Cargo command for one ignored workspace test."""
    return [
        "cargo",
        "test",
        "--release",
        "--workspace",
        name,
        "--",
        "--ignored",
        "--exact",
        "--nocapture",
        "--test-threads=1",
    ]


def run_shard(shard: int, shard_count: int, skipped: Sequence[str]) -> int:
    """Run one deterministic ignored-test shard and return its process code."""
    inventory = subprocess.run(
        cargo_list_command(),
        check=True,
        capture_output=True,
        text=True,
    )
    names = parse_test_names(inventory.stdout)
    if not names:
        raise RuntimeError("Cargo did not report any ignored workspace tests")
    selected = select_shard(names, shard, shard_count, skipped)
    print(
        f"ignored Rust shard {shard + 1}/{shard_count}: "
        f"{len(selected)} of {len(names)} listed tests",
        flush=True,
    )
    for name in selected:
        command = cargo_test_command(name)
        print("+ " + " ".join(command), flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the sharding runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, required=True, help="zero-based shard index")
    parser.add_argument("--shards", type=int, required=True, help="total shard count")
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="exact fully qualified test name executed by a dedicated job",
    )
    args = parser.parse_args(argv)
    if args.shards < 1:
        parser.error("--shards must be at least 1")
    if not 0 <= args.shard < args.shards:
        parser.error("--shard must satisfy 0 <= shard < shards")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the requested ignored-test shard."""
    args = parse_args(argv)
    return run_shard(args.shard, args.shards, args.skip)


if __name__ == "__main__":
    sys.exit(main())

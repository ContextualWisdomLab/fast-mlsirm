#!/usr/bin/env python3
"""List and execute one deterministic shard of ignored Rust tests.

Cargo's test harness lists test names only within one compiled target. Flattening
``cargo test --workspace -- --list`` output loses that target identity, so two
same-named tests in different crates or integration binaries can be collapsed or
executed twice. This helper inventories each workspace target independently,
qualifies every test as ``package/selector/target::test_name``, validates exact
dedicated-test exclusions, and invokes the matching Cargo target with
``--ignored --exact``.

The runner is source-read-only and deliberately excludes separately governed
packages from the general shard inventory.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

_TEST_LINE = re.compile(r"^(?P<name>.+): test$")
_SUPPORTED_SELECTORS = frozenset({"lib", "bin", "test", "example", "doc"})


@dataclass(frozen=True, order=True)
class CargoTarget:
    """One Cargo test-bearing target with a stable audit identity."""

    package: str
    selector: str
    target_name: str

    def __post_init__(self) -> None:
        """Reject malformed target metadata before constructing commands."""
        if not self.package or not self.target_name:
            raise ValueError("Cargo target package and target_name must be non-empty")
        if self.selector not in _SUPPORTED_SELECTORS:
            raise ValueError(f"unsupported Cargo selector: {self.selector}")

    @property
    def identity(self) -> str:
        """Return the stable package/target prefix used by test identifiers."""
        return f"{self.package}/{self.selector}/{self.target_name}"


@dataclass(frozen=True, order=True)
class IgnoredTest:
    """One ignored Rust test tied to exactly one Cargo target."""

    identifier: str
    target: CargoTarget
    test_name: str


def parse_test_names(output: str) -> list[str]:
    """Return sorted unique Rust test names from one target's list output."""
    names = {
        match.group("name")
        for line in output.splitlines()
        if (match := _TEST_LINE.match(line.strip())) is not None
    }
    return sorted(names)


def cargo_metadata_command() -> list[str]:
    """Return the read-only command used to enumerate workspace targets."""
    return ["cargo", "metadata", "--no-deps", "--format-version", "1"]


def _selector_for_target(kinds: set[str]) -> str | None:
    """Map Cargo metadata target kinds to an explicit ``cargo test`` selector."""
    if "lib" in kinds or "proc-macro" in kinds:
        return "lib"
    for selector in ("bin", "test", "example"):
        if selector in kinds:
            return selector
    return None


def parse_workspace_targets(
    metadata_output: str,
    excluded_packages: Iterable[str] = (),
) -> list[CargoTarget]:
    """Return exact test-bearing targets for workspace packages not excluded."""
    metadata = json.loads(metadata_output)
    packages = metadata.get("packages")
    workspace_members = set(metadata.get("workspace_members", ()))
    if not isinstance(packages, list) or not workspace_members:
        raise ValueError("Cargo metadata did not contain a workspace package inventory")

    requested_exclusions = tuple(excluded_packages)
    if len(set(requested_exclusions)) != len(requested_exclusions):
        raise ValueError("excluded package names must be unique")

    workspace_packages = {
        package["name"]: package
        for package in packages
        if package.get("id") in workspace_members
    }
    missing_exclusions = sorted(set(requested_exclusions) - set(workspace_packages))
    if missing_exclusions:
        raise ValueError(
            "excluded packages were not found in the workspace: "
            + ", ".join(missing_exclusions)
        )

    targets: set[CargoTarget] = set()
    for package_name, package in workspace_packages.items():
        if package_name in requested_exclusions:
            continue
        for target in package.get("targets", ()):
            if not target.get("test", False):
                continue
            target_name = target.get("name")
            kinds = set(target.get("kind", ()))
            if not isinstance(target_name, str):
                raise ValueError(f"Cargo target in {package_name} has no valid name")
            selector = _selector_for_target(kinds)
            if selector is None:
                continue
            targets.add(CargoTarget(package_name, selector, target_name))
            if selector == "lib" and target.get("doctest", False):
                targets.add(CargoTarget(package_name, "doc", target_name))
    if not targets:
        raise ValueError("Cargo metadata did not expose any test-bearing targets")
    return sorted(targets)


def _target_selector_arguments(target: CargoTarget) -> list[str]:
    """Return the exact Cargo selector arguments for one target."""
    if target.selector in {"lib", "doc"}:
        return [f"--{target.selector}"]
    return [f"--{target.selector}", target.target_name]


def cargo_list_command(target: CargoTarget) -> list[str]:
    """Return the command that lists ignored tests for exactly one target."""
    return [
        "cargo",
        "test",
        "--release",
        "-p",
        target.package,
        *_target_selector_arguments(target),
        "--",
        "--ignored",
        "--list",
    ]


def cargo_test_command(target: CargoTarget, test_name: str) -> list[str]:
    """Return the exact command for one ignored test in one Cargo target."""
    return [
        "cargo",
        "test",
        "--release",
        "-p",
        target.package,
        *_target_selector_arguments(target),
        test_name,
        "--",
        "--ignored",
        "--exact",
        "--nocapture",
        "--test-threads=1",
    ]


def inventory_ignored_tests(targets: Sequence[CargoTarget]) -> list[IgnoredTest]:
    """List ignored tests target-by-target without collapsing duplicate names."""
    inventory: list[IgnoredTest] = []
    seen: set[str] = set()
    for target in targets:
        completed = subprocess.run(
            cargo_list_command(target),
            check=True,
            capture_output=True,
            text=True,
        )
        for test_name in parse_test_names(completed.stdout):
            identifier = f"{target.identity}::{test_name}"
            if identifier in seen:
                raise ValueError(f"duplicate target-qualified test: {identifier}")
            seen.add(identifier)
            inventory.append(IgnoredTest(identifier, target, test_name))
    return sorted(inventory)


def validated_skip_set(
    identifiers: Sequence[str],
    skipped: Iterable[str],
) -> frozenset[str]:
    """Validate unique exact dedicated-test identifiers against the inventory."""
    inventory = set(identifiers)
    requested = tuple(skipped)
    if len(set(requested)) != len(requested):
        raise ValueError("skipped test identifiers must be unique")
    missing = sorted(set(requested) - inventory)
    if missing:
        raise ValueError(
            "skipped tests were not found by exact target-qualified identifier: "
            + ", ".join(missing)
        )
    return frozenset(requested)


def partition_inventory(
    identifiers: Sequence[str],
    shard_count: int,
    skipped: Iterable[str] = (),
) -> tuple[tuple[str, ...], ...]:
    """Return an exhaustive disjoint partition with no silently empty shard."""
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    inventory = sorted(set(identifiers))
    if len(inventory) != len(identifiers):
        raise ValueError("target-qualified ignored-test identifiers must be unique")
    skip_set = validated_skip_set(inventory, skipped)
    retained = [identifier for identifier in inventory if identifier not in skip_set]
    shards = tuple(
        tuple(
            identifier
            for index, identifier in enumerate(retained)
            if index % shard_count == shard
        )
        for shard in range(shard_count)
    )
    empty = [str(index) for index, selected in enumerate(shards) if not selected]
    if empty:
        raise ValueError(
            "ignored Rust partition contains empty shards: " + ", ".join(empty)
        )
    return shards


def select_shard(
    identifiers: Sequence[str],
    shard: int,
    shard_count: int,
    skipped: Iterable[str] = (),
) -> list[str]:
    """Select one stable shard after exact dedicated-test exclusions."""
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if not 0 <= shard < shard_count:
        raise ValueError("shard must satisfy 0 <= shard < shard_count")
    return list(partition_inventory(identifiers, shard_count, skipped)[shard])


def run_shard(
    shard: int,
    shard_count: int,
    skipped: Sequence[str],
    excluded_packages: Sequence[str],
) -> int:
    """Run one target-qualified ignored-test shard and return its process code."""
    metadata = subprocess.run(
        cargo_metadata_command(),
        check=True,
        capture_output=True,
        text=True,
    )
    targets = parse_workspace_targets(metadata.stdout, excluded_packages)
    inventory = inventory_ignored_tests(targets)
    if not inventory:
        raise RuntimeError("Cargo did not report any ignored workspace tests")

    by_identifier = {test.identifier: test for test in inventory}
    selected = select_shard(
        tuple(by_identifier),
        shard,
        shard_count,
        skipped,
    )
    print(
        f"ignored Rust shard {shard + 1}/{shard_count}: "
        f"{len(selected)} of {len(inventory)} target-qualified tests",
        flush=True,
    )
    for identifier in selected:
        test = by_identifier[identifier]
        command = cargo_test_command(test.target, test.test_name)
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
        help=(
            "exact package/selector/target::test_name identifier executed by "
            "a dedicated job"
        ),
    )
    parser.add_argument(
        "--exclude-package",
        action="append",
        default=[],
        help="exact workspace package governed by a separate evidence job",
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
    return run_shard(
        args.shard,
        args.shards,
        args.skip,
        args.exclude_package,
    )


if __name__ == "__main__":
    sys.exit(main())

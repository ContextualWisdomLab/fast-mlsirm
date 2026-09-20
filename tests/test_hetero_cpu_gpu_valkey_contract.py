# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Independent contract tests for unmet hetero CPU+GPU × Valkey gaps.

Locks the READ-ONLY map in ``HETERO_CPU_GPU_VALKEY_CONTRACT.md`` (U1/U3):

* Valkey distribute path for ``fipc`` / ``fipc_group`` / ``two_tier`` is
  absent in this worktree (no remote stack / no distribute API).
* CPU+GPU concurrent cohort is absent: poly FIPC and two-tier fit APIs have
  no device axis; ``float_path`` is not a fit device selector here.

Filesystem / AST only — no maturin, no live Valkey, no Mac cargo.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parents[1] / "python" / "fast_mlsirm"

_FORBIDDEN_DEVICE_PARAMS = frozenset(
    {
        "rust_device",
        "device",
        "gpu",
        "float_path",
        "requested_device",
        "effective_device",
        "concurrent_devices",
        "cpu_gpu",
    }
)

_VALKEY_DISTRIBUTE_NEEDLES = (
    "ValkeyStreamsOutcomeStore",
    "distribute_fipc",
    "distribute_two_tier",
    "distribute_fipc_group",
    "FIPC_GROUP_PERSON_SCORE",
    "RemoteJobFamily.FIPC",
    "RemoteJobFamily.TWO_TIER",
    "execute_fipc_group_person_score",
    "execute_two_tier",
)


def _function_param_names(path: Path, func_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            names = [arg.arg for arg in node.args.args]
            names.extend(arg.arg for arg in node.args.kwonlyargs)
            if node.args.vararg is not None:
                names.append(node.args.vararg.arg)
            if node.args.kwarg is not None:
                names.append(node.args.kwarg.arg)
            return set(names)
    raise AssertionError(f"{func_name!r} not found in {path}")


def _dataclass_field_names(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            fields: set[str] = set()
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields.add(stmt.target.id)
            return fields
    raise AssertionError(f"{class_name!r} not found in {path}")


def test_valkey_remote_stack_absent_for_fipc_and_two_tier_surfaces() -> None:
    """U1: this WT has no Valkey/remote worker modules to distribute those families."""
    assert not (_PKG / "remote_exec.py").is_file()
    assert not (_PKG / "remote_worker.py").is_file()
    assert not (_PKG / "fipc_group_score.py").is_file()


def test_package_source_has_no_valkey_distribute_api_for_fipc_two_tier() -> None:
    """U1: no ValkeyStreams / distribute_* / remote family wiring in shipped .py."""
    corpus = []
    for path in sorted(_PKG.glob("*.py")):
        corpus.append(path.read_text(encoding="utf-8", errors="replace"))
    joined = "\n".join(corpus)
    missing = [needle for needle in _VALKEY_DISTRIBUTE_NEEDLES if needle not in joined]
    # All needles must be absent — presence would mean a distribute path landed.
    assert missing == list(_VALKEY_DISTRIBUTE_NEEDLES), (
        "Valkey/FIPC/two-tier distribute surface unexpectedly present: "
        f"found {[n for n in _VALKEY_DISTRIBUTE_NEEDLES if n not in missing]}"
    )


@pytest.mark.parametrize(
    ("relpath", "func_name"),
    [
        ("polytomous.py", "fit_poly_fipc"),
        ("two_tier_grm.py", "fit_two_tier_grm"),
    ],
)
def test_fipc_and_two_tier_fit_apis_have_no_device_axis(
    relpath: str, func_name: str
) -> None:
    """U3: poly FIPC / two-tier fits are not dual-device; no rust_device/float_path."""
    params = _function_param_names(_PKG / relpath, func_name)
    leaked = sorted(params & _FORBIDDEN_DEVICE_PARAMS)
    assert leaked == [], f"{func_name} unexpectedly accepts {leaked}"


def test_fit_config_has_single_rust_device_not_concurrent_cohort_api() -> None:
    """U3: FitConfig may expose one rust_device axis for MLSIRM fit only — not concurrent CPU+GPU cohort."""
    fields = _dataclass_field_names(_PKG / "config.py", "FitConfig")
    assert "rust_device" in fields
    for forbidden in (
        "float_path",
        "concurrent_devices",
        "cpu_device",
        "gpu_device",
        "dual_device",
        "device_cohort",
    ):
        assert forbidden not in fields, f"FitConfig must not define concurrent field {forbidden}"


def test_float_path_is_not_equated_to_device_in_this_tree() -> None:
    """U3: ExecutionFloatPath / float_path remote cohort tag is absent here — cannot select GPU."""
    assert not (_PKG / "remote_exec.py").is_file()
    config_src = (_PKG / "config.py").read_text(encoding="utf-8")
    assert "float_path" not in config_src
    assert "ExecutionFloatPath" not in config_src

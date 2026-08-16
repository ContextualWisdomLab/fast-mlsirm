"""Regression tests for automatic release-evidence backend authority."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_sales_readiness():
    script = Path(__file__).resolve().parents[1] / "scripts" / "sales_readiness.py"
    spec = importlib.util.spec_from_file_location("sales_readiness", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _artifact(path: Path, payload: object | None = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if payload is None:
        path.write_text("ok", encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _acceptance_summary(
    tmp_path: Path,
    *,
    auto_backend: str,
    persisted_auto_backend: str,
) -> Path:
    artifacts = tmp_path / "artifacts"
    auto_summary = _artifact(
        artifacts / "fit_auto" / "fit_summary.json",
        {"backend": persisted_auto_backend},
    )
    rust_summary = _artifact(
        artifacts / "fit_rust" / "fit_summary.json",
        {"backend": "rust"},
    )
    payload = {
        "status": "ok",
        "total_duration_seconds": 0.25,
        "steps": [
            {
                "command": "simulate",
                "files": {"responses": _artifact(artifacts / "simulate" / "responses.npy")},
            },
            {
                "command": "fit",
                "backend": auto_backend,
                "out": str(artifacts / "fit_auto"),
                "files": {"summary": auto_summary},
            },
            {
                "command": "fit",
                "backend": "rust",
                "out": str(artifacts / "fit_rust"),
                "files": {"summary": rust_summary},
            },
            {
                "command": "diagnose-fit",
                "files": {"diagnostics": _artifact(artifacts / "diagnostics_fit.json")},
            },
            {
                "command": "diagnose-dimensions",
                "files": {"diagnostics": _artifact(artifacts / "diagnostics_dimensions.json")},
            },
            {
                "command": "render-report",
                "files": {"report": _artifact(artifacts / "fit_report.html")},
            },
        ],
    }
    path = tmp_path / "acceptance_summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _check_map(checks: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(check["name"]): check for check in checks}


def test_rejects_numpy_as_automatic_fit_owner_even_with_explicit_rust(tmp_path):
    """A later explicit Rust fit must not launder a NumPy automatic fit."""
    module = _load_sales_readiness()
    acceptance = _acceptance_summary(
        tmp_path,
        auto_backend="numpy",
        persisted_auto_backend="numpy",
    )

    checks = module._validate_acceptance_summary(
        acceptance,
        require_rust=True,
        max_acceptance_seconds=1.0,
    )

    by_name = _check_map(checks)
    assert by_name["acceptance:auto_fit_rust_authority"]["ok"] is False


def test_rejects_persisted_auto_summary_that_disagrees_with_rust_record(tmp_path):
    """Copied top-level Rust metadata cannot override persisted fit evidence."""
    module = _load_sales_readiness()
    acceptance = _acceptance_summary(
        tmp_path,
        auto_backend="rust",
        persisted_auto_backend="numpy",
    )

    checks = module._validate_acceptance_summary(
        acceptance,
        require_rust=True,
        max_acceptance_seconds=1.0,
    )

    by_name = _check_map(checks)
    assert by_name["acceptance:auto_fit_summary_backend"]["ok"] is False

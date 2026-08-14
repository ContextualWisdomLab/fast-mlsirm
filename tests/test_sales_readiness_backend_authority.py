"""Fail-closed tests for automatic release-acceptance backend authority."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_sales_readiness():
    """Load the sales-readiness script as an isolated test module."""

    script = Path(__file__).resolve().parents[1] / "scripts" / "sales_readiness.py"
    spec = importlib.util.spec_from_file_location("sales_readiness_backend_authority", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_acceptance(tmp_path: Path, *, auto_backend: str) -> Path:
    """Write acceptance evidence with separate automatic and explicit Rust fits."""

    artifacts = tmp_path / "artifacts"
    fit_auto = artifacts / "fit_auto"
    fit_rust = artifacts / "fit_rust"
    fit_auto.mkdir(parents=True)
    fit_rust.mkdir(parents=True)
    (fit_auto / "fit_summary.json").write_text(
        json.dumps({"backend": auto_backend}), encoding="utf-8"
    )
    (fit_rust / "fit_summary.json").write_text(
        json.dumps({"backend": "rust"}), encoding="utf-8"
    )
    summary = {
        "status": "ok",
        "total_duration_seconds": 0.25,
        "steps": [
            {"command": "simulate", "duration_seconds": 0.01},
            {
                "command": "fit",
                "backend": auto_backend,
                "out": str(fit_auto),
                "files": {"summary": str(fit_auto / "fit_summary.json")},
                "duration_seconds": 0.02,
            },
            {"command": "diagnose-fit", "duration_seconds": 0.03},
            {"command": "diagnose-dimensions", "duration_seconds": 0.04},
            {"command": "render-report", "duration_seconds": 0.05},
            {
                "command": "fit",
                "backend": "rust",
                "out": str(fit_rust),
                "files": {"summary": str(fit_rust / "fit_summary.json")},
                "duration_seconds": 0.02,
            },
        ],
    }
    path = tmp_path / "acceptance_summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def _checks_by_name(checks: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Index verifier checks by their stable evidence identifier."""

    return {str(check["name"]): check for check in checks}


def test_acceptance_rejects_numpy_automatic_fit_even_with_explicit_rust(tmp_path: Path) -> None:
    """A later explicit Rust fit must not legitimize a NumPy automatic fit."""

    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path, auto_backend="numpy")

    checks = _checks_by_name(
        module._validate_acceptance_summary(
            acceptance,
            require_rust=True,
            max_acceptance_seconds=None,
        )
    )

    assert checks["acceptance:auto_fit_backend_authority"]["ok"] is False


def test_acceptance_requires_persisted_auto_fit_summary_to_report_rust(tmp_path: Path) -> None:
    """Copied top-level Rust evidence cannot override a non-Rust persisted fit summary."""

    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path, auto_backend="rust")
    summary = json.loads(acceptance.read_text(encoding="utf-8"))
    fit_auto_summary = Path(summary["steps"][1]["files"]["summary"])
    fit_auto_summary.write_text(json.dumps({"backend": "numpy"}), encoding="utf-8")

    checks = _checks_by_name(
        module._validate_acceptance_summary(
            acceptance,
            require_rust=True,
            max_acceptance_seconds=None,
        )
    )

    assert checks["acceptance:auto_fit_summary_backend_authority"]["ok"] is False

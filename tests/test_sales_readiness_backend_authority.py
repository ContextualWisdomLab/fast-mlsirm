"""Commercial-readiness regressions for release backend authority."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_sales_readiness():
    """Load the sales-readiness verifier without making scripts a package."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "sales_readiness.py"
    spec = importlib.util.spec_from_file_location(
        "sales_readiness_backend_authority", script
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _artifact(path: Path) -> str:
    """Create one referenced acceptance artifact and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("evidence", encoding="utf-8")
    return str(path)


def test_sales_readiness_rejects_numpy_automatic_fit_even_with_explicit_rust(
    tmp_path,
) -> None:
    """A later explicit Rust fit cannot legitimize a NumPy automatic fit."""
    module = _load_sales_readiness()
    artifacts = tmp_path / "artifacts"
    summary = {
        "status": "ok",
        "total_duration_seconds": 1.0,
        "steps": [
            {
                "command": "simulate",
                "files": {"responses": _artifact(artifacts / "responses.npy")},
            },
            {
                "command": "fit",
                "backend": "numpy",
                "out": str(artifacts / "fit_auto"),
                "files": {"summary": _artifact(artifacts / "fit_auto" / "fit_summary.json")},
            },
            {
                "command": "fit",
                "backend": "rust",
                "out": str(artifacts / "fit_rust"),
                "files": {"summary": _artifact(artifacts / "fit_rust" / "fit_summary.json")},
            },
            {"command": "diagnose-fit"},
            {"command": "diagnose-dimensions"},
            {"command": "render-report"},
        ],
    }
    acceptance = tmp_path / "acceptance_summary.json"
    acceptance.write_text(json.dumps(summary), encoding="utf-8")

    checks = module._validate_acceptance_summary(
        acceptance, require_rust=True, max_acceptance_seconds=None
    )
    auto_rust = next(check for check in checks if check["name"] == "acceptance:auto_rust_fit")

    assert auto_rust["ok"] is False
    assert auto_rust["backends"] == ["numpy"]

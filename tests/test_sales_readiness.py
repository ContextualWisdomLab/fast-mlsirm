import argparse
import importlib.util
from pathlib import Path


def _load_sales_readiness():
    script = Path(__file__).resolve().parents[1] / "scripts" / "sales_readiness.py"
    spec = importlib.util.spec_from_file_location("sales_readiness", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok", encoding="utf-8")
    return str(path)


def _write_acceptance(tmp_path: Path, *, include_rust: bool = True) -> Path:
    module = _load_sales_readiness()
    artifacts = tmp_path / "artifacts"
    summary = {
        "status": "ok",
        "out": str(artifacts),
        "total_duration_seconds": 0.25,
        "steps": [
            {
                "command": "simulate",
                "duration_seconds": 0.01,
                "files": {
                    "responses": _touch(artifacts / "simulate" / "responses.npy"),
                    "factors": _touch(artifacts / "simulate" / "item_factor.csv"),
                },
            },
            {
                "command": "fit",
                "backend": "rust",
                "duration_seconds": 0.02,
                "out": str(artifacts / "fit_auto"),
                "files": {"summary": _touch(artifacts / "fit_auto" / "fit_summary.json")},
            },
            {
                "command": "diagnose-fit",
                "duration_seconds": 0.03,
                "files": {"diagnostics": _touch(artifacts / "diagnostics_fit" / "fit_diagnostics.json")},
            },
            {
                "command": "diagnose-dimensions",
                "duration_seconds": 0.04,
                "files": {
                    "diagnostics": _touch(artifacts / "diagnostics_dimensions" / "dimension_diagnostics.json")
                },
            },
            {
                "command": "render-report",
                "duration_seconds": 0.05,
                "files": {"report": _touch(artifacts / "fit_report.html")},
            },
        ],
    }
    if include_rust:
        summary["steps"].append(
            {
                "command": "fit",
                "backend": "rust",
                "duration_seconds": 0.02,
                "out": str(artifacts / "fit_rust"),
                "files": {"summary": _touch(artifacts / "fit_rust" / "fit_summary.json")},
            }
        )
    path = tmp_path / "acceptance_summary.json"
    path.write_text(module.json.dumps(summary), encoding="utf-8")
    return path


def test_sales_readiness_manifest_passes_with_acceptance_evidence(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    out = tmp_path / "sales_readiness_manifest.json"
    args = argparse.Namespace(
        repo_root=".",
        acceptance=str(acceptance),
        out=str(out),
        dist=None,
        require_rust=True,
        check_import=False,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "ok"
    assert manifest["contract_value_krw"] == 2_000_000_000
    assert out.exists()


def test_sales_readiness_fails_when_explicit_rust_evidence_is_missing(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path, include_rust=False)
    args = argparse.Namespace(
        repo_root=".",
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        check_import=False,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "failed"
    failed_names = {check["name"] for check in manifest["failed_checks"]}
    assert "acceptance:explicit_rust_fit" in failed_names


def test_project_version_parser_supports_python_310_fallback():
    module = _load_sales_readiness()

    version = module._parse_project_version(
        """
        [build-system]
        requires = ["maturin"]

        [project]
        name = "fast-mlsirm"
        version = "0.1.0"
        """
    )

    assert version == "0.1.0"

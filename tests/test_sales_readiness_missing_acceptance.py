import importlib.util
import json
from pathlib import Path


def _load_sales_readiness():
    script = Path(__file__).resolve().parents[1] / "scripts" / "sales_readiness.py"
    spec = importlib.util.spec_from_file_location("sales_readiness_missing_acceptance", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_missing_acceptance_writes_failed_manifest(tmp_path):
    module = _load_sales_readiness()
    repo_root = Path(__file__).resolve().parents[1]
    missing_acceptance = tmp_path / "missing_acceptance_summary.json"
    out = tmp_path / "sales_readiness_manifest.json"
    args = module.build_parser().parse_args(
        [
            "--repo-root",
            str(repo_root),
            "--acceptance",
            str(missing_acceptance),
            "--out",
            str(out),
        ]
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "failed"
    assert manifest["source_commit"] is None
    failed_by_name = {check["name"]: check for check in manifest["failed_checks"]}
    assert failed_by_name["acceptance:summary"]["ok"] is False
    assert "missing acceptance summary" in failed_by_name["acceptance:summary"]["detail"]
    assert json.loads(out.read_text(encoding="utf-8")) == manifest

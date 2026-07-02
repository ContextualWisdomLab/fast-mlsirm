import argparse
import hashlib
import importlib.util
import json
import zipfile
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


def _write_required_policy_files(root: Path) -> None:
    module = _load_sales_readiness()
    for relative in module.REQUIRED_POLICY_FILES:
        _touch(root / relative)
    token_docs = {
        "README.md": """
        Commercial Readiness
        Enterprise Sales Readiness
        scripts/release_acceptance.py
        scripts/sales_readiness.py
        """,
        "docs/commercial_readiness.md": """
        Seller Acceptance Checklist
        Enterprise Sales Gate
        Security
        Support
        Release Gate
        """,
        "docs/enterprise_sales_readiness.md": """
        KRW 2,000,000,000
        Procurement Evidence
        Customer Acceptance Evidence
        Go/No-Go
        Out of Scope
        """,
        "docs/release_acceptance.md": """
        acceptance_summary.json
        sales_readiness_manifest.json
        --require-rust
        """,
    }
    for relative, text in token_docs.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _write_20b_product_files(root: Path, *, code_connect: bool = False) -> None:
    docs = {
        "docs/20b_product_readiness.md": """
        KRW 2,000,000,000
        Buyer-Facing Product Standard
        Product Design Scope
        Data Analytics Scope
        Figma Code Connect
        Go/No-Go
        """,
        "docs/buyer_demo_storyboard.md": """
        Package Evidence
        Synthetic Data
        Fit Workflow
        Diagnostics Workflow
        Procurement Packet
        """,
        "docs/figma_product_design_packet.md": """
        Figma Code Connect is explicitly out of scope
        Package Evidence
        Procurement Packet
        figma_design_packet.json
        """,
        "docs/roi_evidence_model.md": """
        Driver Metrics
        Required Evidence
        roi_manifest.json
        Caveats
        """,
        "examples/enterprise_demo/README.md": "enterprise demo evidence",
        "docs/superpowers/specs/2026-07-02-20b-product-readiness-design.md": "design spec",
        "docs/superpowers/plans/2026-07-02-20b-product-readiness.md": "implementation plan",
    }
    for relative, text in docs.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    manifests = {
        "examples/enterprise_demo/roi_manifest.json": {
            "contract_value_krw": 2_000_000_000,
            "position": "Procurement evidence model, not a valuation guarantee.",
            "drivers": [{"name": "analyst_hours_saved"}],
            "required_evidence": ["acceptance_summary.json"],
            "go_no_go": {"requires_all_sales_readiness_checks_ok": True},
        },
        "examples/enterprise_demo/benchmark_manifest.json": {
            "benchmark_scope": "Synthetic dense MLS2PLM release-acceptance scenarios.",
            "runtime_budget_seconds": 120,
            "scenarios": [{"name": "small_rust_backend"}],
            "required_artifacts": ["fit_summary.json"],
            "caveats": ["acceptance benchmark contract"],
        },
        "examples/enterprise_demo/figma_design_packet.json": {
            "code_connect": code_connect,
            "mode": "static_product_storyboard",
            "source": "docs/buyer_demo_storyboard.md",
            "figma_artifact_url": "https://www.figma.com/design/example",
            "frames": [{"id": "01-package-evidence"}],
            "handoff": {"product_design_scope": "static buyer workflow"},
        },
        "examples/enterprise_demo/product_completion_manifest.json": {
            "contract_value_krw": 2_000_000_000,
            "scorecard_version": "2026-07-02",
            "checks": [
                {"id": "release_acceptance", "status": "go"},
                {"id": "html_report_csp", "status": "go"},
                {"id": "cli_stack_trace_guard", "status": "go"},
                {"id": "report_table_accessibility", "status": "go"},
                {"id": "figma_buyer_review", "status": "go"},
                {"id": "buyer_evidence_packet", "status": "go"},
            ],
            "go_no_go": {"requires_all_checks_go": True},
        },
    }
    for relative, payload in manifests.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")


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
        require_20b_product=False,
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
        require_20b_product=False,
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


def test_sales_readiness_passes_with_20b_product_evidence(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "ok"
    assert manifest["require_20b_product"] is True
    check_names = {check["name"] for check in manifest["checks"]}
    assert "20b:figma_code_connect_disabled" in check_names


def test_sales_readiness_fails_when_20b_artifact_is_missing(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    (repo_root / "docs" / "20b_product_readiness.md").unlink()
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "failed"
    failed_names = {check["name"] for check in manifest["failed_checks"]}
    assert "20b:file:docs/20b_product_readiness.md" in failed_names


def test_sales_readiness_fails_when_figma_code_connect_is_enabled(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root, code_connect=True)
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "failed"
    failed_names = {check["name"] for check in manifest["failed_checks"]}
    assert "20b:figma_code_connect_disabled" in failed_names


def test_sales_readiness_fails_when_optional_figma_url_is_not_a_design_file(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    path = repo_root / "examples" / "enterprise_demo" / "figma_design_packet.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["figma_artifact_url"] = "https://www.figma.com/board/example"
    path.write_text(json.dumps(payload), encoding="utf-8")
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "failed"
    figma_url_check = next(
        check for check in manifest["failed_checks"] if check["name"] == "20b:figma_artifact_url"
    )
    assert "https://www.figma.com/design/" in figma_url_check["detail"]
    assert "/board/" in figma_url_check["detail"]


def test_sales_readiness_fails_when_completion_scorecard_is_not_go(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    path = repo_root / "examples" / "enterprise_demo" / "product_completion_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checks"][1]["status"] = "blocked"
    path.write_text(json.dumps(payload), encoding="utf-8")
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "failed"
    scorecard_check = next(
        check for check in manifest["failed_checks"] if check["name"] == "20b:completion_scorecard"
    )
    assert scorecard_check["non_go"] == ["html_report_csp"]


def _write_buyer_packet_manifest(tmp_path: Path, *, zip_sha: str | None = None) -> Path:
    packet_zip = tmp_path / "buyer_packet.zip"
    with zipfile.ZipFile(packet_zip, "w") as packet:
        packet.writestr("buyer_evidence_manifest.json", "{}")
    actual_sha = hashlib.sha256(packet_zip.read_bytes()).hexdigest()
    manifest = {
        "status": "ok",
        "contract_value_krw": 2_000_000_000,
        "artifact_count": 12,
        "coverage": {
            "acceptance_summary": True,
            "sales_readiness_manifest": True,
            "wheel": True,
            "sdist": True,
            "product_docs": True,
            "product_manifests": True,
            "acceptance_artifacts": True,
        },
        "zip_file": str(packet_zip),
        "zip_sha256": zip_sha or actual_sha,
    }
    path = tmp_path / "buyer_evidence_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_sales_readiness_validates_required_buyer_packet(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    packet_manifest = _write_buyer_packet_manifest(tmp_path)
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        buyer_packet_manifest=str(packet_manifest),
        require_buyer_packet=True,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "ok"
    check_names = {check["name"] for check in manifest["checks"]}
    assert "buyer_packet:coverage" in check_names
    assert "buyer_packet:zip_sha256" in check_names


def test_sales_readiness_fails_when_buyer_packet_sha_mismatches(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    packet_manifest = _write_buyer_packet_manifest(tmp_path, zip_sha="0" * 64)
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        buyer_packet_manifest=str(packet_manifest),
        require_buyer_packet=True,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "failed"
    failed_names = {check["name"] for check in manifest["failed_checks"]}
    assert "buyer_packet:zip_sha256" in failed_names


def test_sales_readiness_fails_gracefully_when_20b_json_has_wrong_shape(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    (repo_root / "examples" / "enterprise_demo" / "roi_manifest.json").write_text("[]", encoding="utf-8")
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "failed"
    failed_names = {check["name"] for check in manifest["failed_checks"]}
    assert "20b:json_shape:examples/enterprise_demo/roi_manifest.json" in failed_names


def test_sales_readiness_treats_required_20b_json_none_as_empty(tmp_path):
    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path)
    repo_root = tmp_path / "repo"
    _write_required_policy_files(repo_root)
    _write_20b_product_files(repo_root)
    path = repo_root / "examples" / "enterprise_demo" / "roi_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["drivers"] = None
    path.write_text(json.dumps(payload), encoding="utf-8")
    args = argparse.Namespace(
        repo_root=str(repo_root),
        acceptance=str(acceptance),
        out=str(tmp_path / "sales_readiness_manifest.json"),
        dist=None,
        require_rust=True,
        require_20b_product=True,
        check_import=False,
        contract_value_krw=2_000_000_000,
        max_acceptance_seconds=1.0,
    )

    manifest = module.run_sales_readiness(args)

    assert manifest["status"] == "failed"
    json_field_check = next(
        check
        for check in manifest["failed_checks"]
        if check["name"] == "20b:json_fields:examples/enterprise_demo/roi_manifest.json"
    )
    assert json_field_check["empty"] == ["drivers"]

"""Regression tests for the price-neutral acquisition release orchestrator."""

from pathlib import Path
import subprocess
import sys

import pytest

from scripts import build_acquisition_release


ROOT = Path(__file__).resolve().parents[1]


def test_builder_defaults_to_no_transaction_value() -> None:
    """Generic acquisition evidence must not invent a deal value."""
    args = build_acquisition_release.build_parser().parse_args([])

    assert args.contract_value_krw is None


def test_final_sales_namespace_uses_generic_acquisition_profile() -> None:
    """The final gate must require generic evidence without legacy 20B mode."""
    args = build_acquisition_release._sales_args(
        repo_root=ROOT,
        acceptance=ROOT / "acceptance.json",
        dist=ROOT / "dist",
        out=ROOT / "sales.json",
        benchmark=ROOT / "benchmark.json",
        buyer_packet=ROOT / "buyer.json",
        release_index=ROOT / "release-index.json",
        procurement=ROOT / "procurement.json",
        pr_queue=ROOT / "pr-queue.json",
        figma=ROOT / "figma.json",
        require_rust=True,
        check_import=True,
        contract_value_krw=None,
        acquisition=True,
    )

    assert args.require_acquisition_readiness is True
    assert args.require_20b_product is False
    assert args.contract_value_krw is None
    assert args.require_buyer_packet is True
    assert args.require_release_evidence_index is True
    assert args.require_procurement_due_diligence is True
    assert args.require_pr_queue_governance is True
    assert args.require_figma_evidence_sync is True


def test_generic_builder_never_spells_legacy_gate_cli_flag() -> None:
    """The new orchestrator must not silently regress to the compatibility CLI."""
    source = (ROOT / "scripts" / "build_acquisition_release.py").read_text(
        encoding="utf-8"
    )

    assert "--require-20b-product" not in source


def test_release_acceptance_uses_operation_specific_bounded_timeout(monkeypatch) -> None:
    """The outer orchestrator must not kill a valid bounded acceptance run early."""
    observed: dict[str, float] = {}

    def fake_run_bounded_capture(command, *, cwd, timeout_seconds, **kwargs):
        observed["timeout_seconds"] = timeout_seconds
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        build_acquisition_release,
        "run_bounded_capture",
        fake_run_bounded_capture,
    )
    build_acquisition_release._run(
        [sys.executable, str(ROOT / "scripts" / "release_acceptance.py")],
        cwd=ROOT,
    )

    assert observed["timeout_seconds"] == pytest.approx(
        build_acquisition_release._RELEASE_ACCEPTANCE_TIMEOUT_SECONDS
    )
    assert observed["timeout_seconds"] > build_acquisition_release._STAGE_TIMEOUT_SECONDS


def test_non_skip_build_uses_clean_run_specific_distribution_directory(tmp_path: Path) -> None:
    """A new build must not admit stale artifacts from a shared dist directory."""
    shared_dist = tmp_path / "dist"
    shared_dist.mkdir()
    stale_wheel = shared_dist / "stale.whl"
    stale_wheel.write_bytes(b"old")
    out_dir = tmp_path / "evidence"
    prior_run_dist = out_dir / "distribution"
    prior_run_dist.mkdir(parents=True)
    (prior_run_dist / "prior.tar.gz").write_bytes(b"old")
    args = build_acquisition_release.build_parser().parse_args(
        ["--dist", str(shared_dist), "--out", str(out_dir)]
    )

    selected = build_acquisition_release._select_distribution_directory(args, out_dir)

    assert selected == out_dir / "distribution"
    assert selected.is_dir()
    assert list(selected.iterdir()) == []
    assert stale_wheel.read_bytes() == b"old"


def test_skip_build_preserves_explicit_caller_distribution_directory(tmp_path: Path) -> None:
    """Caller-supplied artifacts remain the only allowed reuse path."""
    shared_dist = tmp_path / "dist"
    shared_dist.mkdir()
    artifact = shared_dist / "candidate.whl"
    artifact.write_bytes(b"candidate")
    args = build_acquisition_release.build_parser().parse_args(
        ["--dist", str(shared_dist), "--skip-build"]
    )

    selected = build_acquisition_release._select_distribution_directory(
        args, tmp_path / "evidence"
    )

    assert selected == shared_dist.resolve()
    assert artifact.read_bytes() == b"candidate"


def test_stage_source_identity_is_required_and_exact() -> None:
    """Every generated stage manifest must bind to the sealed source commit."""
    expected = "a" * 40
    build_acquisition_release._require_source_identity(
        "buyer_packet", {"status": "ok", "source_commit": expected}, expected
    )

    with pytest.raises(RuntimeError, match="missing source_commit"):
        build_acquisition_release._require_source_identity(
            "buyer_packet", {"status": "ok"}, expected
        )
    with pytest.raises(RuntimeError, match="source_commit mismatch"):
        build_acquisition_release._require_source_identity(
            "buyer_packet", {"status": "ok", "source_commit": "b" * 40}, expected
        )


def test_source_movement_is_rejected_before_success(monkeypatch, tmp_path: Path) -> None:
    """A run cannot report success after repository HEAD changes mid-orchestration."""
    expected = "a" * 40
    monkeypatch.setattr(build_acquisition_release, "_source_commit", lambda _root: "b" * 40)

    with pytest.raises(RuntimeError, match="source HEAD changed"):
        build_acquisition_release._assert_source_unchanged(tmp_path, expected)

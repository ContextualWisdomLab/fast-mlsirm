"""Regression tests for the price-neutral acquisition release orchestrator."""

from pathlib import Path
import hashlib
import json
import subprocess
import sys

import pytest

from scripts import build_acquisition_release


ROOT = Path(__file__).resolve().parents[1]


def _init_git_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    tracked = root / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=fast-mlsirm-test",
            "-c",
            "user.email=fast-mlsirm-test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        check=True,
    )
    return tracked


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


def test_candidate_selection_rejects_ambiguous_reuse_directory(tmp_path: Path) -> None:
    """One acquisition bundle must bind to exactly one wheel and one sdist."""
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "candidate-a.whl").write_bytes(b"a")
    (dist / "candidate-b.whl").write_bytes(b"b")
    (dist / "candidate.tar.gz").write_bytes(b"sdist")

    with pytest.raises(RuntimeError, match="exactly one wheel and one source distribution"):
        build_acquisition_release._select_candidate_artifacts(dist)


def test_repository_cleanliness_rejects_tracked_and_untracked_inputs(tmp_path: Path) -> None:
    """Exact-source provenance must fail closed on source outside the sealed commit."""
    repo = tmp_path / "repo"
    tracked = _init_git_repo(repo)
    build_acquisition_release._require_clean_source(repo)

    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="working tree is not clean"):
        build_acquisition_release._require_clean_source(repo)
    subprocess.run(["git", "-C", str(repo), "checkout", "--", "tracked.txt"], check=True)

    (repo / "untracked.py").write_text("print('untracked')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="working tree is not clean"):
        build_acquisition_release._require_clean_source(repo)
    (repo / "untracked.py").unlink()

    output = repo / "acquisition-release"
    output.mkdir()
    (output / "generated.json").write_text("{}", encoding="utf-8")
    build_acquisition_release._require_clean_source(
        repo, allowed_untracked_root=output
    )


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


def test_candidate_import_probe_is_bound_to_selected_wheel(monkeypatch, tmp_path: Path) -> None:
    """Import evidence must come from an exact-wheel runtime used for acceptance."""
    wheel = tmp_path / "fast_mlsirm-0.9.1-py3-none-any.whl"
    wheel.write_bytes(b"candidate-wheel")
    out_dir = tmp_path / "evidence"
    out_dir.mkdir()
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, cwd: Path) -> None:
        commands.append(command)

    def fake_capture(command, *, cwd, timeout_seconds, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "package_version": "0.9.1",
                    "distribution_version": "0.9.1",
                    "package_file": str(
                        out_dir
                        / "candidate-runtime-env"
                        / "site-packages"
                        / "fast_mlsirm"
                        / "__init__.py"
                    ),
                    "rust_core": True,
                }
            ),
            "",
        )

    monkeypatch.setattr(build_acquisition_release, "_run", fake_run)
    monkeypatch.setattr(build_acquisition_release, "run_bounded_capture", fake_capture)
    evidence = build_acquisition_release._verify_candidate_import(
        wheel=wheel,
        out_dir=out_dir,
        source_commit="a" * 40,
        require_rust=True,
        python=sys.executable,
    )

    assert evidence["status"] == "ok"
    assert evidence["source_commit"] == "a" * 40
    assert evidence["wheel_sha256"] == hashlib.sha256(b"candidate-wheel").hexdigest()
    venv_commands = [command for command in commands if command[1:3] == ["-m", "venv"]]
    assert len(venv_commands) == 1
    assert "--system-site-packages" in venv_commands[0]
    assert any("--force-reinstall" in command and str(wheel) in command for command in commands)
    assert evidence["rust_core"] is True
    persisted = json.loads(
        (out_dir / "candidate_import_evidence.json").read_text(encoding="utf-8")
    )
    assert persisted["wheel_sha256"] == evidence["wheel_sha256"]


def test_stubbed_acquisition_run_preserves_stage_handoffs_and_final_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    """Exercise candidate-runtime, provenance, repeatability, and evidence handoffs."""
    source_commit = "a" * 40
    out_dir = tmp_path / "evidence"
    shared_dist = tmp_path / "shared-dist"
    shared_dist.mkdir()
    (shared_dist / "stale.whl").write_bytes(b"stale")
    observed_sales: list[object] = []
    clean_source_roots: list[Path | None] = []
    candidate_python = out_dir / "candidate-runtime-env" / "bin" / "python"
    acceptance_commands: list[list[str]] = []

    monkeypatch.setattr(
        build_acquisition_release, "_source_commit", lambda _root: source_commit
    )

    def fake_clean_source(_repo_root, *, allowed_untracked_root=None):
        clean_source_roots.append(allowed_untracked_root)

    monkeypatch.setattr(
        build_acquisition_release, "_require_clean_source", fake_clean_source
    )

    def write_manifest(path: Path) -> dict[str, object]:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "status": "ok",
            "source_commit": source_commit,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def fake_run(command: list[str], *, cwd: Path) -> None:
        if len(command) >= 3 and command[1:3] == ["-m", "build"]:
            dist = Path(command[command.index("--outdir") + 1])
            dist.mkdir(parents=True, exist_ok=True)
            (dist / "fast_mlsirm-0.0.0-py3-none-any.whl").write_bytes(b"wheel")
            (dist / "fast_mlsirm-0.0.0.tar.gz").write_bytes(b"sdist")
            return
        if len(command) >= 2 and Path(command[1]).name == "release_acceptance.py":
            acceptance_commands.append(command)
            acceptance = Path(command[command.index("--out") + 1])
            write_manifest(acceptance / "acceptance_summary.json")
            return
        raise AssertionError(f"unexpected subprocess stage: {command}")

    monkeypatch.setattr(build_acquisition_release, "_run", fake_run)
    monkeypatch.setattr(
        build_acquisition_release,
        "_prepare_candidate_environment",
        lambda **_kwargs: candidate_python,
    )

    def fake_candidate_import(
        *, wheel, out_dir, source_commit, require_rust, python, environment_python=None
    ):
        assert environment_python == candidate_python
        payload = {
            "status": "ok",
            "source_commit": source_commit,
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "rust_core": require_rust,
        }
        build_acquisition_release._write_json(
            out_dir / "candidate_import_evidence.json", payload
        )
        return payload

    monkeypatch.setattr(
        build_acquisition_release, "_verify_candidate_import", fake_candidate_import
    )

    def fake_benchmark(args):
        return write_manifest(Path(args.out) / "benchmark_report.json")

    def fake_sales(args):
        observed_sales.append(args)
        return write_manifest(Path(args.out))

    def fake_packet(args):
        packet_dir = Path(args.out)
        packet_dir.mkdir(parents=True, exist_ok=True)
        delivery = packet_dir / "fast_mlsirm_buyer_evidence_packet.zip"
        delivery.write_bytes(b"packet")
        digest = packet_dir / "fast_mlsirm_buyer_evidence_packet.sha256"
        digest.write_text(hashlib.sha256(delivery.read_bytes()).hexdigest() + "\n", encoding="ascii")
        payload = write_manifest(packet_dir / "buyer_evidence_manifest.json")
        payload["packet_file"] = str(delivery)
        payload["packet_sha256_file"] = str(digest)
        (packet_dir / "buyer_evidence_manifest.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return payload

    def fake_index(args):
        return write_manifest(Path(args.out) / "release_evidence_index.json")

    def fake_procurement(args):
        return write_manifest(Path(args.out) / "procurement_due_diligence_manifest.json")

    def fake_pr_queue(args):
        return write_manifest(Path(args.out) / "pr_queue_governance_manifest.json")

    def fake_figma(args):
        return write_manifest(Path(args.out) / "figma_evidence_sync_manifest.json")

    monkeypatch.setattr(
        build_acquisition_release.build_benchmark_report, "build_report", fake_benchmark
    )
    monkeypatch.setattr(
        build_acquisition_release.sales_readiness, "run_sales_readiness", fake_sales
    )
    monkeypatch.setattr(
        build_acquisition_release.build_buyer_packet, "build_packet", fake_packet
    )
    monkeypatch.setattr(
        build_acquisition_release.build_release_evidence_index, "build_index", fake_index
    )
    monkeypatch.setattr(
        build_acquisition_release.build_procurement_due_diligence,
        "build_procurement_due_diligence",
        fake_procurement,
    )
    monkeypatch.setattr(
        build_acquisition_release.build_pr_queue_governance,
        "build_pr_queue_governance",
        fake_pr_queue,
    )
    monkeypatch.setattr(
        build_acquisition_release.build_figma_evidence_sync,
        "build_figma_evidence_sync",
        fake_figma,
    )

    args = build_acquisition_release.build_parser().parse_args(
        [
            "--repo-root",
            str(tmp_path),
            "--out",
            str(out_dir),
            "--dist",
            str(shared_dist),
            "--check-import",
        ]
    )
    manifest = build_acquisition_release.build_acquisition_release(args)

    isolated_dist = out_dir / "distribution"
    assert all(Path(stage.dist) == isolated_dist for stage in observed_sales)
    assert all(stage.check_import is False for stage in observed_sales)
    assert len(observed_sales) == 3
    final_sales = observed_sales[-1]
    assert final_sales.require_acquisition_readiness is True
    assert Path(final_sales.buyer_packet_manifest).name == "buyer_evidence_manifest.json"
    assert Path(final_sales.release_evidence_index).name == "release_evidence_index.json"
    assert Path(final_sales.procurement_due_diligence).name == "procurement_due_diligence_manifest.json"
    assert Path(final_sales.pr_queue_governance).name == "pr_queue_governance_manifest.json"
    assert Path(final_sales.figma_evidence_sync).name == "figma_evidence_sync_manifest.json"
    assert acceptance_commands and acceptance_commands[0][0] == str(candidate_python)
    assert clean_source_roots and clean_source_roots[0] == out_dir
    assert manifest["source_commit"] == source_commit
    assert manifest["status"] == "ok"
    assert manifest["acceptance_runtime"]["python"] == str(candidate_python)
    assert manifest["artifacts"]["wheel"]["name"].endswith(".whl")
    assert manifest["artifacts"]["sdist"]["name"].endswith(".tar.gz")
    assert manifest["artifacts"]["candidate_import_evidence"]["exists"] is True
    assert manifest["artifacts"]["buyer_packet_delivery"]["exists"] is True
    assert manifest["artifacts"]["buyer_packet_delivery_digest"]["exists"] is True
    assert (out_dir / "acquisition_release_manifest.json").is_file()
    assert (shared_dist / "stale.whl").read_bytes() == b"stale"

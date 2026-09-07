#!/usr/bin/env python
"""Release acceptance smoke test for fast-mlsirm.

This script runs an end-to-end CLI workflow that mirrors a minimal production
verification path:
simulate -> fit (auto and optionally rust) -> diagnostics -> report rendering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    from scripts._bounded_json import parse_json_bounded, read_json_object
except ModuleNotFoundError:  # executed from scripts/ or loaded via importlib
    from _bounded_json import parse_json_bounded, read_json_object


RELEASE_ACCEPTANCE_TIMEOUT_SECONDS = {
    "simulate": 120.0,
    "fit_auto": 900.0,
    "fit_rust": 900.0,
    "diagnose-fit": 300.0,
    "diagnose-dimensions": 900.0,
    "render-report-fit": 120.0,
    "render-report-dimensions": 120.0,
}


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one finalized acceptance artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cli_timeout_seconds(out_label: str) -> float:
    """Return the bounded subprocess deadline for one acceptance operation."""
    try:
        return RELEASE_ACCEPTANCE_TIMEOUT_SECONDS[out_label]
    except KeyError:
        raise RuntimeError("unsupported release acceptance operation") from None


def _source_commit(repo_root: Path) -> str:
    """Return the exact source revision recorded by standalone acceptance.

    The acceptance summary is a portable evidence artifact. It therefore owns
    its source identity instead of relying on a later buyer or acquisition
    manifest to supply provenance for it.
    """
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    commit = completed.stdout.strip()
    if len(commit) not in {40, 64} or any(ch not in "0123456789abcdef" for ch in commit):
        raise RuntimeError("source commit lookup returned an invalid object id")
    return commit


def _admit_output_root(repo_root: Path, output_root: Path) -> Path:
    """Reject output roots whose generated-file exception could cover source."""
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    if output_root == repo_root or output_root in repo_root.parents:
        raise RuntimeError("output directory must not be the repository root or an ancestor")
    return output_root


def _admit_distribution_root(repo_root: Path, distribution_root: Path) -> Path:
    """Reject distribution roots whose artifact exception could cover source."""
    repo_root = repo_root.resolve()
    distribution_root = distribution_root.resolve()
    if distribution_root == repo_root or distribution_root in repo_root.parents:
        raise RuntimeError(
            "distribution directory must not be the repository root or an ancestor"
        )
    return distribution_root


def _is_inert_distribution_artifact(path: Path) -> bool:
    """Return whether an untracked path is a concrete wheel/sdist output only."""
    if path.is_symlink() or not path.is_file():
        return False
    return path.name.endswith(".whl") or path.name.endswith(".tar.gz")


def _require_clean_source(
    repo_root: Path,
    *,
    allowed_untracked_root: Path | None = None,
    allowed_distribution_root: Path | None = None,
) -> None:
    """Require clean source while admitting only explicitly scoped generated files.

    The acceptance output root may contain arbitrary generated acceptance files.
    A separately admitted distribution root may contain only concrete, non-symlink
    wheel/sdist outputs. Tracked/staged changes and every other untracked file
    remain fatal.
    """
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    allowed = allowed_untracked_root.resolve() if allowed_untracked_root is not None else None
    distribution = (
        allowed_distribution_root.resolve()
        if allowed_distribution_root is not None
        else None
    )
    violations: list[str] = []
    for record in completed.stdout.split("\0"):
        if not record:
            continue
        if len(record) < 4:
            violations.append(record)
            continue
        status = record[:2]
        relative_path = record[3:]
        if status != "??":
            violations.append(record)
            continue
        raw_candidate = repo_root / relative_path
        candidate = raw_candidate.resolve()
        if allowed is not None and (candidate == allowed or allowed in candidate.parents):
            continue
        if (
            distribution is not None
            and (candidate == distribution or distribution in candidate.parents)
            and _is_inert_distribution_artifact(raw_candidate)
        ):
            continue
        violations.append(record)
    if violations:
        preview = ", ".join(violations[:5])
        raise RuntimeError(f"source working tree is not clean: {preview}")


def _assert_source_unchanged(
    repo_root: Path,
    expected: str,
    *,
    allowed_untracked_root: Path | None = None,
    allowed_distribution_root: Path | None = None,
) -> None:
    """Fail closed if the source revision or working tree changes during acceptance."""
    actual = _source_commit(repo_root)
    if actual != expected:
        raise RuntimeError(
            f"source HEAD changed during release acceptance: expected {expected}, observed {actual}"
        )
    _require_clean_source(
        repo_root,
        allowed_untracked_root=allowed_untracked_root,
        allowed_distribution_root=allowed_distribution_root,
    )


def _acceptance_artifact_sha256(
    out_dir: Path, steps: object
) -> dict[str, str]:
    """Seal every step-declared artifact under the acceptance evidence root."""
    if not isinstance(steps, list):
        raise RuntimeError("acceptance steps must be a list before artifact sealing")
    evidence_root = out_dir.resolve()
    digests: dict[str, str] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        files = step.get("files")
        if files is None:
            continue
        if not isinstance(files, dict):
            raise RuntimeError("acceptance step files must be an object")
        for raw_path in files.values():
            if not isinstance(raw_path, str) or not raw_path:
                raise RuntimeError("acceptance artifact path must be a non-empty string")
            resolved = Path(raw_path).resolve()
            try:
                relative = resolved.relative_to(evidence_root)
            except ValueError:
                raise RuntimeError(
                    f"acceptance artifact is outside acceptance evidence root: {resolved}"
                ) from None
            if not resolved.is_file():
                raise RuntimeError(f"acceptance artifact is missing: {resolved}")
            digests[relative.as_posix()] = _sha256(resolved)
    return dict(sorted(digests.items()))


def _require_auto_fit_resolved_to_rust(
    summary: dict[str, object], fit_payload: dict[str, object]
) -> None:
    """Fail closed unless the auto fit recorded Rust as the numerical owner.

    Automatic production resolution never selects NumPy. A purchaser running
    release acceptance without ``--require-rust`` must still see Rust on the
    ``fit --backend auto`` path. Pass ``backend="numpy"`` only as an explicit
    reference/parity command, not as an accepted automatic outcome.
    """
    if summary.get("backend") != "rust":
        raise RuntimeError("fit auto backend must resolve to rust")
    if fit_payload.get("backend") != "rust":
        raise RuntimeError("fit_payload backend does not match fit_summary backend")


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    repo_python = Path(__file__).resolve().parents[1] / "python"
    has_pkg = importlib.util.find_spec("fast_mlsirm") is not None
    if not has_pkg:
        existing = env.get("PYTHONPATH", "")
        if existing:
            env["PYTHONPATH"] = f"{existing}{os.pathsep}{repo_python}"
        else:
            env["PYTHONPATH"] = str(repo_python)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _run_cli(
    args: list[str], out_label: str, *, require_json: bool = True
) -> dict[str, object]:
    command = [sys.executable, "-m", "fast_mlsirm.cli", *args]
    if require_json and "--json" not in command:
        command.append("--json")
    started = time.perf_counter()
    timeout_seconds = _cli_timeout_seconds(out_label)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=_cli_env(),
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{out_label} timed out") from None
    duration_seconds = round(time.perf_counter() - started, 6)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        raise RuntimeError(f"{out_label} failed ({command}): {stderr or stdout}")
    raw_output = completed.stdout.strip().splitlines()
    if not raw_output:
        raise RuntimeError(f"{out_label} succeeded without JSON output")
    if not require_json:
        return {
            "status": "ok",
            "command": out_label,
            "stdout": raw_output[-1],
            "duration_seconds": duration_seconds,
        }
    try:
        payload = parse_json_bounded(raw_output[-1])
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object")
        payload["duration_seconds"] = duration_seconds
        return payload
    except ValueError as exc:
        raise RuntimeError(
            f"{out_label} produced non-JSON output: {raw_output[-1]}"
        ) from exc


def _run_acceptance(args: argparse.Namespace) -> dict[str, object]:
    acceptance_started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = _admit_output_root(repo_root, Path(args.out))
    distribution_root = (
        _admit_distribution_root(repo_root, Path(args.distribution_root))
        if args.distribution_root is not None
        else None
    )
    _require_clean_source(
        repo_root,
        allowed_untracked_root=out_dir,
        allowed_distribution_root=distribution_root,
    )
    source_commit = _source_commit(repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "command": "release_acceptance",
        "status": "ok",
        "out": str(out_dir),
        "steps": [],
    }

    sim_out = out_dir / "simulate"
    fit_auto_out = out_dir / "fit_auto"
    fit_rust_out = out_dir / "fit_rust"
    diag_fit_out = out_dir / "diagnostics_fit"
    diag_dim_out = out_dir / "diagnostics_dimensions"
    report_fit_out = out_dir / "fit_report.html"
    report_dim_out = out_dir / "dimension_report.html"

    simulate_payload = _run_cli(
        [
            "simulate",
            "--persons",
            str(args.persons),
            "--dims",
            str(args.dims),
            "--items-per-dim",
            str(args.items_per_dim),
            "--latent-dim",
            str(args.latent_dim),
            "--seed",
            str(args.seed),
            "--out",
            str(sim_out),
        ],
        "simulate",
    )
    report["steps"].append(simulate_payload)

    fit_auto_payload = _run_cli(
        [
            "fit",
            "--responses",
            str(sim_out / "responses.npy"),
            "--factors",
            str(sim_out / "item_factor.csv"),
            "--max-iter",
            str(args.max_iter),
            "--n-restarts",
            str(args.n_restarts),
            "--optimizer",
            "adam",
            "--backend",
            "auto",
            "--latent-dim",
            str(args.latent_dim),
            "--seed",
            str(args.seed),
            "--out",
            str(fit_auto_out),
        ],
        "fit_auto",
    )
    report["steps"].append(fit_auto_payload)

    summary = read_json_object(fit_auto_out / "fit_summary.json")
    _require_auto_fit_resolved_to_rust(summary, fit_auto_payload)

    if args.require_rust:
        fit_rust_payload = _run_cli(
            [
                "fit",
                "--responses",
                str(sim_out / "responses.npy"),
                "--factors",
                str(sim_out / "item_factor.csv"),
                "--max-iter",
                str(args.max_iter),
                "--n-restarts",
                str(args.n_restarts),
                "--optimizer",
                "adam",
                "--backend",
                "rust",
                "--latent-dim",
                str(args.latent_dim),
                "--seed",
                str(args.seed),
                "--out",
                str(fit_rust_out),
            ],
            "fit_rust",
        )
        report["steps"].append(fit_rust_payload)
        rust_summary = read_json_object(fit_rust_out / "fit_summary.json")
        if rust_summary.get("backend") != "rust":
            raise RuntimeError("rust fit did not report rust backend")

    fit_diagnostics_payload = _run_cli(
        [
            "diagnose-fit",
            "--responses",
            str(sim_out / "responses.npy"),
            "--factors",
            str(sim_out / "item_factor.csv"),
            "--params",
            str(fit_auto_out / "params.npz"),
            "--out",
            str(diag_fit_out),
        ],
        "diagnose-fit",
    )
    report["steps"].append(fit_diagnostics_payload)

    dimensionality_payload = _run_cli(
        [
            "diagnose-dimensions",
            "--responses",
            str(sim_out / "responses.npy"),
            "--factors",
            str(sim_out / "item_factor.csv"),
            "--latent-dims",
            args.latent_dims,
            "--folds",
            str(args.folds),
            "--max-iter",
            str(args.max_iter),
            "--seed",
            str(args.seed),
            "--out",
            str(diag_dim_out),
        ],
        "diagnose-dimensions",
    )
    report["steps"].append(dimensionality_payload)

    render_fit_payload = _run_cli(
        [
            "render-report",
            "--diagnostics",
            str(diag_fit_out / "fit_diagnostics.json"),
            "--out",
            str(report_fit_out),
            "--json",
        ],
        "render-report-fit",
        require_json=True,
    )
    report["steps"].append(render_fit_payload)

    render_dim_payload = _run_cli(
        [
            "render-report",
            "--diagnostics",
            str(diag_dim_out / "dimension_diagnostics.json"),
            "--out",
            str(report_dim_out),
            "--json",
        ],
        "render-report-dimensions",
        require_json=True,
    )
    report["steps"].append(render_dim_payload)

    for path in [
        sim_out / "responses.npy",
        sim_out / "item_factor.csv",
        fit_auto_out / "params.npz",
        fit_auto_out / "fit_summary.json",
        diag_fit_out / "fit_diagnostics.json",
        diag_dim_out / "dimension_diagnostics.json",
        report_fit_out,
        report_dim_out,
    ]:
        if not path.exists():
            raise RuntimeError(f"expected artifact missing: {path}")

    if args.require_rust:
        for path in [
            fit_rust_out / "params.npz",
            fit_rust_out / "fit_summary.json",
        ]:
            if not path.exists():
                raise RuntimeError(f"expected rust artifact missing: {path}")

    _assert_source_unchanged(
        repo_root,
        source_commit,
        allowed_untracked_root=out_dir,
        allowed_distribution_root=distribution_root,
    )
    summary_path = out_dir / "acceptance_summary.json"
    summary_payload = {
        "status": "ok",
        "out": str(out_dir),
        "source_commit": source_commit,
        "artifact_sha256": _acceptance_artifact_sha256(out_dir, report["steps"]),
        "steps": report["steps"],
        "total_duration_seconds": round(time.perf_counter() - acceptance_started, 6),
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _assert_source_unchanged(
        repo_root,
        source_commit,
        allowed_untracked_root=out_dir,
        allowed_distribution_root=distribution_root,
    )
    return {"status": "ok", "out": str(out_dir), "report": str(summary_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a fast-mlsirm release acceptance smoke test."
    )
    parser.add_argument(
        "--out",
        default="acceptance_check",
        help="Output directory for generated artifacts.",
    )
    parser.add_argument(
        "--distribution-root",
        help=(
            "Explicit directory containing wheel/sdist build outputs that source "
            "sealing may admit as regular package artifacts."
        ),
    )
    parser.add_argument(
        "--persons", type=int, default=12, help="Number of persons to simulate."
    )
    parser.add_argument(
        "--dims", type=int, default=1, help="Simulated true item dimensions."
    )
    parser.add_argument(
        "--items-per-dim",
        type=int,
        default=2,
        help="Items per dimension for simulation.",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=1, help="Latent dimension for fitting."
    )
    parser.add_argument("--seed", type=int, default=1, help="Simulation seed.")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1,
        help="Max optimization iterations for fitting.",
    )
    parser.add_argument(
        "--n-restarts", type=int, default=1, help="Optimization restarts."
    )
    parser.add_argument(
        "--latent-dims",
        default="1,2",
        help="Comma-separated latent dims for dimension diagnostics.",
    )
    parser.add_argument(
        "--folds", type=int, default=2, help="CV folds for dimensionality diagnostics."
    )
    parser.add_argument(
        "--require-rust",
        action="store_true",
        help="Also run fit in explicit rust backend mode; fail if unavailable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = _run_acceptance(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
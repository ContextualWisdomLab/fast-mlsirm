"""합성 P2/S4 기준 적합의 동일 입력·장치·수렴·시간을 기록한다.

Cai (2010), pp.608–609 Appendices A/B의 기존 테스트 생성기와 적합을
재사용한다. 연구자료 파일을 읽지 않으며, 사람별 결과를 저장하지 않는다.
q/반복/허용오차는 호출자 설정이며 결과를 연구 수치 수용으로 판정하지 않는다.
참고: Cai, L. (2010). A two-tier full-information item factor analysis model
with applications. Psychometrika, 75(4), 581–612.
https://doi.org/10.1007/s11336-010-9178-0
"""

import argparse
import hashlib
import json
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from fast_mlsirm import fit_two_tier_grm
from fast_mlsirm import _core
from fast_mlsirm.regression import absolute_differences

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_two_tier_focal_gaussian_recovery_native import _continuous_six_latent_fixture


def _execution_identity(receipt_path):
    if receipt_path is None:
        raise ValueError("현재 source의 build receipt가 필요합니다.")
    receipt = json.loads(receipt_path.read_text())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True, timeout=10).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True, timeout=10))
    core_path = Path(_core.__file__).resolve()
    if dirty or not any(core_path.name.endswith(suffix) for suffix in EXTENSION_SUFFIXES):
        raise ValueError("clean source와 native extension이 필요합니다.")
    with core_path.open("rb") as binary:
        core_hash = hashlib.file_digest(binary, "sha256").hexdigest()
    if (receipt.get("source_head") != head or receipt.get("source_tree") != tree
            or receipt.get("core_sha256") != core_hash or receipt.get("gpu_feature") is not True):
        raise ValueError("build receipt와 현재 source·extension이 일치하지 않습니다.")
    for field in ("build_command", "rustc_version", "cargo_version"):
        if not isinstance(receipt.get(field), str) or not receipt[field].strip():
            raise ValueError("build command와 toolchain 기록이 필요합니다.")
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (ROOT / "Cargo.lock", ROOT / "pyproject.toml",
                                  ROOT / "crates/mlsirm-core/src/two_tier_grm.rs",
                                  ROOT / "crates/mlsirm-core/src/poly.rs",
                                  ROOT / "crates/mlsirm-core/src/gpu_bifactor.rs",
                                  ROOT / "crates/fast-mlsirm-py/src/lib.rs")}
    return dict(head=head, source_tree=tree, dirty=dirty, core_sha256=core_hash,
                source_sha256=source_hashes,
                core_import_path=str(core_path), core_module=_core.__name__,
                build_receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                build_command_sha256=hashlib.sha256(receipt["build_command"].encode()).hexdigest(),
                rustc_version=receipt["rustc_version"], cargo_version=receipt["cargo_version"],
                gpu_feature=True, build_provenance="supplied receipt matched; not independent attestation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persons", type=int, required=True)
    parser.add_argument("--q-primary", type=int, required=True)
    parser.add_argument("--q-specific", type=int, required=True)
    parser.add_argument("--max-iter", type=int, required=True)
    parser.add_argument("--tol", type=float, required=True)
    parser.add_argument("--n-starts", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu-memory-budget-bytes", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu", "both"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path,
                        help="source_head/source_tree/core_sha256/gpu_feature/build_command/rustc_version/cargo_version을 담은 build JSON")
    parser.add_argument("--fixture-contract", type=Path,
                        default=ROOT / "tests/fixtures/two_tier_reference_gpu/sparse_p2_s4.json")
    args = parser.parse_args()
    if args.persons <= 0:
        parser.error("persons must be positive")
    if args.out.exists():
        parser.error("결과 파일이 이미 있습니다. 새 보고서 경로를 지정하세요.")
    identity = None
    identity_error = None
    try:
        identity = _execution_identity(args.build_receipt)
    except Exception as error:
        identity_error = error
    try:
        output = args.out.open("x")
    except FileExistsError:
        parser.error("결과 파일이 이미 있습니다. 새 보고서 경로를 지정하세요.")
    def write_report():
        output.seek(0)
        output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        output.truncate()
        output.flush()
    report = dict(scope="synthetic execution evidence, not research acceptance",
                  status="preflight", command=sys.argv, runs={},
                  truth_recovery_acceptance="not_evaluated",
                  backend_parity={"status": "not_evaluated"},
                  exit_status_meaning="provenance/device/input and requested convergence; existing pair parity, not truth recovery")
    write_report()
    try:
        if identity_error is not None:
            raise identity_error
        report.update(identity)
        y, ap, asp, sm, threshold, response_hash = _continuous_six_latent_fixture(
            args.persons, np.zeros(6), np.ones(6), args.seed
        )
        fixture = json.loads(args.fixture_contract.read_text())
        expected_controls = dict(n_persons=args.persons, q_primary=args.q_primary,
                                 q_specific=args.q_specific, max_iter=args.max_iter,
                                 tol=args.tol, n_starts=args.n_starts, seed=args.seed,
                                 n_items=16, n_primary=2, n_specific=4, n_cat=4)
        actual_digest = hashlib.sha256(y.astype("<i8").tobytes()).hexdigest()
        if (any(fixture.get(key) != value for key, value in expected_controls.items())
                or actual_digest != response_hash or fixture["responses_sha256"] != response_hash
                or not np.array_equal(np.asarray(fixture["y"]).reshape(y.shape), y)
                or not np.array_equal(np.asarray(fixture["primary_map"]).reshape(ap.shape), ap != 0)
                or not np.array_equal(fixture["specific_map"], sm)
                or any(set(y[:, i]) != set(range(4)) for i in range(16))):
            raise ValueError("합성 fixture와 응답·map·caller controls가 일치하지 않습니다.")
        report["fixture_contract"] = dict(matched=True, path=str(args.fixture_contract),
            sha256=hashlib.sha256(args.fixture_contract.read_bytes()).hexdigest())
        report["truth"] = {}
        for name, array in (("a_primary", ap), ("a_specific", asp), ("threshold", threshold)):
            array = np.asarray(array, dtype="<f8")
            report["truth"][name] = dict(values=array.tolist(), shape=list(array.shape),
                dtype=array.dtype.str, sha256=hashlib.sha256(array.tobytes()).hexdigest())
        report["truth_coordinates"] = "generator coordinates; no post-hoc rotation or alignment; active primary slopes only"
        report["latent_generator"] = dict(distribution="continuous Gaussian", mean=[0.0]*6, sd=[1.0]*6)
        report["primary_map"] = (ap != 0).tolist()
        report["specific_map"] = sm.tolist()
    except Exception as error:
        report.update(status="failed", exit_status=1,
                      preflight_error=dict(error_type=type(error).__name__, error=str(error)))
        write_report()
        output.close()
        return 1
    controls = dict(n_cat=4, n_primary=2, n_specific=4,
                    q_primary=args.q_primary, q_specific=args.q_specific,
                    max_iter=args.max_iter, tol=args.tol,
                    n_starts=args.n_starts, seed=args.seed,
                    primary_correlation="identity")
    digest = hashlib.sha256()
    for array in (y.astype("<i8"), (ap != 0).astype("u1"), sm.astype("<i8")):
        digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
        digest.update(array.tobytes())
    digest.update(json.dumps(controls, sort_keys=True).encode())
    report.update(python=platform.python_version(), platform=platform.platform(),
                  input_sha256=digest.hexdigest(), responses_sha256=response_hash,
                  persons=args.persons, items=16, controls=controls,
                  gpu_memory_budget_bytes=args.gpu_memory_budget_bytes)
    fits = {}
    failed = False
    for device in ("cpu", "gpu") if args.device == "both" else (args.device,):
        report["status"] = "running"
        report["active_device"] = device
        write_report()
        started = time.perf_counter()
        try:
            fit = fit_two_tier_grm(y, ap != 0, sm, **controls, device=device,
                                  gpu_memory_budget_bytes=args.gpu_memory_budget_bytes)
            if fit.backend != device or (device == "gpu" and
                    (not fit.gpu_adapter_name or not fit.gpu_adapter_backend)):
                raise ValueError("요청 장치와 반환 장치·adapter 기록이 일치하지 않습니다.")
            arrays = {}
            for name in ("a_primary", "a_specific", "threshold", "phi", "theta_p_eap", "theta_p_sd", "loglik_trace"):
                array = np.asarray(getattr(fit, name), dtype=np.float64)
                if not np.isfinite(array).all() or array.size == 0:
                    raise ValueError("반환값에 비유한 값 또는 빈 배열이 있습니다.")
                arrays[name] = array.tolist()
            truth_error = {}
            for name, truth in (("a_primary", ap), ("a_specific", asp), ("threshold", threshold)):
                estimated = np.asarray(getattr(fit, name), dtype=np.float64)
                if estimated.shape != truth.shape:
                    raise ValueError("반환 모수와 참모수 shape가 다릅니다.")
                residual = estimated - truth
                if name == "a_primary":
                    residual = residual[ap != 0]
                truth_error[name] = dict(signed_residual=residual.tolist(),
                    max_abs_diff=float(np.max(np.abs(residual))), descriptive_only=True)
            ll = float(fit.loglik_trace[-1])
            if not np.isfinite(fit.final_loglik_change):
                raise ValueError("반환 likelihood 변화가 비유한 값입니다.")
            fits[device] = fit
            report["runs"][device] = dict(
                seconds=time.perf_counter() - started, converged=fit.converged,
                n_iter=fit.n_iter, termination_reason=fit.termination_reason,
                backend=fit.backend, gpu_adapter_name=fit.gpu_adapter_name,
                gpu_adapter_backend=fit.gpu_adapter_backend,
                loglik=ll, aic=2 * fit.n_parameters - 2 * ll,
                bic=fit.n_parameters * np.log(args.persons) - 2 * ll,
                final_loglik_change=fit.final_loglik_change,
                parameters={name: arrays[name] for name in ("a_primary", "a_specific", "threshold", "phi")},
                loglik_trace=arrays["loglik_trace"], truth_error=truth_error,
                n_parameters=fit.n_parameters,
                category_counts=np.asarray(fit.category_counts).tolist(),
            )
            failed |= not fit.converged
        except Exception as error:
            report["runs"][device] = dict(seconds=time.perf_counter() - started,
                                         error_type=type(error).__name__, error=str(error))
            failed = True
        write_report()
    if set(fits) == {"cpu", "gpu"}:
        report["max_absolute_delta"] = {
            name: absolute_differences(
                np.asarray(getattr(fits["gpu"], name), dtype=np.float64).reshape(-1),
                np.asarray(getattr(fits["cpu"], name), dtype=np.float64).reshape(-1),
            )["max_abs_diff"]
            for name in ("a_primary", "a_specific", "threshold", "phi", "theta_p_eap", "theta_p_sd")
        }
        report["loglik_delta"] = report["runs"]["gpu"]["loglik"] - report["runs"]["cpu"]["loglik"]
        report["fit_statistic_delta"] = {name: report["runs"]["gpu"][name] - report["runs"]["cpu"][name]
                                         for name in ("aic", "bic")}
        passed = (all(value < 1e-3 for value in report["max_absolute_delta"].values())
                  and abs(report["loglik_delta"]) < 1e-3
                  and fits["cpu"].n_parameters == fits["gpu"].n_parameters
                  and np.array_equal(fits["cpu"].category_counts, fits["gpu"].category_counts))
        report["backend_parity"] = dict(status="evaluated", passed=passed,
            basis="tests/unit/two_tier_grm_tests.rs:1433–1459; strict max parameter and likelihood gap < 1e-3",
            truth_recovery_criterion=False)
        failed |= not passed
    report["exit_status"] = int(failed)
    report["status"] = "failed" if failed else "complete"
    report.pop("active_device", None)
    write_report()
    output.close()
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

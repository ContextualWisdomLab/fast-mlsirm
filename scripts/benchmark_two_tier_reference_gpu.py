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
    args = parser.parse_args()
    if args.persons <= 0:
        parser.error("persons must be positive")
    try:
        output = args.out.open("x")
    except FileExistsError:
        parser.error("결과 파일이 이미 있습니다. 새 보고서 경로를 지정하세요.")
    def write_report():
        output.seek(0)
        output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        output.truncate()
        output.flush()
    y, ap, _, sm, _, response_hash = _continuous_six_latent_fixture(
        args.persons, np.zeros(6), np.ones(6), args.seed
    )
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
    with Path(_core.__file__).open("rb") as binary:
        build_hash = hashlib.file_digest(binary, "sha256").hexdigest()
    report = dict(scope="synthetic execution evidence, not research acceptance",
                  python=platform.python_version(), platform=platform.platform(),
                  head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
                  core_sha256=build_hash, input_sha256=digest.hexdigest(),
                  responses_sha256=response_hash, persons=args.persons, items=16,
                  controls=controls, gpu_memory_budget_bytes=args.gpu_memory_budget_bytes,
                  command=sys.argv, runs={})
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
            fits[device] = fit
            ll = float(fit.loglik_trace[-1])
            report["runs"][device] = dict(
                seconds=time.perf_counter() - started, converged=fit.converged,
                n_iter=fit.n_iter, termination_reason=fit.termination_reason,
                backend=fit.backend, gpu_adapter_name=fit.gpu_adapter_name,
                gpu_adapter_backend=fit.gpu_adapter_backend,
                loglik=ll, aic=2 * fit.n_parameters - 2 * ll,
                bic=fit.n_parameters * np.log(args.persons) - 2 * ll,
                final_loglik_change=fit.final_loglik_change,
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
    report["exit_status"] = int(failed)
    report["status"] = "failed" if failed else "complete"
    report.pop("active_device", None)
    write_report()
    output.close()
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

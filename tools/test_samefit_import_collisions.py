"""Native CLI regressions; pass the built samefit_import example as argv[1]."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    executable = str(Path(sys.argv[1]).resolve())
    summary = dict(
        anchor=[True, True],
        coordinate_map=[{"Slope": [0, 0]}, {"FirstIntercept": [0, 0]},
                        {"LogGap": [0, 0, 1]}, {"Slope": [0, 1]},
                        {"FirstIntercept": [0, 1]}, {"LogGap": [0, 1, 1]},
                        {"Mean": 1}, {"LogSd": 1}],
        packed=[0., 0.6931471805599453, 0.32663425997828094,
                0., 0.6931471805599453, 0.32663425997828094, 0., 0.],
        q_general=121, loglik_cpu_f64=-4.394449154672439,
        fit_identity="synthetic_collision_control",
    )
    request = json.dumps(dict(y=[0, 1, 2, 0], observed=None, group_id=[0, 1],
                              n_groups=2, n_cat=3, item_ids=["i1", "i2"],
                              q_general=121)).encode()
    cases = [("control", None, None), ("loglik", "loglik_cpu_f64", 0.),
             ("quadrature", "q_general", 241), ("anchor", "anchor", [False, True])]
    with tempfile.TemporaryDirectory(prefix="samefit-collisions-") as tmp:
        root = Path(tmp)
        request_path = root / "request.json"
        request_path.write_bytes(request)
        for name, field, value in cases:
            other = copy.deepcopy(summary)
            if field:
                other[field] = value
            result = json.dumps(dict(
                request_sha256=hashlib.sha256(request).hexdigest(),
                result=dict(rounds=[dict(item_results=[dict(null=summary, alternative=other)])]),
            )).encode()
            result_path = root / (name + ".json")
            result_path.write_bytes(result)
            output = root / (name + "-output.json")
            process = subprocess.run(
                [executable, str(request_path), str(result_path),
                 hashlib.sha256(result).hexdigest(), str(summary["fit_identity"]), str(output)],
                capture_output=True, text=True, timeout=30,
                env=dict(os.environ, FMLS_SAMEFIT_DIAGNOSTICS="0"),
            )
            if field:
                assert process.returncode != 0, (name, process.stdout, process.stderr)
                assert "identity collision" in process.stderr, (name, process.stderr)
                assert not output.exists(), name
            else:
                assert process.returncode == 0, process.stderr
                assert output.exists()
                assert "repeated_locations=2" in process.stdout
            print(name, "PASS", "exit", process.returncode)


if __name__ == "__main__":
    main()

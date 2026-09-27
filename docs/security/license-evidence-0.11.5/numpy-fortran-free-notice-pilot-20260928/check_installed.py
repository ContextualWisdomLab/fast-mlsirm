"""Bounded install/NOTICE/native-loader smoke check, not release acceptance."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sha256", default="011ba92963c8fd230f3e81ac8f20af6c91e08bb139f0497cdb6ede1cc95c9a06")
args = parser.parse_args()
wheel = next((Path(__file__).parent / "input").glob("*.whl"))
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == args.sha256
distribution = importlib.metadata.distribution("numpy")
assert distribution.version == "2.5.2"
assert "OPENBLAS-0.3.34-NOTICES.txt" in distribution.metadata.get_all("License-File")
notice = next(p for p in distribution.files if str(p).endswith("licenses/OPENBLAS-0.3.34-NOTICES.txt"))
assert hashlib.sha256(distribution.locate_file(notice).read_bytes()).hexdigest() == "9f21f7061f26cdc6f173c29a5a2754c68326d7b397c6bc75bfb4f7543ed21ba4"
matrix = np.array([[4., 1., 0.], [1., 3., 1.], [0., 1., 2.]])
rhs = np.array([1., 2., 3.])
solution = np.linalg.solve(matrix, rhs)
assert np.allclose(matrix @ solution, rhs, rtol=1e-12, atol=1e-12)
assert np.allclose(matrix @ np.linalg.inv(matrix), np.eye(3), rtol=1e-12, atol=1e-12)
paths = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                if len(line.split()) >= 6 and line.split()[-1].startswith("/")})
assert not any("libgfortran" in p or "libquadmath" in p for p in paths)
providers = []
for path in paths:
    if "libgcc_s" in path or "libstdc++" in path:
        resolved = str(Path(path).resolve())
        candidates = [path, resolved]
        if path.startswith("/usr/lib/"):
            alias = path[4:]
            if Path(alias).exists() and Path(alias).samefile(path):
                candidates.append(alias)
        for candidate in dict.fromkeys(candidates):
            result = subprocess.run(["dpkg-query", "-S", candidate], capture_output=True, text=True)
            if result.returncode == 0:
                break
        providers.append({"loaded_path": path, "resolved_path": resolved,
                          "ownership_query_path": candidate,
                          "provider_query_exit": result.returncode, "provider": result.stdout.strip()})
print(json.dumps({"status": "local install/NOTICE/loader smoke only; release HOLD", "numpy_path": np.__file__,
                  "python": sys.version, "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                  "embedded_notice_verified": True, "solve_and_inverse_smoke": True,
                  "loaded_gfortran_quadmath": False, "gnu_runtime_providers": providers}, indent=2))

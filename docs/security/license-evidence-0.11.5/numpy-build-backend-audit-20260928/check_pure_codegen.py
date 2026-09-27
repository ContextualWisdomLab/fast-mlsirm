"""Exercise pure Cython code generation only; no NumPy build or numerical verdict."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import Cython

root = Path(__file__).resolve().parent
wheel = root / "input/cython-3.3.0-py3-none-any.whl"
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == "9b24b5c8cd536946b62086fcafee6d5509d3f549f72d553d2336af87ffbe0da1"
assert Cython.__version__ == "3.3.0"
module = Path(Cython.__file__).resolve()
assert module.is_relative_to(Path(sys.prefix).resolve())
assert not list(module.parent.rglob("*.so"))
source = root / "pure_smoke.pyx"
source.write_text("def scaled(double value, double scale):\n    return value * scale\n")
target = root / "pure_smoke.c"
subprocess.run([sys.executable, "-m", "cython", "-3", str(source), "-o", str(target)], check=True)
code = target.read_text()
assert "PyInit_pure_smoke" in code and "scaled" in code
receipt = {"python": sys.version, "cython": Cython.__version__, "cython_module": str(module), "native_cython_modules": 0, "generated_c_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "scope": "pure Python code generator smoke only; no compiled numerical or NumPy verdict"}
(root / "codegen-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))

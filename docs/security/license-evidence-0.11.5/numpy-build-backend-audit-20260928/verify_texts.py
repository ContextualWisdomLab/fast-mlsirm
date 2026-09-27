"""Verify the archived license bytes and the narrow Ninja terms comparison."""
from pathlib import Path
import hashlib
import json
import re

root = Path(__file__).resolve().parent
receipt = json.loads((root / "backend-archive-receipt.json").read_text())
for package in receipt["packages"]:
    for grant in package["grant_files"]:
        suffix = grant["path"].split(".dist-info/licenses/", 1)[1]
        path = (root / "texts" / package["name"] / suffix).resolve()
        assert path.is_relative_to(root / "texts")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == grant["sha256"]
for grant in receipt["cython_companion"]["texts"]:
    path = root / grant["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == grant["sha256"]
canonical = (root / "LICENSE.txt").read_text()
for label in ("(a)", "(b)", "(c)", "(d)"):
    canonical = canonical.replace(label, "")
ninja = (root / "texts/ninja/LICENSE_Apache_20").read_text().split("APPENDIX:", 1)[0]
ninja = ninja.replace("http://www.apache.org/licenses/", "https://www.apache.org/licenses/")
assert re.sub(r"\s+", " ", ninja).strip() == re.sub(r"\s+", " ", canonical).strip()
print("Recorded license hashes and narrow Apache terms equivalence passed; no archive rights verdict.")

"""Fail if a built wheel omits or changes its selected Rust license bundle."""

import hashlib
import sys
import zipfile
from pathlib import Path


wheels = list(Path(sys.argv[1]).glob("*.whl"))
if len(wheels) != 1:
    raise SystemExit(f"expected one wheel, found {len(wheels)}")

expected = Path("LICENSE-THIRD-PARTY").read_bytes()
with zipfile.ZipFile(wheels[0]) as wheel:
    members = [name for name in wheel.namelist()
               if name.endswith(".dist-info/licenses/LICENSE-THIRD-PARTY")]
    if len(members) != 1 or wheel.read(members[0]) != expected:
        raise SystemExit("wheel third-party license bundle differs from selected source")

print(f"{wheels[0].name}: LICENSE-THIRD-PARTY sha256 {hashlib.sha256(expected).hexdigest()}")

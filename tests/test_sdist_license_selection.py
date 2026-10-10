"""Exercise target selection from the real maturin source distribution."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[1]


def test_sdist_carries_executable_target_license_selection(tmp_path: Path) -> None:
    """The published source archive supplies the consumer's exact license inputs."""
    env = dict(os.environ)
    env["CARGO_NET_OFFLINE"] = "true"
    result = subprocess.run(
        [sys.executable, "-m", "maturin", "sdist", "--out", str(tmp_path / "dist")],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    archives = list((tmp_path / "dist").glob("*.tar.gz"))
    assert len(archives) == 1
    required = [
        "tools/select_third_party_license.sh",
        "tools/verify_wheel_license.py",
        "tools/third_party_licenses.snapshot.json",
    ]
    for target in ("aarch64-unknown-linux-gnu", "x86_64-pc-windows-msvc"):
        required.extend([
            f"docs/security/license-evidence-0.11.5/target-notices/LICENSE-THIRD-PARTY-{target}",
            f"docs/security/license-evidence-0.11.5/target-notices/{target}.snapshot.json",
        ])
    with tarfile.open(archives[0], "r:gz") as archive:
        roots = {member.name.split("/", 1)[0] for member in archive.getmembers()}
        assert len(roots) == 1
        prefix = next(iter(roots))
        names = set(archive.getnames())
        missing = [name for name in required if prefix + "/" + name not in names]
        assert not missing, f"sdist omits consumer license inputs: {missing}"
        for name in required:
            assert archive.extractfile(prefix + "/" + name).read() == (ROOT / name).read_bytes()
        archive.extractall(tmp_path / "extracted", filter="data")
    extracted = tmp_path / "extracted" / prefix
    for target in ("x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu", "x86_64-pc-windows-msvc"):
        selected = tmp_path / target
        shutil.copytree(extracted, selected)
        selected_env = {**env, "PATH": str(Path(sys.executable).parent) + os.pathsep + env["PATH"]}
        selection = subprocess.run(
            ["bash", "tools/select_third_party_license.sh", target],
            cwd=selected, env=selected_env, capture_output=True, text=True, timeout=15,
        )
        assert selection.returncode == 0, selection.stderr
        expected = ROOT / "LICENSE-THIRD-PARTY" if target == "x86_64-unknown-linux-gnu" else (
            ROOT / "docs/security/license-evidence-0.11.5/target-notices" / f"LICENSE-THIRD-PARTY-{target}"
        )
        assert (selected / "LICENSE-THIRD-PARTY").read_bytes() == expected.read_bytes()
    unsupported = subprocess.run(
        ["bash", "tools/select_third_party_license.sh", "universal2-apple-darwin"],
        cwd=extracted, env=selected_env, capture_output=True, text=True, timeout=15,
    )
    assert unsupported.returncode != 0 and "no reviewed" in unsupported.stderr

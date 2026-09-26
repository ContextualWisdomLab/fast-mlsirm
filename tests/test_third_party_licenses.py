"""Focused tests for tools/third_party_licenses.py (synthetic crates only)."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "third_party_licenses.py"
_spec = importlib.util.spec_from_file_location("third_party_licenses", SCRIPT)
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

MIT = "Permission is hereby granted, free of charge, to any person obtaining a copy ...\n"


def _crate(cache: Path, name: str, version: str, files: dict[str, str]) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for fname, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(f"{name}-{version}/{fname}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    (cache / "index").mkdir(parents=True, exist_ok=True)
    (cache / "index" / f"{name}-{version}.crate").write_bytes(buf.getvalue())
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _row(name, digest, files, spdx="MIT OR Apache-2.0", own=False):
    return {"name": name, "version": "1.0", "source": "path (this repository)" if own else "registry+x",
            "spdx_normalized": spdx, "elected_license": "MIT", "license_class": "PERMISSIVE",
            "in_binding_target_graph": True, "source_hash": {"measured": digest},
            "license_files_in_artifact": [{"path": p, "sha256": hashlib.sha256(t.encode()).hexdigest(),
                                           "verified_standard_text": ["MIT"]} for p, t in files.items()]}


def _render(tmp_path, rows):
    snap = T.snapshot({"cargo": rows}, [], "x86_64-unknown-linux-gnu")
    return snap, T.render(snap, tmp_path / "cache", {}, tmp_path)


def test_output_is_deterministic_and_follows_the_inventory(tmp_path):
    a = _crate(tmp_path / "cache", "a", "1.0", {"LICENSE-MIT": MIT})
    b = _crate(tmp_path / "cache", "b", "1.0", {"LICENSE-MIT": MIT + "Copyright b\n"})
    rows = [_row("b", b, {"LICENSE-MIT": MIT + "Copyright b\n"}), _row("a", a, {"LICENSE-MIT": MIT}),
            _row("own", "0" * 64, {}, own=True)]
    snap, (text, _) = _render(tmp_path, rows)
    assert [r["name"] for r in snap["rows"]] == ["a", "b"]
    assert _render(tmp_path, list(reversed(rows)))[1][0] == text
    assert _render(tmp_path, rows[1:])[1][0] != text


def test_linked_crate_without_text_fails(tmp_path):
    digest = _crate(tmp_path / "cache", "a", "1.0", {"Cargo.toml": ""})
    with pytest.raises(SystemExit, match="no license text"):
        _render(tmp_path, [_row("a", digest, {})])
    rows = [_row("a", digest, {"LICENSE-MIT": MIT})]
    with pytest.raises(SystemExit, match="missing from the .crate"):
        _render(tmp_path, rows)
    rows[0]["license_class"] = "HOLD"
    with pytest.raises(SystemExit, match="not PERMISSIVE"):
        _render(tmp_path, rows)


def test_apache_notice_is_included_when_present(tmp_path):
    files = {"LICENSE-MIT": MIT, "NOTICE": "Example Project\nCopyright 2020 Example\n"}
    digest = _crate(tmp_path / "cache", "a", "1.0", files)
    _, (text, notices) = _render(tmp_path, [_row("a", digest, {"LICENSE-MIT": MIT})])
    assert notices == 1 and "a 1.0: NOTICE (NOTICE) -----\nExample Project" in text
    _, (text, notices) = _render(tmp_path, [_row("a", digest, {"LICENSE-MIT": MIT}, spdx="MIT")])
    assert notices == 0 and "Example Project" not in text


def test_committed_file_matches_committed_snapshot_header():
    root = SCRIPT.parents[1]
    snap = root / "tools" / "third_party_licenses.snapshot.json"
    text = (root / "LICENSE-THIRD-PARTY").read_text()
    assert f"snapshot sha256 {hashlib.sha256(snap.read_bytes()).hexdigest()}" in text
    assert f"Entries: {len(json.loads(snap.read_text())['rows'])}." in text
    assert T.MARKER in text


def test_libm_source_notices_keep_complete_conditions():
    text = (SCRIPT.parents[1] / "NOTICE-libm-0.2.16-source-notices.txt").read_text()
    blocks = text.split("### block ")[1:]
    sun = [block for block in blocks if "provided that this notice" in block]
    freebsd = [block for block in blocks if "2. Redistributions in binary form" in block]
    assert sun and freebsd
    assert all("is preserved." in block for block in sun)
    assert all("SUCH DAMAGE." in block for block in freebsd)


@pytest.mark.parametrize(("target", "expected"), [
    ("x86_64-unknown-linux-gnu", "d46f307a2e8a49e2d638ee4e0b768c6c908c7786cb9106e74948561bf8cf0af0"),
    ("aarch64-unknown-linux-gnu", "afa61d22b98ec3b8d4797c4af1dc9a6d159b2051a543e658c67f4d2404d2d87c"),
    ("x86_64-pc-windows-msvc", "7a3537a1df4063760bfee6e720b7cc9105721748d3fffd29aed2732d155dde58"),
    ("universal2-apple-darwin", None),
])
def test_wheel_notice_selection_is_target_bound(tmp_path, target, expected):
    root = SCRIPT.parents[1]
    shutil.copyfile(root / "LICENSE-THIRD-PARTY", tmp_path / "LICENSE-THIRD-PARTY")
    source_dir = root / "docs/security/license-evidence-0.11.5/target-notices"
    target_dir = tmp_path / "docs/security/license-evidence-0.11.5/target-notices"
    target_dir.mkdir(parents=True)
    for name in ("aarch64-unknown-linux-gnu", "x86_64-pc-windows-msvc"):
        shutil.copyfile(source_dir / f"LICENSE-THIRD-PARTY-{name}",
                        target_dir / f"LICENSE-THIRD-PARTY-{name}")
    result = subprocess.run(["bash", str(root / "tools/select_third_party_license.sh"), target],
                            cwd=tmp_path, capture_output=True, text=True, check=False)
    if expected is None:
        assert result.returncode != 0 and "no reviewed" in result.stderr
        expected = "d46f307a2e8a49e2d638ee4e0b768c6c908c7786cb9106e74948561bf8cf0af0"
    else:
        assert result.returncode == 0, result.stderr
    assert hashlib.sha256((tmp_path / "LICENSE-THIRD-PARTY").read_bytes()).hexdigest() == expected


def test_built_wheel_must_contain_selected_license_bytes(tmp_path):
    root = SCRIPT.parents[1]
    notice = root.joinpath("LICENSE-THIRD-PARTY").read_bytes()
    tmp_path.joinpath("LICENSE-THIRD-PARTY").write_bytes(notice)
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / "fast_mlsirm-0.11.5-py3-none-any.whl"
    member = "fast_mlsirm-0.11.5.dist-info/licenses/LICENSE-THIRD-PARTY"

    for contents, valid in ((notice, True), (b"wrong target\n", False)):
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(member, contents)
        result = subprocess.run(
            [sys.executable, str(root / "tools/verify_wheel_license.py"), "dist"],
            cwd=tmp_path, capture_output=True, text=True, check=False,
        )
        assert (result.returncode == 0) is valid

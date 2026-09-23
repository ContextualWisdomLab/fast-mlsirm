"""Negative regression tests for tools/license_inventory.py (fail-closed behaviour)."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "license_inventory.py"
MIT_TEXT = """MIT License

Copyright (c) 2026 Test Author

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
LGPL_TEXT = "This library is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License"


def _module():
    """Load the generator from its file path, as the other tools tests do."""
    spec = importlib.util.spec_from_file_location("license_inventory", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


L = _module()


@pytest.mark.parametrize("expr", ["LicenseRef-unreviewed", "NOASSERTION", "NONE", "", "   ", "MIT AND", "(MIT", "MIT WITH Bogus-exception"])
def test_unreviewed_or_malformed_expressions_are_unknown(expr):
    """Unregistered ids, sentinels and grammar errors never classify as permissive."""
    assert L.classify(expr) == "UNKNOWN"
    elected, rationale = L.elect(expr)
    assert elected is None and rationale


def test_failed_disjunction_election_propagates_unknown():
    """A disjunction without a fully permissive branch cannot yield 'None AND MIT'."""
    expr = "(GPL-3.0-only OR LGPL-2.1-only) AND MIT"
    assert L.classify(expr) == "COPYLEFT"
    assert L.elect(expr) == (None, L.ELECTION_FAILED)


def test_permissive_election_keeps_conjunct_and_copyleft_alternative_is_declared():
    """Valid elections still work; a copyleft alternative stays visible as declared class."""
    assert L.elect("(MIT OR Apache-2.0) AND Unicode-3.0")[0] == "MIT AND Unicode-3.0"
    assert L.elect("MIT OR Apache-2.0 OR LGPL-2.1-or-later")[0] == "MIT"
    assert L.classify("MIT OR Apache-2.0 OR LGPL-2.1-or-later") == "COPYLEFT"
    assert L.classify("GPL-3.0-or-later WITH GCC-exception-3.1") == "COPYLEFT"


def _wheel(path: Path, name: str, version: str, license_text: str | None) -> str:
    """Write a minimal wheel and return its sha256."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{name}-{version}.dist-info/METADATA", f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n\n")
        if license_text is not None:
            zf.writestr(f"{name}-{version}.dist-info/licenses/LICENSE", license_text)
    path.write_bytes(buf.getvalue())
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _python_args(tmp_path: Path, *, expression, license_text, lock_hash=None, scope=True, requirements="") -> Namespace:
    """Build python_inventory inputs for one package 'pkg 1.0'."""
    art = tmp_path / "art"
    meta = tmp_path / "meta"
    art.mkdir()
    meta.mkdir()
    digest = None
    if license_text is not False:
        digest = _wheel(art / "pkg-1.0-py3-none-any.whl", "pkg", "1.0", license_text)
    (meta / "pkg-1.0.json").write_text(json.dumps({"info": {"license_expression": expression, "classifiers": []}, "urls": []}))
    uv = tmp_path / "uv.lock"
    uv.write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\nsource = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{lock_hash or digest or "0" * 64}" }}]\n'
    )
    req = tmp_path / "requirements.txt"
    req.write_text(requirements)
    scope_file = tmp_path / "scope.json"
    scope_file.write_text(json.dumps({"pkg": {"scope": "dev", "in_published_artifact_scope": False}} if scope else {}))
    return Namespace(uv_lock=str(uv), requirements=[str(req)], extra_python=[], python_scope=str(scope_file),
                     pypi_meta_dir=str(meta), pypi_artifact_dir=str(art))


def test_declared_license_without_artifact_text_is_hold(tmp_path):
    """Metadata MIT with zero examined artifacts must not read as PERMISSIVE."""
    gaps = []
    (row,) = L.python_inventory(_python_args(tmp_path, expression="MIT", license_text=False), gaps)
    assert row["license_class"] == "HOLD"
    assert row["license_determination"]["text_agrees"] is False


def test_copyleft_text_is_kept_when_metadata_is_absent(tmp_path):
    """No metadata plus MIT and LGPL text holds, keeping LGPL instead of dropping it."""
    gaps = []
    (row,) = L.python_inventory(_python_args(tmp_path, expression=None, license_text=MIT_TEXT + "\n" + LGPL_TEXT), gaps)
    assert row["license_class"] == "UNKNOWN"
    assert row["elected_license"] is None
    assert "LGPL" in row["license_determination"]["copyleft_text_not_in_declared_expression"]
    assert row["hold_reasons"]


def test_copyleft_text_outside_declared_expression_is_hold(tmp_path):
    """Declared MIT with LGPL grant text in the artifact is HOLD, not PERMISSIVE."""
    gaps = []
    (row,) = L.python_inventory(_python_args(tmp_path, expression="MIT", license_text=MIT_TEXT + "\n" + LGPL_TEXT), gaps)
    assert row["license_class"] == "HOLD"
    assert row["license_determination"]["copyleft_text_not_in_declared_expression"] == ["LGPL"]


def test_artifact_hash_mismatch_is_hold_and_text_unused(tmp_path):
    """An artifact whose sha256 is not the lock's expected hash is not evidence."""
    gaps = []
    (row,) = L.python_inventory(_python_args(tmp_path, expression="MIT", license_text=MIT_TEXT, lock_hash="f" * 64), gaps)
    assert row["artifacts_examined"][0]["hash_binding"]["match"] is False
    assert row["license_text_detected_in_artifact"] == []
    assert row["license_class"] == "HOLD"


def test_bound_artifact_with_matching_text_is_permissive(tmp_path):
    """Control: hash-bound artifact whose text matches the declared expression passes."""
    gaps = []
    (row,) = L.python_inventory(_python_args(tmp_path, expression="MIT", license_text=MIT_TEXT), gaps)
    assert (row["license_class"], row["hold_reasons"], gaps) == ("PERMISSIVE", [], [])


def test_missing_scope_is_unclassified_gap_not_exclusion(tmp_path):
    """A package without a scope entry is UNCLASSIFIED with unknown scope, and a gap."""
    gaps = []
    (row,) = L.python_inventory(_python_args(tmp_path, expression="MIT", license_text=MIT_TEXT, scope=False), gaps)
    assert row["scope"] == "UNCLASSIFIED"
    assert row["in_published_artifact_scope"] is None
    assert any("no scope entry" in g for g in gaps)


@pytest.mark.parametrize("line", ["other==2.0\n", "other>=2.0\n", "-e .\n", "other @ https://x/other.whl\n"])
def test_requirements_lines_that_do_not_parse_are_gaps(tmp_path, line):
    """Unhashed pins and non-pin requirement lines are reported, not silently skipped."""
    gaps = []
    L.python_inventory(_python_args(tmp_path, expression="MIT", license_text=MIT_TEXT, requirements=line), gaps)
    assert any("requirements.txt" in g for g in gaps)


def _crate(cache: Path, name: str, version: str, files: dict[str, str]) -> str:
    """Write a .crate (tar.gz) into a registry-cache layout and return its sha256."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for fname, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(f"{name}-{version}/{fname}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    (cache / "index.crates.io").mkdir(parents=True, exist_ok=True)
    (cache / "index.crates.io" / f"{name}-{version}.crate").write_bytes(buf.getvalue())
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _rust_args(tmp_path: Path, *, lock_checksum_override=None, drop_from_metadata=False) -> Namespace:
    """Build rust_inventory inputs with one registry crate 'dep 1.0'."""
    cache = tmp_path / "cache"
    digest = _crate(cache, "dep", "1.0", {"LICENSE-MIT": MIT_TEXT, "Cargo.toml": ""})
    lock = tmp_path / "Cargo.lock"
    lock.write_text(
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        f'source = "registry+https://github.com/rust-lang/crates.io-index"\nchecksum = "{lock_checksum_override or digest}"\n'
    )
    packages = [] if drop_from_metadata else [{
        "name": "dep", "version": "1.0", "source": "registry+https://github.com/rust-lang/crates.io-index",
        "license": "MIT OR Apache-2.0", "license_file": None, "manifest_path": "/nonexistent/Cargo.toml",
    }]
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"packages": packages}))
    sbom = tmp_path / "sbom.json"
    sbom.write_text(json.dumps({"components": []}))
    tree = tmp_path / "tree"
    tree.mkdir()
    return Namespace(cargo_metadata_workspace=str(meta), cargo_metadata_binding=str(meta), cargo_lock_workspace=str(lock),
                     cargo_lock_binding=str(lock), cargo_registry_cache=str(cache), wheel_sbom=str(sbom), tree_dir=str(tree))


def test_crate_license_text_is_read_only_from_hash_bound_crate(tmp_path):
    """Matching .crate bytes: text comes from the archive and the row passes."""
    gaps = []
    (row,) = L.rust_inventory(_rust_args(tmp_path), gaps)
    assert row["source_hash"]["match"] is True
    assert [f["path"] for f in row["license_files_in_artifact"]] == ["LICENSE-MIT"]
    assert row["license_files_in_artifact"][0]["artifact_sha256"] == row["source_hash"]["measured"]
    assert (row["license_class"], gaps) == ("PERMISSIVE", [])


def test_crate_hash_mismatch_is_hold_and_text_unread(tmp_path):
    """A .crate that does not match the Cargo.lock checksum is not evidence."""
    gaps = []
    (row,) = L.rust_inventory(_rust_args(tmp_path, lock_checksum_override="0" * 64), gaps)
    assert row["source_hash"]["match"] is False
    assert row["license_files_in_artifact"] == []
    assert row["license_class"] == "HOLD"
    assert row["value_origin"].startswith("UNVERIFIED")


def test_lock_entry_missing_from_metadata_is_a_gap_and_unknown(tmp_path):
    """Every lock package yields a row; a metadata/lock mismatch is reported."""
    gaps = []
    (row,) = L.rust_inventory(_rust_args(tmp_path, drop_from_metadata=True), gaps)
    assert row["license_class"] == "UNKNOWN"
    assert any("not in cargo metadata" in g for g in gaps)


def test_registry_crate_without_license_file_is_hold(tmp_path):
    """Cargo metadata alone cannot replace license text from the exact archive."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    digest = _crate(cache, "dep", "1.0", {"Cargo.toml": ""})
    Path(args.cargo_lock_workspace).write_text(
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_binding).write_text(Path(args.cargo_lock_workspace).read_text())
    (row,) = L.rust_inventory(args, [])
    assert row["license_files_in_artifact"] == []
    assert row["license_class"] == "HOLD"
    assert "contains no license file" in " ".join(row["hold_reasons"])


def test_cargo_metadata_mit_does_not_hide_separate_lgpl_text(tmp_path):
    """A permissive declaration cannot erase an additional copyleft grant."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    digest = _crate(cache, "dep", "1.0", {"LICENSE-MIT": MIT_TEXT, "COPYING.LESSER": LGPL_TEXT})
    Path(args.cargo_lock_workspace).write_text(
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_binding).write_text(Path(args.cargo_lock_workspace).read_text())
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert "LGPL" in " ".join(row["hold_reasons"])


def test_crate_archive_is_read_once_for_hash_and_license_text(tmp_path, monkeypatch):
    """The bytes hashed are the same bytes used to extract license evidence."""
    args = _rust_args(tmp_path)
    original = L.read_stable_bytes
    calls = []

    def counted(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(L, "read_stable_bytes", counted)
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "PERMISSIVE"
    assert len(calls) == 1


def test_crate_path_replacement_during_read_is_hold(tmp_path, monkeypatch):
    """Replacing the source path after open cannot retain a trusted hash binding."""
    args = _rust_args(tmp_path)
    crate = next(Path(args.cargo_registry_cache).glob("*/dep-1.0.crate"))
    original_open = Path.open

    class ReplacingReader:
        def __init__(self, handle):
            self.handle = handle
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return self.handle.__exit__(*exc)
        def fileno(self):
            return self.handle.fileno()
        def read(self):
            data = self.handle.read()
            replacement = crate.with_suffix(".replacement")
            replacement.write_bytes(data)
            replacement.replace(crate)
            return data

    def replacing_open(path, *open_args, **open_kwargs):
        handle = original_open(path, *open_args, **open_kwargs)
        return ReplacingReader(handle) if path == crate else handle

    monkeypatch.setattr(Path, "open", replacing_open)
    (row,) = L.rust_inventory(args, [])
    assert row["source_hash"]["match"] is False
    assert row["license_class"] == "HOLD"
    assert "source path was not stable" in " ".join(row["hold_reasons"])


@pytest.mark.parametrize(
    "license_text",
    [
        "UNKNOWN",
        "Commercial redistribution is prohibited. Permission is not granted.",
        "Permission is hereby granted, free of charge, but this grant does not apply. "
        'THE SOFTWARE IS PROVIDED "AS IS". IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE.',
    ],
)
def test_cargo_permissive_metadata_requires_verified_unqualified_text(tmp_path, license_text):
    """Names, restrictions, and negated standard phrases are not permissive evidence."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    digest = _crate(cache, "dep", "1.0", {"LICENSE": license_text})
    Path(args.cargo_lock_workspace).write_text(
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_binding).write_text(Path(args.cargo_lock_workspace).read_text())
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert row["elected_text_present_in_artifact"] is False


def test_nested_uppercase_lgpl_grant_is_collected_and_holds(tmp_path):
    """Nested candidate paths and case variants cannot hide copyleft text."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    uppercase_lgpl = LGPL_TEXT.upper()
    digest = _crate(
        cache,
        "dep",
        "1.0",
        {"LICENSE-MIT": MIT_TEXT, "vendor/other/COPYING": uppercase_lgpl},
    )
    Path(args.cargo_lock_workspace).write_text(
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_binding).write_text(Path(args.cargo_lock_workspace).read_text())
    (row,) = L.rust_inventory(args, [])
    assert {f["path"] for f in row["license_files_in_artifact"]} == {
        "LICENSE-MIT", "vendor/other/COPYING"
    }
    assert row["license_class"] == "HOLD"
    assert "LGPL" in " ".join(row["hold_reasons"])


def test_declared_nested_license_file_is_collected(tmp_path):
    """Cargo metadata's LicenseFile path is evidence even without a license-like basename."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    digest = _crate(cache, "dep", "1.0", {"legal/grant.txt": MIT_TEXT})
    meta_path = Path(args.cargo_metadata_workspace)
    metadata = json.loads(meta_path.read_text())
    metadata["packages"][0]["license_file"] = "legal/grant.txt"
    meta_path.write_text(json.dumps(metadata))
    Path(args.cargo_metadata_binding).write_text(meta_path.read_text())
    Path(args.cargo_lock_workspace).write_text(
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_binding).write_text(Path(args.cargo_lock_workspace).read_text())
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "PERMISSIVE"
    assert row["license_files_in_artifact"][0]["path"] == "legal/grant.txt"
    assert row["license_files_in_artifact"][0]["declared_license_file"] is True


def test_unknown_notice_alongside_mit_is_hold(tmp_path):
    """An unparsed candidate file remains UNKNOWN even when MIT text is present."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    digest = _crate(cache, "dep", "1.0", {"LICENSE-MIT": MIT_TEXT, "NOTICE": "UNKNOWN"})
    Path(args.cargo_lock_workspace).write_text(
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_binding).write_text(Path(args.cargo_lock_workspace).read_text())
    (row,) = L.rust_inventory(args, [])
    assert row["elected_text_present_in_artifact"] is True
    assert row["license_class"] == "HOLD"
    assert "unrecognized" in " ".join(row["hold_reasons"])


def test_full_mit_text_with_additional_condition_is_hold(tmp_path):
    """A canonical grant plus an extra restriction is not the canonical MIT license."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    restricted = MIT_TEXT + "\nAdditional condition: Use is permitted solely for academic research.\n"
    digest = _crate(cache, "dep", "1.0", {"LICENSE": restricted})
    Path(args.cargo_lock_workspace).write_text(
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_binding).write_text(Path(args.cargo_lock_workspace).read_text())
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert row["elected_text_present_in_artifact"] is False


def test_missing_declared_license_file_is_hold_even_with_root_mit(tmp_path):
    """A different root license cannot substitute for Cargo metadata's missing path."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    digest = _crate(cache, "dep", "1.0", {"LICENSE": MIT_TEXT})
    meta_path = Path(args.cargo_metadata_workspace)
    metadata = json.loads(meta_path.read_text())
    metadata["packages"][0]["license_file"] = "legal/terms.txt"
    meta_path.write_text(json.dumps(metadata))
    Path(args.cargo_metadata_binding).write_text(meta_path.read_text())
    Path(args.cargo_lock_workspace).write_text(
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_binding).write_text(Path(args.cargo_lock_workspace).read_text())
    (row,) = L.rust_inventory(args, [])
    assert row["elected_text_present_in_artifact"] is True
    assert row["license_class"] == "HOLD"
    assert "license_file is absent" in " ".join(row["hold_reasons"])


def test_positive_phrase_only_is_not_verified_for_other_supported_license(tmp_path):
    """A known Apache phrase is detection evidence, not a complete verified grant."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    digest = _crate(cache, "dep", "1.0", {"LICENSE": "Apache License Version 2.0"})
    meta_path = Path(args.cargo_metadata_workspace)
    metadata = json.loads(meta_path.read_text())
    metadata["packages"][0]["license"] = "Apache-2.0"
    meta_path.write_text(json.dumps(metadata))
    Path(args.cargo_metadata_binding).write_text(meta_path.read_text())
    Path(args.cargo_lock_workspace).write_text(
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_binding).write_text(Path(args.cargo_lock_workspace).read_text())
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert row["elected_text_present_in_artifact"] is False

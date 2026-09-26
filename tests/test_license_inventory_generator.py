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
MIT_WITH_HEADER = MIT_TEXT.replace(
    "MIT License\n\n", "MIT License\n\nCopyright (c) 2026 Test Author\n\n", 1
)
LGPL_TEXT = "This library is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License"


def _module():
    """Load the generator from its file path, as the other tools tests do."""
    spec = importlib.util.spec_from_file_location("license_inventory", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


L = _module()

REVIEWED = json.loads((Path(__file__).parent / "fixtures/license_inventory_reviewed_texts.json").read_text())


@pytest.mark.parametrize("row", REVIEWED, ids=lambda r: r["package"])
def test_reviewed_whole_license_bytes_and_mutations(row):
    text = row["text"]
    assert hashlib.sha256(text.encode()).hexdigest() == row["raw_sha256"]
    normalized = L.re.sub(r"[ \t\r\n]+", " ", text).strip(" \t\r\n")
    assert hashlib.sha256(normalized.encode()).hexdigest() == row["normalized_sha256"]
    assert L.verified_standard_text(text) == [row["identifier"]]
    assert L.verified_standard_text(text.replace("\n", "\r\n")) == [row["identifier"]]
    for changed in (text.replace("License", "Restriction", 1) if "License" in text else text.replace("software", "hardware", 1),
                    text + "\nCommercial use is prohibited.", "extra\n" + text, text + "\nextra"):
        assert changed != text
        assert L.verified_standard_text(changed) == []


@pytest.mark.parametrize("extra", [None, "COPYRIGHT", "missing-mit", "restricted", "LGPL"])
def test_reviewed_alternative_preserves_all_candidate_validation(tmp_path, extra):
    args = _rust_args(tmp_path)
    apache = next(r["text"] for r in REVIEWED if r["identifier"] == "Apache-2.0")
    files = {"LICENSE-MIT": L.MIT_CANONICAL_BODY, "LICENSE-APACHE": apache}
    if extra == "COPYRIGHT":
        files["COPYRIGHT"] = "Except as otherwise noted in individual files; some code derives from libstd."
    elif extra == "missing-mit":
        del files["LICENSE-MIT"]
    elif extra == "restricted":
        files["LICENSE-APACHE"] += "\nCommercial use is prohibited."
    elif extra == "LGPL":
        files["vendor/COPYING"] = LGPL_TEXT
    digest = _crate(Path(args.cargo_registry_cache), "dep", "1.0", files)
    lock = Path(args.cargo_lock_workspace)
    content = lock.read_text()
    old = L.tomllib.loads(content)["package"][0]["checksum"]
    lock.write_text(content.replace(old, digest))
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == ("PERMISSIVE" if extra is None else "HOLD")
    if extra == "missing-mit":
        assert row["elected_text_present_in_artifact"] is False


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


def _native_wheel(
    path: Path,
    *,
    native_license: str | None,
    bind_notice: bool = True,
    license_body: str = "",
) -> str:
    """Write the minimal shape used by real wheels with vendored native notices."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("pkg-1.0.dist-info/METADATA", "Metadata-Version: 2.4\nName: pkg\nVersion: 1.0\n\n")
        zf.writestr("pkg-1.0.dist-info/licenses/LICENSE", MIT_TEXT)
        zf.writestr("pkg.libs/libexample.so", b"synthetic native bytes")
        if bind_notice:
            zf.writestr(
                "pkg-1.0.dist-info/licenses/NOTICE",
                "Name: native-component\nFiles: pkg.libs/libexample.so\n"
                f"License: {native_license or ''}\n"
                + "\n".join(f" {line}" if line else " ." for line in license_body.splitlines())
                + "\n",
            )
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


@pytest.mark.parametrize(
    "native_license",
    [
        "LGPL-2.1-or-later",
        "GPL-3.0-or-later WITH GCC-exception-3.1",
        "AGPL-3.0-only",
        "LicenseRef-unreviewed",
        None,
    ],
)
def test_vendored_native_nonpermissive_or_unknown_license_holds(tmp_path, native_license):
    """Permissive package metadata cannot hide a bound native component policy failure."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    wheel = Path(args.pypi_artifact_dir) / "pkg-1.0-py3-none-any.whl"
    digest = _native_wheel(wheel, native_license=native_license)
    Path(args.uv_lock).write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{digest}" }}]\n'
    )
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert row["vendored_native_license_holds"]


def test_vendored_native_without_bound_notice_holds(tmp_path):
    """A native wheel member with no matching notice stanza is not silently accepted."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    wheel = Path(args.pypi_artifact_dir) / "pkg-1.0-py3-none-any.whl"
    digest = _native_wheel(wheel, native_license=None, bind_notice=False)
    Path(args.uv_lock).write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{digest}" }}]\n'
    )
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert "no bound notice stanza" in " ".join(row["vendored_native_license_holds"])


def test_vendored_native_verified_permissive_election_remains_allowed(tmp_path):
    """A hash-bound native component may elect its verified permissive alternative."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    wheel = Path(args.pypi_artifact_dir) / "pkg-1.0-py3-none-any.whl"
    digest = _native_wheel(
        wheel,
        native_license="MIT OR LGPL-2.1-or-later",
        license_body=MIT_TEXT,
    )
    Path(args.uv_lock).write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{digest}" }}]\n'
    )
    (row,) = L.python_inventory(args, [])
    assert row["vendored_native_license_holds"] == []
    assert row["license_class"] == "PERMISSIVE"


@pytest.mark.parametrize("native_license", ["MIT", "MIT OR LGPL-2.1-or-later"])
def test_vendored_native_label_without_verified_grant_holds(tmp_path, native_license):
    """A hash-bound SPDX label alone does not verify a native component grant."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    wheel = Path(args.pypi_artifact_dir) / "pkg-1.0-py3-none-any.whl"
    digest = _native_wheel(wheel, native_license=native_license)
    Path(args.uv_lock).write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{digest}" }}]\n'
    )
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert "no verified elected grant" in " ".join(row["vendored_native_license_holds"])


@pytest.mark.parametrize("native_license", ["MIT", "MIT OR LGPL-2.1-or-later"])
def test_vendored_native_additional_condition_holds(tmp_path, native_license):
    """Canonical MIT text plus a commercial-use restriction is not permissive evidence."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    wheel = Path(args.pypi_artifact_dir) / "pkg-1.0-py3-none-any.whl"
    digest = _native_wheel(
        wheel,
        native_license=native_license,
        license_body=MIT_TEXT + "\nAdditional restriction: commercial use prohibited.",
    )
    Path(args.uv_lock).write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{digest}" }}]\n'
    )
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert "no verified elected grant" in " ".join(row["vendored_native_license_holds"])


@pytest.mark.parametrize("native_license", ["MIT", "MIT OR LGPL-2.1-or-later"])
def test_vendored_native_pre_license_condition_holds(tmp_path, native_license):
    """A condition before License is part of the notice and invalidates verification."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    wheel = Path(args.pypi_artifact_dir) / "pkg-1.0-py3-none-any.whl"
    _native_wheel(wheel, native_license=native_license, license_body=MIT_TEXT)
    result = io.BytesIO()
    with zipfile.ZipFile(wheel) as old, zipfile.ZipFile(result, "w") as new:
        for entry in old.infolist():
            data = old.read(entry.filename)
            if entry.filename.endswith("/NOTICE"):
                data = data.replace(
                    b"License: ",
                    b"Description: Additional restriction: commercial use prohibited\nLicense: ",
                    1,
                )
            new.writestr(entry.filename, data)
    wheel.write_bytes(result.getvalue())
    digest = hashlib.sha256(result.getvalue()).hexdigest()
    Path(args.uv_lock).write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{digest}" }}]\n'
    )
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert "no verified elected grant" in " ".join(row["vendored_native_license_holds"])


@pytest.mark.parametrize("native_license", ["0BSD", "CC0-1.0", "BSD-3-Clause-Open-MPI"])
def test_vendored_native_unfingerprinted_label_without_grant_holds(tmp_path, native_license):
    """A native component needs scope-bound permission evidence even for these ids."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    wheel = Path(args.pypi_artifact_dir) / "pkg-1.0-py3-none-any.whl"
    digest = _native_wheel(wheel, native_license=native_license)
    Path(args.uv_lock).write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{digest}" }}]\n'
    )
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert "no verified elected grant" in " ".join(row["vendored_native_license_holds"])


@pytest.mark.parametrize(
    "needle,replacement",
    [
        (b"Name: native-component", b"Unscoped permission statement\nName: native-component"),
        (b"Files: pkg.libs/libexample.so", b"Files: pkg.libs/libexample.so\n continuation"),
        (b"License: MIT", b"Intermediate statement\nLicense: MIT"),
        (b"License: MIT", b"Name: duplicate\nFiles: pkg.libs/libexample.so\nLicense: MIT"),
        (b" MIT License", b" MIT License\nUnindented middle statement"),
        (b" SOFTWARE.\n", b" SOFTWARE.\nTrailing statement\n"),
    ],
)
def test_vendored_native_notice_rejects_unparsed_nonblank_bytes(tmp_path, needle, replacement):
    """Every nonblank notice byte must belong to the verified header/body grammar."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    wheel = Path(args.pypi_artifact_dir) / "pkg-1.0-py3-none-any.whl"
    _native_wheel(wheel, native_license="MIT", license_body=MIT_TEXT)
    result = io.BytesIO()
    with zipfile.ZipFile(wheel) as old, zipfile.ZipFile(result, "w") as new:
        for entry in old.infolist():
            data = old.read(entry.filename)
            if entry.filename.endswith("/NOTICE"):
                assert needle in data
                data = data.replace(needle, replacement, 1)
            new.writestr(entry.filename, data)
    wheel.write_bytes(result.getvalue())
    digest = hashlib.sha256(result.getvalue()).hexdigest()
    Path(args.uv_lock).write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{digest}" }}]\n'
    )
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert row["vendored_native_license_holds"]


@pytest.mark.parametrize("case", ["separate-condition", "unmatched-prefix", "invalid-second-match"])
def test_vendored_native_all_candidate_files_and_stanzas_must_verify(tmp_path, case):
    """Package acceptance consumes every candidate file and every native stanza."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    wheel = Path(args.pypi_artifact_dir) / "pkg-1.0-py3-none-any.whl"
    _native_wheel(
        wheel,
        native_license="MIT OR LGPL-2.1-or-later",
        license_body=MIT_TEXT,
    )
    result = io.BytesIO()
    with zipfile.ZipFile(wheel) as old, zipfile.ZipFile(result, "w") as new:
        for entry in old.infolist():
            data = old.read(entry.filename)
            if entry.filename.endswith("/NOTICE") and case == "unmatched-prefix":
                data = (
                    b"Unparsed package-wide condition\n"
                    b"Name: unrelated\nFiles: unused.so\nLicense: MIT\n .\n"
                    + data
                )
            elif entry.filename.endswith("/NOTICE") and case == "invalid-second-match":
                data += (
                    b"Name: extra\nFiles: pkg.libs/libexample.so\nLicense: MIT\n"
                    b" Unverified condition\n"
                )
            new.writestr(entry.filename, data)
        if case == "separate-condition":
            new.writestr(
                "pkg-1.0.dist-info/licenses/NOTICE-CONDITIONS",
                "Unparsed package-wide condition\n",
            )
    wheel.write_bytes(result.getvalue())
    digest = hashlib.sha256(result.getvalue()).hexdigest()
    Path(args.uv_lock).write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{digest}" }}]\n'
    )
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert row["vendored_native_license_holds"]
    assert "not fully verified" in " ".join(row["hold_reasons"])


def test_vendored_native_unmatched_verified_stanza_holds_package(tmp_path):
    """A valid grant for a nonexistent binary is still unconsumed artifact evidence."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    wheel = Path(args.pypi_artifact_dir) / "pkg-1.0-py3-none-any.whl"
    _native_wheel(wheel, native_license="MIT", license_body=MIT_TEXT)
    result = io.BytesIO()
    with zipfile.ZipFile(wheel) as old, zipfile.ZipFile(result, "w") as new:
        for entry in old.infolist():
            data = old.read(entry.filename)
            if entry.filename.endswith("/NOTICE"):
                body = "\n".join(f" {line}" if line else " ." for line in MIT_TEXT.splitlines())
                data += (
                    b"Name: absent-component\nFiles: absent.so\nLicense: MIT\n"
                    + body.encode()
                    + b"\n"
                )
            new.writestr(entry.filename, data)
    wheel.write_bytes(result.getvalue())
    digest = hashlib.sha256(result.getvalue()).hexdigest()
    Path(args.uv_lock).write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{digest}" }}]\n'
    )
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert "not fully verified" in " ".join(row["hold_reasons"])


def _replace_wheel(args, writer):
    wheel = Path(args.pypi_artifact_dir) / "pkg-1.0-py3-none-any.whl"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        writer(zf)
    wheel.write_bytes(buf.getvalue())
    digest = hashlib.sha256(buf.getvalue()).hexdigest()
    Path(args.uv_lock).write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{digest}" }}]\n'
    )


def test_metadata_declared_custom_license_file_is_consumed(tmp_path):
    """A nonstandard License-File cannot hide a package-wide restriction."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)

    def writer(zf):
        zf.writestr("pkg-1.0.dist-info/METADATA", "Metadata-Version: 2.4\nName: pkg\nVersion: 1.0\nLicense-File: terms.txt\n\n")
        zf.writestr("pkg-1.0.dist-info/licenses/LICENSE", MIT_TEXT)
        zf.writestr("pkg-1.0.dist-info/licenses/terms.txt", "Commercial use is prohibited.\n")

    _replace_wheel(args, writer)
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert "terms.txt" in " ".join(row["hold_reasons"])


@pytest.mark.parametrize(
    "header",
    [
        "License-File:terms.txt",
        "license-file: terms.txt",
        "License-File:\tterms.txt",
        "License-File:\n terms.txt",
    ],
)
def test_metadata_license_file_uses_standard_header_parsing(tmp_path, header):
    """Whitespace, case, and folding cannot make a declared file disappear."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)

    def writer(zf):
        zf.writestr(
            "pkg-1.0.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: pkg\nVersion: 1.0\n{header}\n\n",
        )
        zf.writestr("pkg-1.0.dist-info/licenses/LICENSE", MIT_TEXT)
        zf.writestr("pkg-1.0.dist-info/licenses/terms.txt", "Commercial use is prohibited.\n")

    _replace_wheel(args, writer)
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert "terms.txt" in " ".join(row["hold_reasons"])


def test_metadata_parser_defect_holds(tmp_path):
    """Malformed metadata cannot fall back to a permissive partial parse."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)

    def writer(zf):
        zf.writestr(
            "pkg-1.0.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: pkg\nMalformed header\nLicense-File: terms.txt\n\n",
        )
        zf.writestr("pkg-1.0.dist-info/licenses/LICENSE", MIT_TEXT)
        zf.writestr("pkg-1.0.dist-info/licenses/terms.txt", MIT_TEXT)

    _replace_wheel(args, writer)
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert "parser reported defects" in " ".join(row["vendored_native_license_holds"])


@pytest.mark.parametrize("declared", ["missing.txt", "..", "../LICENSE", "/LICENSE", r"dir\LICENSE"])
def test_invalid_or_missing_metadata_license_file_holds(tmp_path, declared):
    args = _python_args(tmp_path, expression="MIT", license_text=False)

    def writer(zf):
        zf.writestr("pkg-1.0.dist-info/METADATA", f"Metadata-Version: 2.4\nName: pkg\nVersion: 1.0\nLicense-File: {declared}\n\n")
        zf.writestr("pkg-1.0.dist-info/licenses/LICENSE", MIT_TEXT)

    _replace_wheel(args, writer)
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert row["vendored_native_license_holds"]


def test_duplicate_wheel_member_holds_before_last_entry_can_mask_first(tmp_path):
    args = _python_args(tmp_path, expression="MIT", license_text=False)

    def writer(zf):
        zf.writestr("pkg-1.0.dist-info/METADATA", "Metadata-Version: 2.4\nName: pkg\nVersion: 1.0\n\n")
        zf.writestr("pkg-1.0.dist-info/licenses/LICENSE", MIT_TEXT)
        zf.writestr("pkg-1.0.dist-info/licenses/NOTICE", "Commercial use is prohibited.\n")
        zf.writestr("pkg-1.0.dist-info/licenses/NOTICE", MIT_TEXT)

    _replace_wheel(args, writer)
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert "duplicate wheel member" in " ".join(row["vendored_native_license_holds"])


@pytest.mark.parametrize("pattern,expected", [("pkg.libs/libexample.so", "HOLD"), ("pkg.libs/libexample*.so", "PERMISSIVE")])
def test_native_files_scope_is_exact_unless_notice_declares_glob(tmp_path, pattern, expected):
    args = _python_args(tmp_path, expression="MIT", license_text=False)

    def writer(zf):
        zf.writestr("pkg-1.0.dist-info/METADATA", "Metadata-Version: 2.4\nName: pkg\nVersion: 1.0\n\n")
        zf.writestr("pkg-1.0.dist-info/licenses/LICENSE", MIT_TEXT)
        zf.writestr("pkg.libs/libexample_unrelated.so", b"synthetic")
        body = "\n".join(f" {line}" if line else " ." for line in MIT_TEXT.splitlines())
        zf.writestr("pkg-1.0.dist-info/licenses/NOTICE", f"Name: component\nFiles: {pattern}\nLicense: MIT OR LGPL-2.1-or-later\n{body}\n")

    _replace_wheel(args, writer)
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == expected
    assert bool(row["vendored_native_license_holds"]) is (expected == "HOLD")


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


def test_copyright_header_cannot_hide_an_additional_condition(tmp_path):
    """A copyright-looking line is unverified content, not removable syntax."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    tainted = MIT_TEXT.replace(
        "MIT License\n\n",
        "MIT License\n\nCopyright (c) 2026 Test Author Use is permitted solely for academic research\n\n",
        1,
    )
    digest = _crate(cache, "dep", "1.0", {"LICENSE": tainted})
    Path(args.cargo_lock_workspace).write_text(
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_binding).write_text(Path(args.cargo_lock_workspace).read_text())
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert row["elected_text_present_in_artifact"] is False


@pytest.mark.parametrize(
    "header",
    [
        "Copyright (c) 2026 Test Author",
        "Copyright (c) 2026 Test Author Commercial use is prohibited",
    ],
)
def test_unreviewed_copyright_header_is_hold(tmp_path, header):
    """No inferred author-name grammar may erase an unreviewed header line."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    text = MIT_TEXT.replace("MIT License\n\n", f"MIT License\n\n{header}\n\n", 1)
    digest = _crate(cache, "dep", "1.0", {"LICENSE": text})
    lock = (
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_workspace).write_text(lock)
    Path(args.cargo_lock_binding).write_text(lock)
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert row["elected_text_present_in_artifact"] is False


def test_canonical_mit_body_without_unreviewed_header_remains_permissive(tmp_path):
    """The exact canonical body remains the bounded positive control."""
    args = _rust_args(tmp_path)
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "PERMISSIVE"
    assert row["elected_text_present_in_artifact"] is True


def test_prior_copyright_header_fixture_now_holds(tmp_path):
    """Record the intentional PERMISSIVE-to-HOLD change for the old fixture."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    digest = _crate(cache, "dep", "1.0", {"LICENSE": MIT_WITH_HEADER})
    lock = (
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_workspace).write_text(lock)
    Path(args.cargo_lock_binding).write_text(lock)
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert row["elected_text_present_in_artifact"] is False


@pytest.mark.parametrize(
    "other_name,other_text",
    [
        ("COPYING", MIT_TEXT + "\nAdditional condition: Use is permitted solely for academic research.\n"),
        ("NOTICE", "Apache License Version 2.0"),
    ],
)
def test_verified_mit_cannot_hide_an_unverified_candidate_file(tmp_path, other_name, other_text):
    """Every collected candidate is independently verified or the row is held."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    digest = _crate(cache, "dep", "1.0", {"LICENSE": MIT_TEXT, other_name: other_text})
    Path(args.cargo_lock_workspace).write_text(
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_binding).write_text(Path(args.cargo_lock_workspace).read_text())
    (row,) = L.rust_inventory(args, [])
    assert row["elected_text_present_in_artifact"] is True
    assert row["license_class"] == "HOLD"
    assert other_name in " ".join(row["hold_reasons"])

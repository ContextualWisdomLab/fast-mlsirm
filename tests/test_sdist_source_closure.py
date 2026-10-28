"""Check the actual sdist against the compile-time resume source catalog.

Run directly with an archive produced by ``maturin sdist``. This deliberately
checks bytes, not just the manifest glob; wheel compilation remains the end-to-end
gate (``python -m build`` builds its wheel from the sdist by default).
"""
import re
import sys
import tarfile
from pathlib import Path


def check_sdist(archive: Path) -> int:
    """Require every catalog input, at its original path and with original bytes."""
    root = Path(__file__).resolve().parents[1]
    catalog = root / "crates/mlsirm-core/src/mask_grm_resume_sources.rs"
    paths = re.findall(r'out\.insert\("([^"]+)"', catalog.read_text())
    assert paths, "empty source catalog"
    with tarfile.open(archive) as tar:
        prefixes = {m.name.split('/')[0] for m in tar.getmembers()}
        assert len(prefixes) == 1, prefixes
        prefix = prefixes.pop()
        failures = []
        for path in paths:
            try:
                member = tar.extractfile(f"{prefix}/{path}")
            except KeyError:
                failures.append(f"missing: {path}")
                continue
            if member is None or member.read() != (root / path).read_bytes():
                failures.append(f"changed: {path}")
        assert not failures, '\n'.join(failures)
    return len(paths)


def test_out_of_crate_catalog_inputs_are_explicit_sdist_includes():
    """Keep external compile-time inputs in the declared sdist closure."""
    import tomllib

    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    included = set()
    for entry in config["tool"]["maturin"]["include"]:
        if entry["format"] == "sdist":
            included.update(p.relative_to(root).as_posix() for p in root.glob(entry["path"]) if p.is_file())
    catalog = root / "crates/mlsirm-core/src/mask_grm_resume_sources.rs"
    paths = set(re.findall(r'out\.insert\("([^"]+)"', catalog.read_text()))
    external = {p for p in paths if not p.startswith("crates/") and p != "Cargo.toml"}
    assert external
    assert not external - included, sorted(external - included)


if __name__ == "__main__":
    print(f"PASS: {check_sdist(Path(sys.argv[1]))} catalog inputs preserved")

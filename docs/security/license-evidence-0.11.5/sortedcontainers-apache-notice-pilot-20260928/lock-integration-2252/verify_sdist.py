"""Verify the exact PR #2252 s1 source archive scope."""

import hashlib
import tarfile
from pathlib import Path


archive = Path(__file__).with_name("fast_mlsirm-0.11.4.tar.gz")
assert hashlib.sha256(archive.read_bytes()).hexdigest() == (
    "25dd09d899ff0dc1103b3917c7d98ff71132afedcce88fcc523e4b6fc0108a4f"
)
with tarfile.open(archive) as source:
    names = source.getnames()
    assert len(names) == 529
    assert not any("third_party" in Path(name).parts for name in names)
    assert "fast_mlsirm-0.11.4/LICENSE" in names
    assert "fast_mlsirm-0.11.4/pyproject.toml" in names
print("PR #2252 sdist scope verified: 529 members, no third_party wheel")

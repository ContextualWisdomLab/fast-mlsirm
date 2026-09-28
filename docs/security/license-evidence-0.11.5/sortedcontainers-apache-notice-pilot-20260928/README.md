# Sortedcontainers Apache notice pilot

**Verdict: local pilot only. Python HOLD remains.** This does not alter the
official wheel, the release lock, or any published fast-mlsirm artifact.

The [PyPI 2.4.0 release](https://pypi.org/project/sortedcontainers/2.4.0/)
lists the pure-Python wheel in this directory with SHA256
`a163dcaede0f1c021485e957a39245190e74249897e2ae4b2aa38595db237ee0`.
The original `sortedcontainers-2.4.0.dist-info/LICENSE` is preserved verbatim
(SHA256 `1db7cae7fce6452e2e608e401a0f953e0133e4c2d75db69fb8ae851d2086f5b6`).
It names Apache 2.0 and points to the Apache license, but contains only its
short application header. The [Apache Software Foundation's complete text](https://www.apache.org/licenses/LICENSE-2.0.txt)
is retained as `LICENSE-2.0.txt` (11,358 bytes, SHA256
`cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`).

`repair_sortedcontainers.py` adds that complete text as
`sortedcontainers-2.4.0.dist-info/LICENSE-APACHE-2.0.txt` and rebuilds only
`RECORD`. Every original package and metadata member, including the short
header, stays byte-identical. The two independent repacks in `A/` and `B/`
are byte-identical, SHA256
`7fc4ec6d6c59f25bff4c90950c3b2c08da3d6977a21176c3d0e6ff93f1b55a00`.
Both ZIPs and RECORD rows verify. Re-run without overwriting existing outputs:

```bash
shasum -a 256 -c SHA256SUMS
python3 repair_sortedcontainers.py
```

The unchanged current validator blob `fdbc78212d16027c0f8954c706e5b6325622ecba`
at main `3a05e8a0` recognizes the new full text as Apache-2.0. It still
marks the original short header unverified. A **simulation only** added the
repacked wheel SHA, original member path and original member SHA to
`REVIEWED_PYTHON_COMPANIONS` with `Apache-2.0`; then both candidate files
verified. No such mapping was committed or used by release admission.

An isolated `uv` CPython 3.12.14 environment installed `A/` offline and
passed a `SortedList([3, 1, 2])` ordering smoke; the installed full notice
was 11,358 bytes. This does not prove the complete package test suite,
trusted source compilation, reviewed redistribution decision, locked release
input, or the published twelve-wheel matrix. The original Python HOLD of
three remains until those separate release requirements are accepted.

## Fresh source-build repeat

The official PyPI sdist `sortedcontainers-2.4.0.tar.gz` is also retained here
(SHA256 `25caa5a06cc30b6b83d11423433f65d1f9d76c4c6a0c90e3379eaa43b9bfdb88`).
`source-build/build_source_repeat.sh` checks the source, original wheel and
ASF notice hashes, extracts the source three times, and builds with local
CPython 3.12.14, hash-pinned setuptools 80.9.0 and wheel 0.45.1, offline
without build isolation. A uses unmodified source. B and C add only the
complete ASF text as a source-root `LICENSE-APACHE-2.0.txt`; setuptools places
it in wheel `dist-info/licenses/` automatically. `SOURCE_DATE_EPOCH=1621202600`
is fixed. The script refuses an existing output directory:

```bash
shasum -a 256 -c source-build/SHA256SUMS
bash source-build/build_source_repeat.sh /tmp/new-sortedcontainers-source-repeat
```

The tested replay produced source A wheel SHA256 `436e950b98f6809755f5943d10e83eab417d608f9d5dd067e64edcce063fb926`;
B and C were byte-identical at SHA256 `5ed427f4b9542cce09e258fa66b0d8fd6f1e4abab08f2e8e9621cfc38d6fac0c`.
The script checked each wheel's complete ZIP RECORD, every package Python
module against the official wheel, the original LICENSE bytes, and the added
full notice. An isolated offline CPython 3.12.14 install of B passed
SortedList, SortedSet and SortedDict ordering smoke and installed the exact
11,358-byte notice. This source-backed local candidate still has no reviewed
release lock election, complete package test result, rights acceptance, or
published twelve-wheel proof. The original Python HOLD remains.

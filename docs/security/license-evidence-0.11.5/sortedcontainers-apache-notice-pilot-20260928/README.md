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

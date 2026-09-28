# Source-matched upstream tests

The upstream `v2.4.0` tag peels to Git commit
`a1f52d6713dd2c2713a881d4f4d86ed68ff71cab`. Its commit-addressed
source archive has SHA256 `db312aac77ad185dc48f1e79817556588c675e5060785b7461e79c5d9c0deac2`.
All four `sortedcontainers/*.py` members match the official 2.4.0 sdist
byte for byte. The archive was safely extracted to an isolated temporary
directory; the test and documentation trees were copied without the upstream
package tree. The installed import path was the local source-built B wheel,
SHA256 `5ed427f4b9542cce09e258fa66b0d8fd6f1e4abab08f2e8e9621cfc38d6fac0c`.

In a fresh CPython 3.12.14 `uv` environment with pytest 9.1.1, upstream
coverage tests passed 291/291, stress tests 6/6, documentation examples 1/1,
and installed-package module doctests 66/66. The four original logs and their
hashes, installed test-tool versions, source URL and module hashes are in
`receipt.json`. The test commands, from the test-only root, were:

```bash
.venv/bin/python -m pytest -q --override-ini=testpaths=tests tests/test_coverage_*.py
.venv/bin/python -m pytest -q --override-ini=testpaths=tests tests/test_stress_*.py
.venv/bin/python -m pytest -q --override-ini=testpaths=docs --doctest-glob='*.rst' docs
.venv/bin/python -m pytest -q --doctest-modules .venv/lib/python3.12/site-packages/sortedcontainers
```

These tests cover the source-backed local candidate. The release still locks
the official wheel; no reviewed companion election, final rights decision or
published fast-mlsirm twelve-wheel acceptance has occurred. The original
Python HOLD remains.

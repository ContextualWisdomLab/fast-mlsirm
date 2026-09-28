# Sortedcontainers 2.4.0 development wheel

`sortedcontainers-2.4.0-py3-none-any.whl` is a source-built candidate for the
locked development and fuzz dependency. Its SHA256 is
`5ed427f4b9542cce09e258fa66b0d8fd6f1e4abab08f2e8e9621cfc38d6fac0c`.
The official PyPI 2.4.0 sdist has SHA256
`25caa5a06cc30b6b83d11423433f65d1f9d76c4c6a0c90e3379eaa43b9bfdb88`.
All four package Python modules match that sdist and the official PyPI wheel
byte for byte. The source build adds only the complete Apache-2.0 terms from
the Apache Software Foundation; the original short Apache application header
is preserved. The complete notice has SHA256
`cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`.

The immutable [source-build and upstream-test receipt](https://github.com/ContextualWisdomLab/fast-mlsirm/tree/f92a9e924556b9ab8eef8f17465340f053d06245/docs/security/license-evidence-0.11.5/sortedcontainers-apache-notice-pilot-20260928)
contains the official inputs, hash-pinned backend requirements, replay script,
two identical fresh builds, and 364 passing upstream tests. `uv.lock` binds
the exact wheel hash and `tools/license_inventory.py` requires both license
members to verify. No release artifact or PyPI package has been published from
this candidate. Other Python and Cargo license HOLDs remain separate.

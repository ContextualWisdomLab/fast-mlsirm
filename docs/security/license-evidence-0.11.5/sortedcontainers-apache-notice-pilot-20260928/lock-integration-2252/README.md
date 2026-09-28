# PR #2252 lock candidate and s1 package scope

Exact PR head `3d3d7ce35d7107df62d359b33b5bc90392ee8edf` on fast main
`3a05e8a0dfec52977a2d2aa3dce70d2de12b0275` adds a hash-bound local
sortedcontainers development wheel. The candidate wheel SHA256 is
`5ed427f4b9542cce09e258fa66b0d8fd6f1e4abab08f2e8e9621cfc38d6fac0c`.
An isolated one-package inventory with the exact uv lock hash, official PyPI
metadata and candidate wheel returned `PERMISSIVE`, Apache-2.0 and zero row
holds. The exact-head license-inventory test file passed 247 tests; four
release-contract files passed 88; `uv lock --check`, locked CPython 3.12 dev
sync, Ruff and a refreshed fixable HIGH/CRITICAL Trivy scan passed. The
same-head hosted Semgrep, osv-scan, dependency-review and trivy-fs checks
were SUCCESS when inspected before this receipt. Other hosted checks were
queued, not successful.

On `s1.cluster.seonghobae.me`, a fresh isolated clone at that exact commit
built the retained source archive with maturin 1.15.0, Cargo 1.97.1 and
`CARGO_NET_OFFLINE=true`. The 1,687,060-byte tarball SHA256 is
`25dd09d899ff0dc1103b3917c7d98ff71132afedcce88fcc523e4b6fc0108a4f`.
It has 529 members, includes the project LICENSE and pyproject.toml, and has
no `third_party` member. Verify the retained archive with:

```bash
python3 verify_sdist.py
```

This is a 0.11.4 package-scope test of a development dependency change. It
does not prove a 0.11.5 release, original Python three-HOLD closure, Cargo
rights, native twelve-leg validation or published twelve-wheel acceptance.

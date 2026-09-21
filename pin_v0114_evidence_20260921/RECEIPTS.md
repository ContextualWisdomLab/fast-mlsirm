# Local receipts — late-life PR #247 (pin) and PR #246 (p_wald), 2026-09-21

Author: Claude session term_9464f3b6. These are local artifacts only; there is no hosted CI state here (231d owns CI queries).

## PR #247 — pin fast-mlsirm to `e44b52d851596910dc7a8fa3f174f2aff03872fa` (tag v0.11.4)

Branch `seonghobae/pin-fast-mlsirm-v0.11.4`, head `2a27b7f078284de09c3ee85385da5ca88a0b057c`. It changes `requirements.txt` and the README pin sentence only.

| Artifact (this dir) | sha256 | Check |
|---|---|---|
| `fast_mlsirm-0.11.4.tar.gz` | `761f13783484ba7ef82c810c9c740f113d20cf52f879556056dfae9b36e39731` | equals the PyPI JSON digest (`urls.txt`) |
| `fast_mlsirm-0.11.4-cp312-…universal2.whl` | `d026033bc4f649534c1f4d9516c7cd081fc064bd292a4e2d302acdb9f2ee060e` | equals the PyPI JSON digest |

Results observed in this session:
- sdist vs `git archive e44b52d8`: 307 non-`PKG-INFO` files, **0 differing, 0 not in tag**.
- Wheel `fast_mlsirm/*.py` vs the tag's `python/fast_mlsirm`: identical. Only the compiled `_core*.so` differs, and some files exist only in the wheel.
- Isolated uv venv (py3.12) with `fast-mlsirm==0.11.4` from the PyPI index: `direct_url.json` is absent (installer uv), and the installed `_core.cpython-312-darwin.so` sha256 prefix is `882e189da7dc9933`, the same as the `_core` inside the wheel above.
- The research repo's `fast_mlsirm` usages (`usages.txt`, 24 distinct lines, 33 names) all resolve in 0.11.4: **0 missing**.
- `pytest tests/` on the pin branch with `PYTHONPATH=analysis`: **7 passed, 2 skipped**. The skips are `test_mc_stop_library_contract`, blocked on fast-mlsirm#2013, which is unreleased and out of scope.
- The PR #246 test in the same venv: 1 passed, 1 skipped (contract absent on that branch).

Re-verify locally (no network except `git archive` from the local fast-mlsirm clone):

```sh
D=pin_v0114_evidence_20260921
shasum -a 256 $D/*.tar.gz $D/*.whl
T=$(mktemp -d); mkdir $T/tag; git -C <fast-mlsirm clone> archive e44b52d851596910dc7a8fa3f174f2aff03872fa | tar -x -C $T/tag
tar -xzf $D/fast_mlsirm-0.11.4.tar.gz -C $T && cd $T/fast_mlsirm-0.11.4 && \
  find . -type f ! -name PKG-INFO | while read f; do cmp -s "$f" "$T/tag/$f" || echo "DIFF $f"; done
```

## PR #246 — `p_wald_two_sided` producer (contract G3)

Branch `seonghobae/h1h5-p-wald-two-sided`, head `b46d8e2f9c578f2ca67681297f1f410e96878d6a`. Body: `PR246_body.md`. Local results: `origin/main` driver 1 failed (`KeyError: 'p_wald_two_sided'`); branch 1 passed + 1 skipped; 2 passed with a temporary copy of the contract (bc48fcf, sha256 `7da18b6d…e464a9`).

## Not covered

No numerical acceptance (G1/G7), no unreleased W/E or MC-stop APIs, no R tests.

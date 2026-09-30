# PerFit package-comparator fixture provenance

Status: **package comparator / test-fixture provenance only**

This file resolves the repository-local provenance reference used by `tests/unit/personfit_np_tests.rs`. It does not establish primary-source equation authority, a universal person-fit cutoff, or validity for any operational use.

## Immutable package identity

The comparator is the CRAN mirror repository `cran/PerFit` at commit `c9df433cba3d7b03d16284e832d55785cb90464c` (`version 1.4.7`, CRAN publication date 2025-04-02). The complete-data dichotomous port was checked against the package implementation in:

- `R/G.R`;
- `R/Gnormed.R`;
- `R/NCI.R`;
- `R/U3.R`;
- `R/ZU3.R`;
- `R/C.Sato.R`;
- `R/Cstar.R`;
- `R/Accessory.R` (`final.PFS`);
- `R/SanityChecks.R` (`Sanity.prv`).

Package-source agreement is software-comparator evidence. It must not be promoted to evidence that the original statistical publication was directly read or that a statistic is valid for a particular decision.

## Main deterministic fixture

The Rust fixture records a deterministic `numpy.default_rng(2033)` construction with `N=12`, `I=8`. Row 9 is the planted reversed pattern, row 10 is all zero, and row 11 is all one. The fixture deliberately contains tied item proportions so the package-compatible stable original-column-index tie break is exercised.

The exact response rows are tracked in `tests/unit/personfit_np_tests.rs`; that executable source is the canonical byte-level fixture. This document records provenance and mutation-sensitive expectations rather than duplicating the response matrix as a second authority.

## Mutation-sensitive expected values

The Rust tests retain seven focused arithmetic mutations and the expected observable change that kills each mutation:

| ID | Mutation | Pinned consequence |
|---|---|---|
| MU1 | reverse the package-compatible item ordering | `g[9]` changes from 10 to 8 |
| MU2 | use `NC * (I - NC + 1)` as the Gnormed denominator | `gnormed[1]` changes from 0.625 to 0.5 |
| MU3 | reverse the `NCI = 1 - 2 * Gnormed` transformation | the pinned NCI sign/value changes |
| MU4 | use the wrong U3 endpoint sum in the numerator | `u3[1]` changes from the pinned package-comparator value |
| MU5 | omit the square root in the ZU3 variance normalization | `zu3[9]` changes to the documented large erroneous value |
| MU6 | use a non-Guttman denominator vector for Sato C | the pinned `c_sato` outputs collapse to `NaN` |
| MU7 | use `sfp + slp` in the C* denominator | `cstar[9]` changes from the pinned package-comparator value |

The exact expected binary64 values and failure messages live next to the executable tests. This file does **not** claim that an external mutation-testing tool log is retained: the repository currently preserves the mutation definitions and their test-observable kill conditions, not an independently archived mutation-run artifact.

## Scope and scientific boundary

The comparator fixture covers complete dichotomous `{0,1}` data only. Missing-value imputation and polytomous variants are out of scope. CPU `f64` is the reference arithmetic for this scalar kernel.

Primary-source equation, ordering, tie, perfect-row, and degeneracy evidence is tracked separately under issue #1790. If a directly read primary source disagrees with PerFit or current Rust behavior, record that discrepancy before changing arithmetic; package compatibility alone does not authorize a scientific formula decision.

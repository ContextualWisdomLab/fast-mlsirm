# Cross-engine conformance manifest

## Purpose

`fast_mlsirm.conformance` is the reusable provenance boundary for issue #1077.
It records a capability inventory, comparison layer, independent engine
identity, parameter-mapping fingerprint, preregistered tolerances, and
reproducibility metadata without importing or executing an external engine.

The manifest is evidence metadata, not a correctness oracle. A row marked
`covered` means that a declared independent comparison scope and mapping were
recorded; it does not mean that construct validity, fairness, or high-stakes
approval has been established.

## Invariants

- `protected_main_sha` in the manifest and provenance must be identical.
- Capability identities, engine identities, and tolerance estimand identities
  are unique within their respective collections.
- `covered` requires an independent engine and a mapping fingerprint.
- `partially_covered`, `no_independent_engine`, `not_comparable`, and `planned`
  remain explicit states; absence of an engine never becomes a pass.
- Capability ordering is canonicalized before the manifest SHA-256 fingerprint
  is computed.
- External packages, proprietary binaries, raw participant data, direct PII,
  and restricted item content remain outside the reusable contract.

## Ownership and next slice

`fast-mlsirm` owns this source-free contract and its validation. A separate,
license-compliant harness owns equation fixtures, parameter alignment, engine
execution, normalized outputs, and accessible evidence rendering. The harness
must remain optional and isolated from the runtime wheel, Rust workspace, and
ordinary production CI. Numerical equations and estimators remain Rust-owned.

This slice does not claim that the protected public capability inventory is
complete, nor does it add R, Stan, `mirt`, TAM, `ltm`, or `eRm` as dependencies.
The next implementation slice should populate the manifest from a reviewed
capability inventory and emit result artifacts whose raw and normalized
fingerprints are bound to this metadata.

## Research basis — APA 7

Chalmers, R. P. (2012). *mirt: A multidimensional item response theory
package for the R environment*. Journal of Statistical Software, 48(6),
1–29. https://doi.org/10.18637/jss.v048.i06

Mair, P., & Hatzinger, R. (2007). Extended Rasch modeling: The eRm package
for the application of IRT models in R. *Journal of Statistical Software,
20*(9), 1–20. https://doi.org/10.18637/jss.v020.i09

Morris, T. P., White, I. R., & Crowther, M. J. (2019). Using simulation
studies to evaluate statistical methods. *Statistics in Medicine, 38*(11),
2074–2102. https://doi.org/10.1002/sim.8086

Rizopoulos, D. (2006). ltm: An R package for latent variable modeling and
item response analysis. *Journal of Statistical Software, 17*(5), 1–25.
https://doi.org/10.18637/jss.v017.i05

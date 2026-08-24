# Cross-engine conformance inventory

This document defines the first source-free inventory slice for independent
numerical validation. It records an explicit estimand, parameterization,
identification and comparison scope before any external engine is used. An
engine is validation evidence only; it is never a runtime, build, package,
release or sole correctness dependency.

The inventory distinguishes capability coverage from execution status:

- `covered` and `partially_covered` require independent evidence;
- both statuses also require at least one executed evidence row; a planned or
  unavailable run cannot be presented as completed coverage;
- `no_independent_engine`, `not_comparable` and `planned` remain explicit;
- `passed`, `failed`, `indeterminate`, `not_executed` and `not_applicable` are
  preserved rather than collapsed into a green result.

Each evidence row binds the compared engine/version/license, comparison layer,
parameter mapping, fixture, execution environment and result artifact by
immutable identifiers. Full Git SHA-1 and SHA-256 commit identifiers are
accepted because the provenance contract must survive a repository hash-format
migration. The manifest contains no raw responses, restricted datasets,
proprietary binaries or external-engine imports.

This slice intentionally does not execute numerical comparisons or claim
construct validity, fairness, or high-stakes approval. The next customer-useful
step is an isolated harness that populates fixed-parameter evidence for one
protected-main capability and emits the corresponding result artifact.

## Research basis (APA 7)

Chalmers, R. P. (2012). mirt: A multidimensional item response theory package
for the R environment. *Journal of Statistical Software, 48*(6), 1–29.
https://doi.org/10.18637/jss.v048.i06

Morris, T. P., White, I. R., & Crowther, M. J. (2019). Using simulation studies
to evaluate statistical methods. *Statistics in Medicine, 38*(11), 2074–2102.
https://doi.org/10.1002/sim.8086

Mair, P., & Hatzinger, R. (2007). Extended Rasch modeling: The eRm package for
the application of IRT models in R. *Journal of Statistical Software, 20*(9),
1–20. https://doi.org/10.18637/jss.v020.i09

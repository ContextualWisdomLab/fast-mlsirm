# External-validation profile contract

## Decision

`fast_mlsirm.external_validation` owns only a provider-neutral, source-free
manifest for preregistered validation evidence. It records the intended score
interpretation, population, setting, decision use, immutable software/data
identities, evidence class, evidence status, and analysis cutoff.

The contract does not certify validity, fairness, transportability, or a
high-stakes decision. It does not store participant text, direct identifiers,
restricted test content, or hosted lifecycle data. The downstream product that
owns administration and consent remains responsible for lawful data access,
external study design, human review, and decision governance.

## Required invariants

- `preregistered_at` and every evidence `available_time` include a timezone and
  are not later than `analysis_cutoff`.
- Development, internal-validation, and external-validation dataset identity
  cohorts are disjoint. This prevents a declared transport result from silently
  reusing a development identity.
- Evidence classes and non-success states remain explicit; `failed`,
  `indeterminate`, `not_executed`, and `not_applicable` are not converted to
  `passed`.
- Exact package-owned records are admitted before any field access, and hostile
  caller-defined subclasses cannot execute attribute callbacks during
  validation.
- The normalized manifest has a deterministic SHA-256 identity, but that
  identity is provenance evidence rather than a validity certificate.

## Research basis (APA 7th ed.)

American Educational Research Association, American Psychological Association,
& National Council on Measurement in Education. (2014). *Standards for
educational and psychological testing*. American Educational Research
Association.

International Test Commission. (2018). ITC guidelines for translating and
adapting tests (Second edition). *International Journal of Testing, 18*(2),
101–134. https://doi.org/10.1080/15305058.2017.1398166

Morris, T. P., White, I. R., & Crowther, M. J. (2019). Using simulation studies
to evaluate statistical methods. *Statistics in Medicine, 38*(11), 2074–2102.
https://doi.org/10.1002/sim.8086

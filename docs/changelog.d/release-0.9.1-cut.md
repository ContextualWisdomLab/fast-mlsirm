# Release cut 0.9.1

## Changed

- Project version is bumped to 0.9.1 in `pyproject.toml`, `crates/mlsirm-core`,
  and `crates/fast-mlsirm-py`. The accumulated `Unreleased` notes now form the
  `[0.9.1] - 2026-08-25` release section: new governed contracts (a judge
  construct-measurement contract for LLM-as-a-Judge orchestration, a Rust-owned
  independent longitudinal state layer, and a joint MAP hierarchical
  continuous-time AR(1) Rasch estimator), extended-precision identity
  preservation at the Rust boundary (Rasch CML control/group identity,
  G-theory mastery-cut identity, RSM tolerance identity through `f64`), a
  continuation of the hostile-callback/conversion-protocol hardening sweep
  across dozens of public entry points (CAT, ATA, CDM, DIF, equating, facets,
  fitting diagnostics, G-theory, inference, interaction maps, judge panels,
  linking, MHRM, Mokken, Oakes uncertainty, parallel analysis, polytomous
  prediction/recovery, Rasch CML, RSM, subscores, Warm WLE, and
  Benjamini-Hochberg admission, among others), governance/provenance fail-closed
  controls for release, buyer-evidence, PR queue, and procurement source-commit
  provenance, method-literature citation ADRs, and reproducibility work
  binding `uv.lock` resolution to the declared Python floor.
- This cut also removes seven authoritative fragments that no longer carried
  genuinely unreleased content: six whose content was already recorded verbatim
  in the `[0.9.0] - 2026-08-24` section by that release's fold but whose files
  were never deleted (`1028-fitstats-sx2-control-callback-safety.md`,
  `1032-gtheory-control-preflight.md`, `1266-gtheory-resource-admission.md`,
  `1268-gtheory-dstudy-row-bound.md`, `1269-gtheory-numpy-dstudy-controls.md`,
  `1314-linking-evidence-admission.md`), plus the standing predecessor note
  `release-0.9.0-cut.md`, whose substance is permanently recorded in that same
  section and in git history, mirroring the precedent set by the v0.9.0 cut's
  removal of the stale 0.8.0 leftover.
- Released authoritative fragments are removed from `docs/changelog.d`; the
  directory again holds only genuinely unreleased notes.

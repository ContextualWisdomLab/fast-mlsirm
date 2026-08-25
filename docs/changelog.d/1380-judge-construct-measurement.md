# Judge construct measurement contract

## Added

- Add a package-level LLM-as-a-Judge measurement contract that treats each rubric criterion as one dichotomous or polytomous item, enforces zero-based category coding, and separates the three-item identification floor from the package's five-item default and seven-item recommended facet policy.
- Keep the package-wide facet ceiling at 11 items even when callers provide a custom `JudgeConstructPolicy`; custom policies may tighten the quality envelope but cannot silently raise the documented hard maximum.
- Persist the originating validated construct policy on package-created `JudgeConstructSpec` records and replay that policy at projection time, so post-construction criterion mutation cannot silently turn a below-minimum or above-maximum facet into an apparently policy-compliant handoff.
- Project judged responses into deterministic persons × items matrices only when every result carries exactly the non-blank criterion identities declared by the construct specification, and preserve `spec.criterion_ids` as the authoritative output-column order so downstream item parameters cannot be mislabeled by lexical key ordering.
- Pass the authoritative `spec.criterion_ids` order directly to a package-owned projection helper instead of relying on matching lexical sorts or mutating the shared `LLMJudgeResult.to_irt_row()` method at import time.
- Validate `item_type`, category-count semantics, and the `allow_short_form` Boolean control before iterating caller criterion evidence; preserve exact built-in and concrete NumPy Boolean short-form controls while rejecting callback-bearing truth-value providers without executing caller code.
- Revalidate direct and post-construction-mutated `JudgeConstructSpec` records at the projection trust boundary, including non-blank criterion identities, the global 3..11 item envelope, originating policy bounds when present, and dichotomous/polytomous category semantics, before reading item identities or marshalling rows.
- Keep GRM/GPCM fitting, likelihood, scoring, recovery, and all production psychometric arithmetic Rust-owned; the new surface performs validation, marshalling, and recovery evidence only.

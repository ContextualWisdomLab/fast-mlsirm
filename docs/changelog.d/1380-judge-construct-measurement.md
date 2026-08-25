# Judge construct measurement contract

## Added

- Add a package-level LLM-as-a-Judge measurement contract that treats each rubric criterion as one dichotomous or polytomous item, enforces zero-based category coding, and separates the three-item identification floor from the package's five-item default and seven-item recommended facet policy.
- Keep the package-wide facet ceiling at 11 items even when callers provide a custom `JudgeConstructPolicy`; custom policies may tighten the quality envelope but cannot silently raise the documented hard maximum.
- Project judged responses into deterministic persons × items matrices only when every result carries exactly the non-blank criterion identities declared by the construct specification, and preserve `spec.criterion_ids` as the authoritative output-column order so downstream item parameters cannot be mislabeled by lexical key ordering.
- Revalidate direct and post-construction-mutated `JudgeConstructSpec` records at the projection trust boundary, including non-blank criterion identities, the global 3..11 item envelope, and dichotomous/polytomous category semantics, before reading item identities or marshalling rows.
- Keep GRM/GPCM fitting, likelihood, scoring, recovery, and all production psychometric arithmetic Rust-owned; the new surface performs validation, marshalling, and recovery evidence only.

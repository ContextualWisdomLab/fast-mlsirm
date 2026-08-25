# Judge construct measurement contract

## Added

- Add a package-level LLM-as-a-Judge measurement contract that treats each rubric criterion as one dichotomous or polytomous item, enforces zero-based category coding, and separates the three-item identification floor from the package's five-item default and seven-item recommended facet policy.
- Project judged responses into deterministic persons × items matrices only when every result carries exactly the criterion identities declared by the construct specification, and preserve `spec.criterion_ids` as the authoritative output-column order so downstream item parameters cannot be mislabeled by lexical key ordering.
- Keep GRM/GPCM fitting, likelihood, scoring, recovery, and all production psychometric arithmetic Rust-owned; the new surface performs validation, marshalling, and recovery evidence only.

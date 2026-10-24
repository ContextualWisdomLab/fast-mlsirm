# Specific-factor location: incremental Rust integration

## Scope and base

This Rust-only integration extends the existing multigroup bifactor GRM with an explicit `estimate_specific_means` opt-in, default false. The reference group stays at zero means/unit scales. Focal first moments, centered variances, affine fitting and terminal nodes, reflection and parameter count consume the new locations.

The parent is PR #2406, commit `a35b7701db50035a84e917d821dc2f4198647af0`. Its source differs from the older research staging tree. This incremental port retains the parent's analytic item Hessian, slope-prior objective and AC pattern cache. It does not copy whole frozen source files. The GPU moment tuple and included tree shader carry first moments consistently; no hardware execution is credited to this port.

The mathematical basis remains the parent's bifactor response model and marginal EM references: Cai, Yang, and Hansen (2011), pp. 230–232, Equations 15–17, and Bock and Aitkin (1981), pp. 445–448. The implementation applies the existing posterior first/second-moment update on each specific affine coordinate. The source delivery being ported documents its independent full-product-grid oracle and prior CPU/Metal checks; those historical checks are not current-head hardware acceptance.

References: Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information item bifactor analysis. *Psychological Methods, 16*(3), 221–248. https://doi.org/10.1037/a0023350

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika, 46*(4), 443–459. https://doi.org/10.1007/BF02293801

## Current execution evidence

On macOS, bounded CPU execution with two build/test jobs:

- New library unit cohort: 4 passed. Full product-grid likelihood/moments with missing blocks, terminal EAP/likelihood, reflection, and first-step centered variance.
- Selected integration cohort: 11 passed across six executables, including four location tests, multigroup recovery, free anchors, reflection, single-group reduction and the existing no-prior reference.
- `cargo check --locked --offline -p mlsirm-core --tests`: exit 0 with the GPU feature enabled. This compiles Rust; it does not validate a shader through a physical adapter.
- `cargo check --locked --offline --manifest-path crates/fast-mlsirm-py/Cargo.toml`: exit 0. Python retains its default-zero-location contract.
- `git diff --check`: exit 0.

The full CPU library invocation does **not** pass: it reaches the 260-second observation deadline and its owned test child is subsequently terminated and verified absent. A focused replay of `nominal::tests::nominal_reduces_to_fit_nominal_at_d1` fails with loglik trace difference `2.9911461751908064e-8`; its source and test bytes are unchanged from the parent. No baseline runtime PASS or complete-suite PASS is inferred. Existing compiler warnings remain.

The first integration attempt exposes a source-contract difference: staging accepts zero specific dimensions, but this parent rejects them. The adapted regression asserts the parent's existing rejection with and without the new opt-in; this port does not add general-only fitting support. Staging-only per-start/dispatch receipt assertions are outside this narrower API and are not represented as tested here.

## Unmet gates and follow-ups

- Independent exact-diff approval is pending; a bounded read-only reviewer invocation returns no verdict before timeout.
- Physical GPU execution count for this integration is zero. Hardware acceptance requires the existing shared host lease and unchanged exact-source tests.
- Python opt-in/result transport, nonzero-location downstream scoring and location-aware observed information require separate consumer work. Existing zero-location Oakes APIs are not extended here; do not use them for nonzero-specific-mean fits.
- The mask/CPU-information optimization depends on a mask API absent from main and PR #2406. It is not part of this patch.
- Strict GPU and q22 B1 successor patches remain separate. A B1 review packet exists, but independent re-review approval and actual q22 GPU acceptance are pending.
- Parent PR #2406, exact-head CI/security and protected-main integration remain open. Publication of this stacked draft is not a release or scientific adoption.

Local execution logs and receipts are retained under the integration owner's `upstream_integration_20261024/specific_mean/` report directory. No study response rows or study estimates belong to this patch.

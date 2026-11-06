# Specific-factor location: incremental Rust integration

## Scope and base

This Rust-only integration extends the existing multigroup bifactor GRM with an explicit `estimate_specific_means` opt-in, default false. The reference group stays at zero means/unit scales. Focal first moments, centered variances, affine fitting and terminal nodes, reflection and parameter count consume the new locations.

The parent is PR #2406, commit `a35b7701db50035a84e917d821dc2f4198647af0`. The specific-mean predecessor is `ca30a1fd985470adda84d744ebe21987f539c0b7`. This incremental port retains the parent's analytic item Hessian, slope-prior objective and AC pattern cache. The GPU moment tuple and included tree shader carry first moments consistently; no hardware execution is credited to this port.

The mathematical basis remains the parent's bifactor response model and marginal EM references: Cai, Yang, and Hansen (2011), pp. 230–232, Equations 15–17, and Bock and Aitkin (1981), pp. 445–448. The implementation applies the existing posterior first/second-moment update on each specific affine coordinate. Historical source-delivery CPU/Metal checks are not current-head hardware acceptance.

## F1: location identification admission

Independent source review of the predecessor requests changes: an entirely free specific block admits `mu_S += c` together with `d_ik -= a_iS*c` in every focal free item. The predictor, threshold ordering, slopes and likelihood remain unchanged. The regression executes this invariance through the CPU E-step and reproduces the predecessor's erroneous acceptance of `specific_map=[0,0,1,1,-1]`, `anchor=[true,true,false,false,true]`.

With `estimate_specific_means=true` and two or more groups:

- Every specific block requires at least one common item; an entirely free block fails input validation.
- Before a start can compete for the winning objective, its fitted common-item loading rows must have full column rank for `[general, specific_0, ..., specific_S-1]`. A row contains `a_general` and that item's one specific loading, with zeros in other specific columns.
- For each focal group, only common items observed in **both** that group and reference group 0 enter the rank screen. Separate full-rank but disjoint observed anchor banks do not link their locations. Indirect-only linking through other focal groups is outside this conservative admission contract.
- Rank is checked after fitting, since the deterministic initialization deliberately uses equal loading ratios. Row rescaling, column normalization and twice-reorthogonalized modified Gram–Schmidt use a numerical residual cutoff `64 * f64::EPSILON * max(anchor_rows, dimensions)`. A deficient/nonfinite start is rejected; if every start fails, fitting returns an error with its first failure.

Per-block anchor presence alone does **not** establish joint location rank. Tests reject proportional common loading rows and a zero specific loading, and admit nonproportional full-rank rows even without a general-only anchor. The rank screen is a local necessary condition for pure location shifts with fixed loadings. It does not establish full-model, variance, rotation, practical or Fisher-information identification. Cai et al. (2011, p. 230) require sufficient linking restrictions and recommend inspecting Fisher-information definiteness. The location-only screen is derived here from their graded slope-intercept predictor (p. 225, Equation 6), not presented as a theorem from that article. Those full-text sections are read from the retained article. The false/zero-mean and single-group contracts remain separately tested.

## F2: physical-GPU test admission

The predecessor GPU fit test accepts equal CPU fallback output because it checks only estimates, likelihood and `max_iter_reached`. A deterministic negative control executes that old predicate with a real CPU fit and reproduces false admission before the repair.

The ignored `specific_mean_gpu_fit_matches_cpu_and_terminal_dispatch` test now lives in `tests/unit/specific_mean_tests.rs`, beside a private `cfg(test)` observer. It retains the predecessor's response fixture, 7/7-node grid, two-iteration budget and parity tolerances, and uses two starts. The observer resets the existing thread-local GPU receipt before each E-step, then captures its actual return receipt and the real stopping branch. The shared test oracle requires:

- A nonempty receipt sequence for every start, including nonwinning starts, with contiguous iteration numbers and exactly the final sweep marked terminal.
- `used=true`, nonempty backend/device identity and no fallback reason on **every** sweep, including the terminal likelihood E-step.
- The winning terminal iteration and receipt count to match the returned `n_iter` and likelihood trace.
- The existing CPU/GPU estimate, scale, likelihood and budget-termination comparisons.

The terminal likelihood E-step is distinct from the existing CPU final EAP pass. This is not an all-arithmetic-on-GPU claim. Production fallback semantics and public result/config fields gain no dispatch policy or new receipt API in this repair.

CPU-only controls reject an actual CPU substitute, empty/missing-start receipts, early or terminal CPU fallback, retained fallback reasons, missing backend/device identity, missing terminal markers and reordered iteration numbers. A synthetic receipt-shaped positive exercises only the test oracle; it is not hardware evidence. GPU execution and adapter initialization are not performed in this repair. Both physical-GPU tests remain ignored. GPU-feature compilation does not validate a shader on a physical adapter.

## Bounded successor execution evidence

On macOS, with two build/test jobs, explicit scratch temporary storage, finite command deadlines and source-hash-bound logs:

- Specific-mean CPU library cohort: **8 passed**, including the CPU likelihood-invariance and receipt controls.
- Selected CPU integration cohort: **15 passed** across six executables; the specific-mean executable contributes eight cases, including unanchored, proportional-rank, unobserved focal anchor and disjoint reference/focal anchor rejections.
- Existing analytic item Hessian cases: **2 passed**; original-pair Newton Hessian case: **1 passed**; common/free slope-prior cases: **2 passed**; existing AC cache/storage cases at 121/241 nodes: **8 passed**.
- `cargo check --locked --offline -p mlsirm-core --tests`: exit 0 with the default GPU feature; no GPU test execution.
- `cargo check --locked --offline --manifest-path crates/fast-mlsirm-py/Cargo.toml`: exit 0. Python retains its default-zero-specific-mean contract.
- `git diff --check`: exit 0.

These are **36 unique selected CPU tests**, not a full-suite PASS. The first AC cache selector matches zero tests and is excluded; the corrected actual module selector executes eight. Existing compiler warnings remain. RED logs retain the predecessor's unanchored-block acceptance, proportional-loading acceptance, CPU-only GPU-oracle acceptance, and an intermediate screen's disjoint-observed-bank acceptance. A first formatting check reports an added trailing blank line before its correction; that receipt is not a passing check.

The predecessor's full CPU library invocation reaches its 260-second deadline and its owned test child is subsequently terminated and verified absent. The focused `nominal::tests::nominal_reduces_to_fit_nominal_at_d1` replay fails with likelihood-trace difference `2.9911461751908064e-8`; its source/test bytes are unchanged from the parent. These historical NONPASS results remain, and this repair does not rerun or relabel that full suite.

## Unmet gates and follow-ups

- Predecessor exact16 independent review is **CHANGES_REQUESTED**, source-only; the excluded documentation is not independently reviewed. This F1/F2 successor requires fresh independent exact-source review. The earlier bounded reviewer timeout supplies no verdict. No formal App approval is claimed.
- Physical GPU execution count for this integration is zero. Actual GPU acceptance remains pending under the existing shared host lease and the strengthened exact-source tests.
- Python opt-in/result transport, location-aware downstream scoring and observed information require separate consumer work. Existing Oakes APIs describe **zero-specific-mean constrained models**; they are unsupported for any free-specific-mean fit, even if fitted means happen to be zero. Rust public struct additions in the port still affect external exhaustive literals.
- Study-scale recovery, mean-on MAP convergence/sensitivity, three or more groups, large mean shifts and full Fisher-information identification are unverified. Small regression fixtures do not establish study accuracy or scientific adoption.
- Mask/CPU-information work remains separate. **B1 source review passed, original-main port and actual GPU execution pending.** No B1 or mask patch is included here.
- Parent PR #2406, current-head hosted CI/security and protected-main integration remain separate gates. The predecessor CodeQL Actions job fails with `No space left on device`, other CI is skipped and formal reviews are zero in its retained receipt. Successor publication is a stacked draft update, not a release, merge or scientific adoption.

Predecessor logs remain under `upstream_integration_20261024/specific_mean/`; successor raw logs, bounded command receipts, RED/GREEN evidence and exact-source manifests are under `upstream_integration_20261024/specific_mean_f1_f2/`. No study response rows or study estimates belong to this patch.

## References

Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information item bifactor analysis. *Psychological Methods, 16*(3), 221–248. https://doi.org/10.1037/a0023350

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika, 46*(4), 443–459. https://doi.org/10.1007/BF02293801

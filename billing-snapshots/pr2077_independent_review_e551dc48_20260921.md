# Independent model-contract review — PR #2077

| Field | Value |
|-------|-------|
| **Target** | ContextualWisdomLab/fast-mlsirm PR [#2077](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/2077) |
| **Head** | `e551dc484c2806b6030836edad0953a21669b8ff` (`seonghobae/two-tier-expected-raw-etg`) |
| **Reviewer role** | Independent model-contract review only (read-only; no PERF/FIPC ownership) |
| **Date (UTC)** | 2026-09-20 |
| **Verdict** | **ACCEPT-WITH-NITS** |

**Scope.** Five contract questions only. No cargo/maturin/A4. #2079 FIPC out of scope. No auto-merge / admin-merge.

**Changed surface.** `python/fast_mlsirm/two_tier_grm.py` (+307), `tests/test_two_tier_expected_raw.py` (+353), `_legacy_init.py` exports.

---

## Paper basis (what does / does not ground this API)

This path is **two-tier / bifactor graded expected raw score**, not MLS2PLM latent-space η.

| Source | Role for this PR |
|--------|------------------|
| **Cai (2010)** *Psychometrika* 75(4), 581–612 | Two-tier structure: primaries may correlate (Φ free); specifics mutually orthogonal and orthogonal to primaries; item loads ≥1 specific at most (pp. **582**, **586**). Independent Gaussian product over specifics follows from normality + orthogonality (p. **587**). |
| **Gibbons et al. (2007)** *APM* 31(1), 4–19 | Bifactor GRM; Gauss–Hermite form (eq. 8, p. 7); linear predictor with G + ≤1 specific (eq. 9); dimension reduction under orthogonal N(0,1) (eqs. 11–14, p. 8). Linearity ⇒ `E[T\|θ_G] = Σ_i E[Y_i\|θ_G]` with per-item 1-D GH over S (already implemented as `check_bifactor_expected_total_score_monotonicity`). |
| **Chalmers (2012)** *JSS* 48(6); mirt `bfactor` / two-tier docs | Orthogonal secondary traits fixed variance 1; primaries may covary — matches repo `two_tier_grm.rs` module contract and the PR’s “orthogonal mirt ID ⇒ independent nuisances” *example*. |
| **Kang & Jeon (2025) MLS2PLM; Jeon et al. (2021) LSIRM** | **Not** the formula contract for this API. Do not reinterpret E[T\|G] through MLS2PLM η / distance. Cited only to mark non-applicability under AGENTS.md paper-first rule. |

**Manuscript special case (G+4+W).** Treating G and W as *orthogonal primaries* (Φ = I) plus orthogonal specifics is a **restriction** of Cai’s general two-tier (where Φ may be free). Under that restriction, integrating W and S as independent reference Gaussians for `E[T|G]` is coherent. If Φ_(G,W) ≠ 0, correct nuisance integration is **Φ-conditional** (not implemented; correctly fail-closed on `from_fit`).

---

## (1) Per-dimension `primary_ref_*` / `specific_ref_*` vs W / S

**Finding: correct and explicit.**

Mapping in `expected_total_score_two_tier_given_primary` (`two_tier_grm.py` ~657–681):

| Vector | Index | Consumed when |
|--------|-------|----------------|
| `primary_ref_mean` / `primary_ref_sd` | primary column `p` | `p ≠ focal` and `a_primary[i,p] ≠ 0` (W lives here when it is a non-focal primary) |
| `specific_ref_mean` / `specific_ref_sd` | `specific_map[i]` | `sid ≥ 0` and `a_specific[i] ≠ 0` |
| Focal primary slot | length `n_primary` | Present for shape consistency; **unused** in the integral (documented) |

Scalar mean/sd broadcast is documented as the producer case where every nuisance is N(0,1). Unequal W vs S must pass arrays — covered by `test_per_dimension_ref_sd_changes_integral_vs_scalar_bundle` and `test_g4w_distinct_w_vs_s_reference_variances`.

**Nit.** Prior token remains `"independent_standardized"` while refs may be non-unit SD. Docs mitigate; rename later would reduce misread risk.

---

## (2) `orthogonal_primary_identification` + Φ == I — theater or real?

**Split verdict: real numeric gate + honest attestation; not estimator preservation.**

| Mechanism | Behavior | Theater? |
|-----------|----------|----------|
| `_require_identity_phi(..., atol=0.0)` | Rejects any non-exact I (including Φ≈I at 1e-8) | **No** — hard fail-closed for `from_fit` |
| `orthogonal_primary_identification is True` | Required kwarg; False/omitted raises | **Not fake-pass theater** — cannot omit |
| Library `fit_two_tier_grm` | Free-estimates primary correlations (Fisher-z M-step); module docs: Φ unit diagonal, **free off-diagonals** (`crates/mlsirm-core/src/two_tier_grm.rs` ~31–38, 583+) | Flag **does not** verify this estimator preserved ortho constraints — **the estimator does not offer fix-Φ=I** |

Docs at `from_fit` (~751–759) correctly state: Φ==I is necessary but not sufficient; `TwoTierGrmFit` has no identification metadata; flag is consumer confirmation until metadata exists.

**Implication for consumers.** `from_fit` is appropriate for **externally** orthogonal fits (e.g. mirt G+4+W with Φ fixed at I) or hand-built `TwoTierGrmFit` stubs with Φ≡I. A normal free-Φ `fit_two_tier_grm` result with P>1 will almost never pass exact I — good. Attesting `True` after overwriting `fit.phi = I` on correlated slopes remains a **provenance** silent-wrong path (see §5).

---

## (3) Asymmetric fixture adequacy

| Fixture | Adequacy |
|---------|----------|
| Bifactor parity vs `check_bifactor_expected_total_score_monotonicity` (atol 1e-10) | **Strong** for 1-primary reduction |
| W-cross vs specific-only curve change | **Strong** qualitative |
| 1-item asymmetric thresholds + distinct W/S SD | **Strong** for ref-scale sensitivity |
| MC vs GH (unit N(0,1), 2 nuisances, abs 0.02) | **Adequate** for unit refs |
| Dual gates on stub `from_fit` | **Adequate** for gate logic |
| G+4+W 16-item | **Partial** — wording/reverse/bounds; **symmetric** thresholds `[1.2,0,-1.2]`; mid-grid `E[T]=24` is a symmetry artifact |
| Distinct W/S on G4W grid | Shows vectors differ; weak at G=0 by symmetry |

**Nits (not blockers):** no MC/closed-form check for **non-unit** refs; no explicit test that focal slot of `primary_ref_*` is ignored; `from_fit` never driven by a real `fit_two_tier_grm` / mirt artifact; no sparse `specific_map` hole coverage.

---

## (4) Real consumer contract vs late-life G+4+W

PR body + docstring treat G+4+W as an **example**, not a universal library contract (`two_tier_grm.py` ~502–504, 593–596). That is the right reusable-API stance.

Manuscript alignment under orthogonal ID:

- Focal G = column 0; W = non-focal primary; S via `specific_map` — implemented.
- Independent reference nuisances — default refs + `nuisance_prior` gate.
- `E[T|G]=Σ E[Y_i|G]` — correct; bifactor helper parity supports the reduction.
- Public exports in `_legacy_init.py` — present.

**Consumer seam (nit / operational).** Late-life wiring should call `given_primary` with manuscript slopes/refs, or `from_fit` only on **Φ≡I orthogonal-ID** parameter objects — **not** free-Φ internal fits with φ overwritten to I. PR #243 EAP aggregates correctly remain auxiliary (out of this review’s numeric path).

---

## (5) Remaining silent-wrong-value risks

| Risk | Severity | Notes |
|------|----------|-------|
| `given_primary` has **no** Φ / ID gate | **High** (by design) | Correlated-Φ slopes + independent GH ⇒ wrong E[T] with no error. Callers must not treat this as a safe `from_fit` substitute. |
| Attest `True` + force `phi = I` on correlated-fit slopes | **High** | Dual gate cannot detect slope provenance. |
| Scalar default `ref_sd=1.0` when W/S SDs differ | **Medium** | Mitigated by per-dim API + tests; defaults still invite N(0,1). |
| `specific_map.astype(np.int64)` without finite-integer precheck | **Medium** | Truncation / non-finite → wrong `sid` → wrong `specific_ref_*` (fit path validates more strictly). |
| Prior name `"independent_standardized"` with non-unit SD | **Low** | Label vs capability. |
| Exact `coef != 0.0` skip | **Low** | Pattern zeros OK; tiny floats still integrated. |
| Product meshgrid for >2 nuisances | Perf only for G+4+W (≤2) | Not silent-wrong at manuscript width. |

No failing-test snippet was required for a BLOCK; none of the above rise to merge-block for the **stated scoring API** under external orthogonal ID.

---

## Verdict and required actions

### **ACCEPT-WITH-NITS**

The E[T|θ_focal] scoring contract is paper-coherent under Cai/Gibbons orthogonality restrictions, fails closed on Φ-conditional demand for `from_fit`, maps W/S refs correctly, and documents that consumer confirmation is attestation — not library estimator proof.

### Blocker list

**None** for merge of this scoring API under the manuscript contract (external orthogonal ID + explicit refs).

Would become **BLOCK** if the PR claimed `from_fit` + `orthogonal_primary_identification=True` means `fit_two_tier_grm` estimated under orthogonal primaries — it does not, and current docs correctly avoid that claim.

### Nits for PERF (non-blocking)

1. Validate `specific_map` as finite integers before `astype(np.int64)` (align with fit-path checks).
2. Optional: MC or analytic check under non-unit `primary_ref_sd` / `specific_ref_sd`.
3. Optional: assert focal `primary_ref_*` slot is unused (mutate focal sd; E[T] unchanged).
4. Doc/API: note `from_fit` targets externally orthogonal / Φ≡I fits; consider identification metadata on `TwoTierGrmFit` in a later PR.
5. Consider renaming prior token when non-unit refs are first-class (or document “standardized” = independent Gaussian product, not unit variance).

### Out of scope (explicit)

- #2079 FIPC / A4 remeasure wiring.
- Implementing Φ-conditional nuisance integration.
- Adding fix-Φ=I to `fit_two_tier_grm`.

---

## References

Cai, L. (2010). A two-tier full-information item factor analysis model with applications. *Psychometrika, 75*(4), 581–612. https://doi.org/10.1007/s11336-010-9178-0  
(pp. 582, 586–587 for tier covariance / orthogonality.)

Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. *Journal of Statistical Software, 48*(6), 1–29. https://doi.org/10.18637/jss.v048.i06

Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E., Bhaumik, D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover, A. (2007). Full-information item bifactor analysis of graded response data. *Applied Psychological Measurement, 31*(1), 4–19. https://doi.org/10.1177/0146621606289485  
(eqs. 8–14, pp. 7–8.)

Kang, I., & Jeon, M. (2025). Multidimensional latent space item response models: A note on the relativity of conditional dependence. *Psychometrika, 90*(2), 799–826. https://doi.org/10.1017/psy.2025.5  
(**Non-basis** for this two-tier expected-raw API.)

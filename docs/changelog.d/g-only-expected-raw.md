# G-only expected raw scoring

## Added

`fast_mlsirm.two_tier_g_only.expected_raw_g_only` calls the strict native
GPU mean scorer. For a two-primary bank in `(G, W)` order with `Phi == I`,
it integrates `W ~ N(mean_w, sd_w²)` and independent, zero-mean specifics
with explicit SDs. It returns zero-based category-total means and the native
backend/device, dispatch, fallback and two nuisance-node counts. Both node
counts and all nuisance-prior parameters are required. `cpu` and `auto` are
refused; there is no numerical CPU fallback.

The estimand is `F(g) = E[T | G=g]`. Passing `fit.theta_p_eap[:, 0]` gives
`F(E[G|responses])`, not `E[F(G)|responses]`. W EAP is unused. Nuisance
priors belong to the caller and are not inferred from fit quadrature or
silently selected from group/reference fits. The existing two-primary
plug-in API keeps its current meaning.

The mean is the sum of item marginal means, using the cumulative GRM
identity `E[Y] = sum P(Y >= k)`. It does not return a score distribution,
variance, posterior uncertainty or a converged/adopted fit. Shared nuisance
factors matter for distributions and covariance; no independence of item
responses after marginalization is claimed.

**Precision and range**

The native shader uses pairwise binary-tree sums and normalizes the
uploaded f32 quadrature masses on device. Host admission bounds uploaded
transforms, absolute affine predictors and reassociated two-factor
products by `f32::MAX / 4` before numerical dispatch. A finite f64/f32
input alone does not establish admissibility. This conservative rule can
refuse representable end results that rely on overflow cancellation.

The retained exact-final native cohort has 29 passing cases on Metal / Apple
M1, including a 24-item binary constant total of exactly 12 at 15, 121 and
241 nuisance nodes, a 24-item nonconstant native Lord-Wingersky comparison
(maximum absolute discrepancy 6.06996676922e-6, unchanged 2e-5 fixture
criterion), and finite extreme/reassociated overflow refusals. These are
synthetic software checks, not study quadrature or scientific acceptance.

**Verification**

- CPU-only boundary controls: `pytest tests/test_two_tier_g_only_controls.py`.
- Explicit physical-GPU acceptance:
  `pytest tests/gpu_two_tier_g_only_acceptance.py`; an external supervisor
  must first hold the host's real shared GPU lease and set
  `FMLS_E_GPU_LEASED=1`. The environment flag is an assertion, not a lock.
- Rust fixture targets: `e_scoring_review`, `e_scoring_review_findings`,
  `e_scoring_precision`; run their ignored hardware cases only under that
  same lease, with one test thread. No GPU absence skips are used.

The scoped change imports the exact repaired scorer and its tests, not the
separate GPU item optimizer, two-word Newton shader, or GRM fitting changes.
It depends only on the already-existing GPU buffer/dispatch utility.

**Primary sources**

Cai (2010, p. 589, Eqs. 11-12) defines the cumulative GRM/category
probabilities. Cai (2015, pp. 542-543, Eqs. 14-17) defines the two-tier
specific-factor integration. Linearity of expectation supplies the
mean-only reduction. The caller's independent-normal W marginal and the
plug-in EAP choice are explicit API restrictions, not a claim that those
papers equate plug-in and posterior expected totals.

Cai, L. (2010). A two-tier full-information item factor analysis model with
applications. *Psychometrika, 75*(4), 581-612.
https://doi.org/10.1007/s11336-010-9178-0

Cai, L. (2015). Lord-Wingersky algorithm version 2.0 for hierarchical item
factor models with applications in test scoring, scale alignment, and model
fit testing. *Psychometrika, 80*(2), 535-559.
https://doi.org/10.1007/s11336-014-9411-3

Full-text locators are verified from Zotero attachments HB7X6FLI (PDF p. 9)
and DZ3IMIV9 (PDF pp. 8-9). Publisher PDFs are not redistributed here.

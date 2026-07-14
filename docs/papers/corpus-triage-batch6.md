# Corpus triage — batch 6

The sixth reading set is, by the sender's own note, largely a re-presentation
of batches 3–5 (the same ~190 PDFs) plus a short new head list. Rather than
re-litigate every duplicate, this note records the one **new** implementation
this round and points at the earlier triage docs for everything already
dispositioned.

## Implemented in this batch

| Paper(s) | Feature |
|---|---|
| Maydeu-Olivares & Joe (2005, 2006); Cai & Hansen (2013); Maydeu-Olivares (2013); Maydeu-Olivares & Joe (2014) | **M2 / RMSEA2 limited-information goodness-of-fit** — `mlsirm_core::fitstats::m2_rmsea2` (+ `fast_mlsirm.fitstats.m2`). Univariate + bivariate residual margins, df, χ² p-value, RMSEA2 with a 90% noncentral-χ² CI, and the bivariate SRMSR. Model-implied margins and the up-to-4th-order `Xi_2` covariance entries are exact via the local-independence factorization over the `(theta, xi)` node set (Cai-Hansen dimension reduction); `Delta_2` central-differenced; the quadratic form solved through one Cholesky of `Xi_2`. Rust compute path, NumPy parity reference (1e-6), calibration + local-dependence tests in both suites. |

`M2/RMSEA2` moves from the roadmap to done; it was the top consolidated
roadmap item after `BIFAC2PLM` (batch 5).

## Newly-listed head papers (first 10) — disposition

- **Brossman & Lee (2013)** MIRT observed/true-score equating; **Yao &
  Boughton (2009)** mixed-type MIRT linking; **Kim & Lee (2006)** linking
  methods — the linking/equating roadmap item (below). The EAPsum + TCC
  machinery already produces the score tables these procedures operate on.
- **Wang (2015)** latent-trait estimation in compensatory MIRT — covered by
  the marginal engine's EAP/MAP per-dimension scoring.
- **van den Berg, Glas & Boomsma (2007)** variance decomposition via an IRT
  measurement model — the multilevel random-intercept structure (σ_u²/ICC)
  already reports the between/within variance split.
- **Woehr & Meriac (2010)** polytomous DIF; **Carter et al. (2011)** mixed-model
  / ideal-point survey IRT; **Böckenholt et al. (2017)** response-style
  multi-process IRT — the polytomous-kernel and C-class-mixture roadmap items.
- **Kahraman (2013)** unidimensional interpretation of multidimensional items —
  covered (batch 5): per-dimension EAP + the EAPsum "projected" serving scale.
- **Pastor (2003)** applied multilevel IRT — implemented multilevel structure.
- **Khodadady & Ghanizadeh (2011)** EFL concept-mapping application; **Milanzi
  et al. / "Reliability measures" (2015)** manifest-vs-latent correlation
  functions — the latter is the `empirical_reliability` basis (batch 5); the
  former is an applied study (context only).

## Everything else

Already dispositioned in `corpus-triage-batch3.md`, `-batch4.md`, and
`-batch5.md` (implemented, already-covered, or foundational/context). No new
dispositions are warranted for the duplicated tail.

## Roadmap (consolidated; explicitly requested across batches)

1. **General C-class mixture IRT** — Sawatzky et al. (2016); Carter et al.
   (2011); Zickar et al. (2004 faking classes); Finch & Pierson (2011): the ZI
   mixture generalizes (class-weighted E-step already exists); class-specific
   item parameters are the added state.
2. **3PL/4PL estimation** (Barton-Lord 1981; Falk & Cai 2016 semiparametric-
   with-guessing): response-kernel change; the table architecture keeps the GPU
   path untouched (as for BIFAC2PLM).
3. **Polytomous kernels** — Muraki (1990 GPCM); Thissen, Cai & Bock (2010
   nominal model); Böckenholt et al. (2017); De Jong et al. (2008 ERS);
   Weijters et al. (2013 reversed items); Woehr & Meriac (2010 polytomous DIF).
4. **Linking/equating + projective transforms** — Brossman & Lee (2013); Yao &
   Boughton (2009); Kim & Lee (2006); Ip & Chen (2012 projective IRT);
   Stocking-Lord.
5. **Response-time integration** — van der Linden et al. (2010); Wise (2017
   rapid-guessing flags); Kyllonen & Zu (2016).

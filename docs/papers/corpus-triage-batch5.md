# Corpus triage — batch 5 (~44 papers)

Disposition of the fifth supplied reading set (same legend as batches 3-4).

## Implemented in this batch

| Paper | Feature |
|---|---|
| Stanley & Edwards (2016), reliability and model fit; Milanzi et al. (2015), manifest vs latent correlation functions | `empirical_reliability()` — marginal EAP reliability per trait dimension, documented with the papers' caveat that the coefficient presumes a well-fitting model |

## Already covered (earlier basis)

- CAT applications — Makransky & Glas (2013 organizational MCAT); Haley et
  al. (2009 item-bank replenishment = FIPC use case); the MCAT fatigue and
  heterogeneous-population CAT applications: `cat_next_item` +
  FIPC/anchoring implement the operational loop; Sawatzky et al. (2016)
  motivates the mixture roadmap item below.
- AES — Loukina & Buzick (2017 spoken-language auto-scoring use):
  `validate_judge` gates.
- Person misfit / aberrance — Tendeiro (2016 l_z(p) in unfolding contexts),
  Wise (2017 rapid guessing — the RT-based flag is roadmap; the response-
  pattern side is covered by l_z*/resampling person fit and the ZI class).
- Multilevel IRT applications — Pastor (2003); Frazier et al. (2015):
  implemented multilevel structure.
- Unidimensional interpretations of multidimensional items — Kahraman
  (2013); Ip & Chen (2012 projective IRT); Ip (2010 functionally
  unidimensional): the marginal engine reports per-dimension EAPs and the
  EAPsum tables give the "projected" unidimensional serving scale; the
  formal projective-model transformation is noted under the linking roadmap.
- Reise bifactor cluster (2007, 2012), Thomas (2012), Zhang et al. (2014),
  Reise, Moore & Maydeu-Olivares (2011 target rotations), Toland et al.
  (2017): all reinforce the top roadmap item below.

## Implemented after triage (same change set)

- **`BIFAC2PLM` bifactor / inner-product interaction kind** — Gibbons &
  Hedeker (1992); Cai, Yang & Hansen (2011); the Reise cluster.
  `InteractionKind::{None, Distance, Inner}` now parameterizes every eta
  site: `eta += dot(zeta_i, x)` (bilinear/Hoff form; the dichotomous
  bifactor at `latent_dim = 1` with `lambda_i = zeta_i`), reusing the
  conditional-factorization E-step (the Gibbons-Hedeker dimension
  reduction), the tables, and the GPU kernels unchanged. Positive-manifold
  loading init; tau inert; marginal estimator only (JML guards). Rust/NumPy
  parity at 1e-9; loading-recovery tests in both suites.

## Roadmap (consolidated; explicitly requested across batches)

> Superseded by `corpus-triage-batch6.md`. **M2/RMSEA2 is now implemented**
> (`fitstats::m2_rmsea2`); the remaining items below carry forward.

3. **General C-class mixture IRT** — Sawatzky et al. (2016); Carter et al.
   (2011); Zickar et al. (2004 faking classes); Finch & Pierson (2011): the
   ZI mixture generalizes (class-weighted E-step already exists); class-
   specific item parameters are the added state.
4. **3PL/4PL estimation** (Barton-Lord; Falk & Cai 2016 semiparametric-
   with-guessing strengthens the case).
5. **Polytomous kernels** — Thissen, Cai & Bock (2010 nominal model);
   Böckenholt et al. (2017 response styles); De Jong et al. (2008 ERS);
   Weijters et al. (2013 reversed items); Vispoel & Kim (2014); Wakita et
   al. (2012); Woehr & Meriac (2010 polytomous DIF).
6. **Response-time integration** — Wise (2017 rapid-guessing flags);
   Kyllonen & Zu (2016).
7. **Linking/equating + projective transformations** — Ip & Chen (2012).

## Foundational / context (documentation only)

Reise & Revicki / van der Linden handbooks (batch 4); Christensen, Kreiner &
Mesbah (2012 Rasch in health); Van der Ark et al. (2015 proceedings); Reeve
et al. (2007 PROMIS calibration practice — the operational template our
screening/serving pipeline mirrors); Brown, Inceoglu & Lin (2017 forced
choice); Luo et al. (2013 robust Bayesian longitudinal); Hamano & Sato
(2005 association rules via IRT); Blom et al. (2010 unit nonresponse);
remaining applied/clinical PDFs.

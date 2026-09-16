# Fixed-item parameter calibration (FIPC) for polytomous GRM

## Added

- Add fixed-item parameter calibration for the unidimensional GRM
  (`mlsirm_core::poly::fit_poly_fipc`, Python `fast_mlsirm.polytomous.fit_poly_fipc`)
  and the polytomous bifactor GRM
  (`mlsirm_core::bifactor_grm::fit_bifactor_grm_fipc`, Python
  `fast_mlsirm.bifactor_grm.fit_bifactor_grm_fipc`): caller-fixed anchor
  items from a reference calibration plus a flag vector; non-anchor item
  parameters and the focal population's latent mean/variance (general
  factor; specific variances where identified) estimated by MML-EM with the
  prior distribution updated after every M-step — the MWU-MEM method (Kim,
  2006, eqs. 14-15, pp. 361-362), the only compared variant that recovered
  shifted focal distributions without under-estimation (pp. 377-378; Paek &
  Young, 2005). No rescaling of the latent points after an EM cycle, no
  reflection canonicalization: fixed anchors pin the orientation, including
  reverse-keyed anchors. Quadrature node counts stay caller arguments; the
  unidimensional rule set gains the 121-node Gauss-Hermite rule
  (`numpy.polynomial.hermite_e.hermegauss(121)`, weights normalized) as the
  study floor.
- Validate against `mirt::fixedCalib` (MWU-MEM default) on a committed
  unidimensional GRM fixture (`tests/fixtures/poly_fipc_grm/`, R 4.x with
  mirt 1.46.1): free-item agreement plus a same-objective profile-likelihood
  corroboration (mirt's stacked-data empirical-histogram loglik is recorded
  but not directly comparable). Recovery under a known mean/variance shift,
  equivalence to concurrent calibration with anchors fixed at truth, and
  reverse-keyed anchors are covered for both models; study-scale runs
  (`N = 1,020`, `q_theta = 121` unidimensional) run as ignored
  statistical-studies tests with bands from measured multi-seed spreads.
  Paper basis: Kim (2006, JEM 43(4), 355-381,
  https://doi.org/10.1111/j.1745-3984.2006.00021.x) and Paek & Young (2005,
  AME 18(2), 199-215, https://doi.org/10.1207/s15324818ame1802_4).

# Polytomous latent-space final-bank likelihood

## Fixed

- Return the observed-data marginal log likelihood evaluated at the returned GRM/GPCM item parameters, using the existing final posterior pass. Previously the receipt described the parameters before the last M-step, while the returned item bank and person scores used the updated parameters. Estimation, convergence, priors, and posterior scores are unchanged. Regression coverage recomputes the returned-bank quadrature normalizer for both models, one/two latent dimensions, and one/three iterations, including missing observations.

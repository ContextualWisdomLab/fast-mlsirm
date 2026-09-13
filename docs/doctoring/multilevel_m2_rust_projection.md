# Multilevel M2 Rust projection ownership

This slice closes the numerical ownership gap for the two dense projected-M2
quadratic forms in `m2_multilevel`: the fitted-model residual and the
cluster-robust independence baseline now cross the PyO3 boundary through
`projected_m2`. Python continues to assemble the multilevel moment layout and
cluster-total covariance in this bounded change; the full moment-construction
migration remains tracked by issue #627.

The shared random-intercept construction is retained. It is a repository
implementation choice grounded in the multilevel IRT principle that clustered
responses require a higher-level latent component rather than iid person-level
independence (Fox & Glas, 2001). The cluster-robust limited-information
covariance context follows Jamil et al. (2025), whose setting is not identical;
the implementation therefore does not claim a paper reproduction.

## References (APA 7th)

Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel IRT
model. *Psychometrika, 66*, 271–288. https://doi.org/10.1007/BF02294839

Jamil, H., Moustaki, I., & Skinner, C. (2025). Pairwise likelihood estimation
and limited-information goodness-of-fit test statistics for binary factor
analysis models under complex survey sampling. *British Journal of
Mathematical and Statistical Psychology, 78*(1), 258–285.
https://doi.org/10.1111/bmsp.12358

Maydeu-Olivares, A., & Joe, H. (2006). Limited information goodness-of-fit
testing in multidimensional contingency tables. *Psychometrika, 71*(4),
713–732. https://doi.org/10.1007/s11336-005-1295-9

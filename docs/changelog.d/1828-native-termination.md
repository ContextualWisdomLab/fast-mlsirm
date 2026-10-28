# Polytomous LSIRM termination evidence

## Fixed

The native polytomous LSIRM fit now reports convergence evidence at the
returned parameter state: the observed-data log-likelihood trace, signed final
change, applied tolerance, criterion identity, iteration count, and termination
reason. Non-finite observed likelihoods return an error before posterior scores
are returned.

The stopping criterion is the observed-data likelihood change. It is distinct
from the MAP/penalized M-step objective used for item updates and does not imply
a global maximum. This follows the EM convergence scope in Wu, C. F. J. (1983),
“On the convergence properties of the EM algorithm,” *The Annals of
Statistics*, 11(1), 95–103. https://doi.org/10.1214/aos/1176346060

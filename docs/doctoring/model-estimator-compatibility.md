# Model–estimator compatibility contract

## Decision

`FitConfig.validate()` is the public preflight boundary for the supported
model–estimator matrix. `MIRT`, `MLS2PLM`, `MLSRM`, `ULS2PLM`, and `ULSRM`
accept both `jmle` and `mmle`. `BIFAC2PLM` accepts `mmle` only and rejects
`jmle` with a stable `ValueError` before fitting starts.

This is a capability contract, not an estimator formula change. The runtime
fitter retains its defensive late check, but callers now receive an actionable
configuration error instead of reaching an unsupported execution branch.

## Scientific rationale

The bifactor restriction is naturally handled by full-information marginal
maximum likelihood in the implementation. Bock and Aitkin (1981) establish
the EM-based marginal maximum-likelihood foundation for item-response models;
Gibbons and Hedeker (1992) describe the full-information item bifactor
restriction and its marginal-likelihood simplification. The compatibility
matrix therefore prevents an estimator identity from being advertised before
its model-specific numerical path exists.

## Verification and rollback

The focused contract tests cover the rejected `BIFAC2PLM + jmle` pair, the
accepted `BIFAC2PLM + mmle` pair, and every currently supported estimator pair
for the remaining public models. Rollback is a single change to the
configuration compatibility map; the runtime estimator implementations and
serialized fit results are unchanged.

## References (APA 7th)

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of
item parameters: Application of an EM algorithm. *Psychometrika, 46*(4),
443–459. https://doi.org/10.1007/BF02293801

Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor
analysis. *Psychometrika, 57*(3), 423–436.
https://doi.org/10.1007/BF02295430

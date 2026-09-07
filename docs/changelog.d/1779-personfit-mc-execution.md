# Execute deterministic person-fit Monte Carlo acceptance

## Fixed

Run the existing 500-replication person-fit U3 reversed-respondent acceptance in the normal Rust test suite instead of leaving it behind `#[ignore]`. The simulation keeps n=60 respondents, I=20 items, fixed-seed reproducibility, the production `person_fit_np` call, the planted hardest-half reversed respondent, and all failure replications in the denominator.

The scientific decision no longer treats a raw point estimate of 95% as sufficient evidence for a 95% detection target. The test now reports the Monte Carlo standard error and requires the lower endpoint of a two-sided 95% Wilson score interval for the detection probability to remain at or above 95%. A focused boundary regression proves that 475/500 detections—exactly 95% by point estimate—does not pass once finite-replication uncertainty is represented. The 500-replication design also keeps the Monte Carlo standard error at the 95% decision boundary below one percentage point; a failing run reports detections, replications, point estimate, MCSE, Wilson lower bound, and target.

This follows simulation-study guidance to choose replication counts from the precision required for key performance measures and to report Monte Carlo standard errors, rather than interpreting an unqualified simulation percentage as exact. The Wilson score lower bound provides the explicit finite-binomial margin used by the acceptance test.

The repository execution contract continues to inspect the attributes attached to this exact Rust test independently of attribute order and rejects both Rust `ignore` syntaxes: the MetaWord form `#[ignore]` and the MetaNameValueStr reason form `#[ignore = "..."]`. Either form removes the test from ordinary execution, so neither may silently remove this scientific acceptance from the normal suite.

Scientific traceability:

Morris, T. P., White, I. R., & Crowther, M. J. (2019). Using simulation studies to evaluate statistical methods. *Statistics in Medicine, 38*(11), 2074–2102. https://doi.org/10.1002/sim.8086

Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association, 22*(158), 209–212. https://doi.org/10.1080/01621459.1927.10502953

Rust syntax/behavior basis: *The Rust Reference*, “Testing attributes — The `ignore` attribute,” https://doc.rust-lang.org/reference/attributes/testing.html#the-ignore-attribute (accessed 2026-09-07).

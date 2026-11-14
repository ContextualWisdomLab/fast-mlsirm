# Replicated published-parameter diagnostic

Robitzsch (2024), *Information*, 15(11), 668, doi:10.3390/info15110668, Eq. 2, §3.1 and Table A1 NORM. Standard normal ability, N=2000, ten binary 2PL items; native binary GRM intercept = -a*b. This is not the paper’s AIC experiment, not polytomous GRM, bifactor, or DIF validation.

20 independent deterministic SplitMix64 streams (seeds 202610290000 + index*1000003), Q=121, canonical truth-independent initialization (slopes 1, empirical marginal logits). Native core artifact reused unchanged; only test seed selection and coverage exclusion changed. max_iter=500, tol=1e-4 RELATIVE: tol*(1+abs(previous likelihood)); Newton controls unchanged. Actual shared canonical GPU flock held and released, no competing lease/no waiting loop.

## Outcome: diagnostic NONPASS

8/20 converged; 12/20 returned native GPU M-step line-search failure with CPU fallback forbidden. All eight returned fits converged in 7 iterations, each with E=8/M=70 GPU dispatches and CPU numerical fallbacks=0. Failure logs retain completed GPU stage counts; failures did not return parameter estimates. Failure proportion=0.60, binomial MCSE=sqrt(.6*.4/20)=0.1095. Native stopping and fit code were not changed. No reruns or failed-seed replacement.

The table below is CONDITIONAL ON THE EIGHT SUCCESSFUL FITS; unconditional 20-replication bias/RMSE is unavailable, not zero. MCSE(bias)=sample SD(error)/sqrt(8); MCSE(RMSE)=sample SD(error²)/(2*RMSE*sqrt(8)), a delta-method approximation. 20 attempted replications are a bounded diagnostic, not a 500-replication study or scientific acceptance. Coarse test threshold passes do not override the 60% native failure rate or raw parameter errors.

| Item | Parameter | Returned n | Conditional MC bias | Conditional RMSE | MCSE bias | MCSE RMSE (delta) |
|---|---|---|---|---|---|---|
| 1 | a | 8 | -0.081446 | 0.096240 | 0.019379 | 0.021913 |
| 1 | b | 8 | -0.137723 | 0.207277 | 0.058549 | 0.060485 |
| 2 | a | 8 | -0.033365 | 0.086730 | 0.030258 | 0.017481 |
| 2 | b | 8 | -0.051151 | 0.077262 | 0.021886 | 0.015340 |
| 3 | a | 8 | -0.103855 | 0.130001 | 0.029555 | 0.027288 |
| 3 | b | 8 | -0.058897 | 0.106329 | 0.033460 | 0.017029 |
| 4 | a | 8 | -0.063903 | 0.375214 | 0.139746 | 0.069988 |
| 4 | b | 8 | -0.036872 | 0.062365 | 0.019011 | 0.012545 |
| 5 | a | 8 | -0.421077 | 0.469668 | 0.078633 | 0.079793 |
| 5 | b | 8 | -0.034825 | 0.062269 | 0.019511 | 0.011089 |
| 6 | a | 8 | -0.004912 | 0.100252 | 0.037846 | 0.024903 |
| 6 | b | 8 | -0.026070 | 0.038487 | 0.010701 | 0.007109 |
| 7 | a | 8 | -0.039908 | 0.122239 | 0.043670 | 0.037138 |
| 7 | b | 8 | -0.010320 | 0.064559 | 0.024087 | 0.012300 |
| 8 | a | 8 | -0.063907 | 0.086805 | 0.022204 | 0.021054 |
| 8 | b | 8 | 0.000974 | 0.027381 | 0.010343 | 0.006422 |
| 9 | a | 8 | 0.016433 | 0.079213 | 0.029288 | 0.012541 |
| 9 | b | 8 | 0.006817 | 0.043018 | 0.016054 | 0.004970 |
| 10 | a | 8 | -0.046572 | 0.086187 | 0.027410 | 0.016414 |
| 10 | b | 8 | -0.004512 | 0.042652 | 0.016030 | 0.006351 |

## Review disposition

- Coverage finding accepted: hardware test now requires `all(feature="gpu", not(coverage))`. Coverage-mode compile with gpu cfg but no native linkage passed, and portable simulator test passed.
- Item 10 +0.32 proposal conflicts with retained primary Table A1 extraction: its NORM cell is “− 0.32”, while item 9 is “−0.32”. Preserve -0.32. The retained primary-tables.json Table index 3 contains all 10 pairs. Fresh extractor omitted tables; direct HTTP returned 403; browser CDP handshake failed. Do not claim fresh visual verification or silently change truth to satisfy reviewer prose.
- No native fit/stop-control modifications and no study/participant data accessed.

## Artifacts

execution.json binds source/bank/core/binary SHA256, seeds, exits, timings, per-item errors and complete successful dispatch receipts. rep-01.log through rep-20.log preserve every actual native return/failure. parameter-metrics.json contains the full precision conditional metrics. run.py is the executed driver (the predecessor-list step is NOT an assertion RED and is not credited as TDD evidence). Existing smoke result remains a separate single-dataset observation.

# Regression report values: method and scope

The late-life regression caller needs sample means and sample standard deviations for centered moderator probes, and normal-Wald interval endpoints for HC contrast estimates. These are reusable numerical calculations; the caller retains its study-specific variable map and contrast vectors.

- R Core Team, *R stats: Standard Deviation* (`sd` manual, Details), specifies the `n - 1` denominator. This API accepts finite vectors with at least two observations, uses a stable online mean/second-moment update, and returns sample SD. A constant vector returns SD zero; the paper caller rejects it when constructing probes. Source: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/sd.html
- Pennsylvania State University, Department of Statistics, *STAT 501, Lesson 13: Weighted Least Squares & Logistic Regressions* (coefficient Wald confidence interval equation), gives estimate ± standard-normal critical value × SE. This API takes an explicit confidence level and returns the normal-Wald interval. It is an asymptotic interval under a supplied SE, not a Student-t interval, and does not replace two-stage bootstrap uncertainty. Source: https://online.stat.psu.edu/stat501/Lesson13

The API adds no dataset-specific constants. The paper's 95% level remains a caller choice. Existing `mokken::normal_upper_quantile` supplies the critical value; no new inverse-CDF implementation is needed.

## Source check

Both project Zotero libraries were searched before opening these public manuals. The cited sources are HTML, so PDF/printed page mapping and OCR checks do not apply.

| Claim | Opened location and operative text | Judgment and limit |
| --- | --- | --- |
| Sample SD uses `n - 1` | R stats `sd` manual, Details: “Like `var` this uses denominator `n - 1`.” | Supports the denominator. Finite input validation is this API's contract. |
| Normal-Wald endpoints | STAT 501 Lesson 13, confidence interval equation: `β̂ᵢ ± z₁₋α/₂ se(β̂ᵢ)` | Supports the algebra with a standard-normal critical value. Its example is logistic regression; using a supplied HC SE is an explicit asymptotic choice and does not establish two-stage uncertainty. |

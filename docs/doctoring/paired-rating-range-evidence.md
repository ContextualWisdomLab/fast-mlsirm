# Paired rating-range evidence for automated scoring

## Decision

`paired_rating_range_evidence` is a **descriptive paired-sample diagnostic** for an automated scorer and a paired reference rating stream using the same ordinal category system. It quantifies observed category support and dispersion without pretending to estimate an inferential generalized many-facet range-restriction parameter.

The production statistics are owned by `mlsirm-core::rating_range`; PyO3 and Python only validate/marshal and expose that Rust result.

## Why this evidence is separate from agreement

High agreement or association can coexist with systematically different scale use. For example, an automated scorer can preserve the relative ordering of essays while avoiding extreme categories, or it can use fewer distinct categories than the reference ratings. Therefore QWK, exact/adjacent agreement, Pearson/Spearman association, severity, and category-range evidence answer different questions and must remain separate.

Recent automated-writing research reinforces that rank/aggregate alignment alone can obscure diagnostically important behavior. Banno, Knill, and Gales (2026) explicitly motivate profile-based evaluation because rank-based correlations can hide analytic-dimension structure and rater effects, and they use two-facet Rasch calibration to adjust rater severity. Jiao, Song, and Lee (2025) likewise compare human and LLM writing raters using many-facet Rasch analysis rather than treating raw LLM scores as an error-free reference. These results support retaining scorer/rater behavior as measurement evidence rather than collapsing it into one correlation coefficient.

## Relation to generalized MFRM range restriction

Uto and Ueno (2020) distinguish rater severity, consistency, and **range restriction**, defining range restriction as overuse of a limited subset of rating categories; central tendency is a special case that overuses central categories. Their generalized many-facet model parameterizes these characteristics jointly.

The diagnostic implemented here does **not** reproduce that model. It does not estimate rater-specific category-transition parameters, a posterior range parameter, or an inferential probability that a rater is range-restricted. It only summarizes the exact paired validation cases supplied by the caller. A later generalized MFRM/rMFRM implementation requires its own likelihood, identification, true-parameter recovery, sparse-design validation, uncertainty, CPU/GPU parity, model-comparison and Python product contracts.

## Inputs

Let paired ratings be

\[
(A_n, R_n), \qquad n=1,\ldots,N,
\]

where `A` is the automated rating, `R` is the paired reference rating, and both use integer categories

\[
0,1,\ldots,K-1,
\]

with \(2 \le K \le 1000\) and \(N\ge 2\).

Both sequences must be the same length. Labels outside the declared category set fail closed.

## Descriptive summaries

For each rating stream \(X\in\{A,R\}\), the Rust core reports:

- observed minimum \(x_{\min}\);
- observed maximum \(x_{\max}\);
- distinct category count \(d_X\);
- observed span \(s_X=x_{\max}-x_{\min}\); and
- empirical population-divisor standard deviation

\[
\sigma_X
=
\sqrt{\frac{1}{N}\sum_{n=1}^{N}(x_n-\bar x)^2}.
\]

The population divisor is intentional: these are descriptive statistics over the complete paired validation cases supplied to this diagnostic, not an unbiased estimator of a superpopulation variance.

Relative summaries are:

\[
\text{span ratio}=\frac{s_A}{s_R},
\]

when \(s_R>0\),

\[
\text{distinct-category ratio}=\frac{d_A}{d_R},
\]

and

\[
\text{SD ratio}=\frac{\sigma_A}{\sigma_R},
\]

when \(\sigma_R>0\). Zero reference span or SD returns `None`, never NaN or infinity.

Endpoint gaps are

\[
G_L=A_{\min}-R_{\min},
\qquad
G_U=R_{\max}-A_{\max}.
\]

These are signed so one-sided truncation remains distinguishable from central compression.

## Conservative Boolean evidence

`narrower_observed_support` is true only when both

\[
s_A<s_R
\]

and

\[
d_A<d_R.
\]

This intentionally avoids calling same-span internal-category differences “narrower support.”

`central_tendency_signal` is stricter. It requires narrower observed support **and** both automated endpoints to lie strictly inside the paired reference endpoints:

\[
G_L>0 \quad\text{and}\quad G_U>0.
\]

The name is deliberately `signal`, not `diagnosis`: it is one descriptive pattern consistent with central-category compression, not a fitted rater characteristic.

## Interpretation examples

- Automated `[1,1,2,3,3]` vs reference `[0,1,2,3,4]` on five categories: narrower support and central-tendency signal are both true.
- Automated `[0,1,2,3]` vs reference `[0,1,2,3]`: full support matches; both flags are false.
- Automated `[0,1,2,3]` vs reference `[0,1,2,4]`: upper-tail support differs, but both endpoints are not inward; central-tendency signal remains false.
- A degenerate reference stream such as `[2,2,2,2]` does not identify span or SD ratios, so those fields are unavailable rather than infinite.

## Validation and evidence requirements

The feature must retain:

- hand-calculated numerical oracles;
- identical-range, middle-compression, one-sided truncation, same-span/fewer-category and degenerate-reference cases;
- strict shape/length/category/count validation;
- caller-array immutability;
- direct Python-to-Rust delegation evidence for every public field;
- Rust integration tests proving the public core contract;
- complete public rustdoc/docstrings and owned-production statement/branch coverage;
- exact-head Rust/PyO3, Python, package, GPU-no-skip, fuzz, Security Scan and SAST evidence before merge.

No universal scorer acceptance threshold is encoded.

## Related product boundary

This diagnostic supports the automated essay-scoring validation roadmap but does not itself change the governed essay validation-report schema, introduce feedback generation, add a provider SDK, or implement generalized MFRM/rMFRM. It advances the ability to detect a scorer that appears aligned on average while using a materially different portion of the scale.

## APA 7 references

Banno, S., Knill, K., & Gales, M. (2026). Towards self-referential analytic assessment: A profile-based approach to L2 writing evaluation with LLMs. In *Proceedings of the 21st Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2026)* (pp. 174–189). Association for Computational Linguistics. https://doi.org/10.18653/v1/2026.bea-1.13

Jiao, H., Song, D., & Lee, W.-C. (2025). *Comparing human and AI rater effects using the many-facet Rasch model* [Preprint]. arXiv. https://arxiv.org/abs/2505.18486

Uto, M., & Ueno, M. (2020). A generalized many-facet Rasch model and its Bayesian estimation using Hamiltonian Monte Carlo. *Behaviormetrika, 47*, 469–496. https://doi.org/10.1007/s41237-020-00115-7

# Paired Rating Range Evidence Design

## Status

Approved as the next bounded validation slice under issue #397. This design adds a conservative, descriptive paired-sample diagnostic for automated-versus-reference rating range use. It does **not** introduce a new likelihood, estimate a generalized many-facet range-restriction parameter, or claim that one held-out sample establishes a stable rater trait.

## Buyer-visible gap

The governed essay validation report already exposes agreement, descriptive correlation, overall/subgroup standardized mean differences, and human-human degradation. Issue #397 also requires detection of an injected range-compressed rater. The current product has no explicit evidence object for the common pattern where an automated scorer uses fewer score categories, fails to reach one or both reference endpoints, or exhibits materially narrower observed dispersion on the **same paired responses**.

Recent automated-essay-scoring evidence makes this gap operationally relevant. Jiao, Song, and Lee (2026) evaluate LLM rater effects with Many-Facet Rasch measurement rather than treating aggregate agreement as the entire validity argument. A separate 2026 human-versus-GPT essay study reports high internal consistency together with upper-end range compression, showing why agreement and range use must be reported separately. Uto and Ueno (2020) define range restriction as overuse of a limited number of categories and show that conventional MFRM with one common rating scale cannot represent rater-specific category-usage structure.

## Selected approach

Add one Rust-owned paired categorical diagnostic and a thin typed Python wrapper:

```text
paired automated labels + paired reference labels
                    │
                    ▼
       Rust paired range evidence
                    │
                    ▼
          immutable Python result
                    │
                    ▼
  later essay validation report adapter
```

The first slice deliberately uses **paired labels on the identical validation cases**. That controls gross case-mix differences between the automated and reference distributions without pretending that observed category use identifies a population rater parameter.

A full rater-specific-threshold MFRM remains a later model-development slice. Uto and Ueno's rMFRM formulation replaces the common threshold vector with rater-specific transition parameters `d_rk`; that is the appropriate inferential model for range restriction, but it requires its own Rust estimator, identification, recovery, CPU/GPU evidence, and relation-safe model comparison. This PR must not smuggle that model in as an unvalidated heuristic.

## Public contract

Rust core:

```rust
pub struct PairedRatingRangeEvidence {
    pub sample_size: usize,
    pub automated_min: usize,
    pub automated_max: usize,
    pub reference_min: usize,
    pub reference_max: usize,
    pub automated_distinct_categories: usize,
    pub reference_distinct_categories: usize,
    pub automated_span: usize,
    pub reference_span: usize,
    pub automated_sd: f64,
    pub reference_sd: f64,
    pub span_ratio: Option<f64>,
    pub distinct_category_ratio: f64,
    pub sd_ratio: Option<f64>,
    pub lower_endpoint_gap: i64,
    pub upper_endpoint_gap: i64,
    pub narrower_observed_support: bool,
    pub central_tendency_signal: bool,
}

pub fn paired_rating_range_evidence(
    automated: &[u32],
    reference: &[u32],
    category_count: usize,
) -> Result<PairedRatingRangeEvidence, String>;
```

Python:

```python
@dataclass(frozen=True)
class RatingRangeEvidence:
    ...


def paired_rating_range_evidence(
    automated: np.ndarray,
    reference: np.ndarray,
    *,
    category_count: int,
) -> RatingRangeEvidence:
    ...
```

All arithmetic is delegated to Rust. Python validates/marshals only and must not recompute the statistics.

## Definitions

For the paired automated labels `A` and reference labels `H`:

- observed span: `max(label) - min(label)`;
- distinct-category count: number of categories actually represented;
- dispersion: empirical standard deviation on the common ordinal category-number scale;
- span ratio: `span(A) / span(H)` when `span(H) > 0`, otherwise unavailable;
- distinct-category ratio: `distinct(A) / distinct(H)`;
- SD ratio: `SD(A) / SD(H)` when `SD(H) > 0`, otherwise unavailable;
- lower endpoint gap: `min(A) - min(H)`;
- upper endpoint gap: `max(H) - max(A)`.

The conservative Boolean `narrower_observed_support` is true only when both:

1. the automated observed span is strictly smaller than the reference span; and
2. the automated scorer uses strictly fewer distinct categories on the same cases.

`central_tendency_signal` is even stricter: both automated endpoints lie strictly inside the paired reference endpoints.

These flags describe the **observed held-out sample**. They do not estimate population range-restriction severity, prove construct underrepresentation, or authorize scorer rejection by themselves.

## Degenerate reference behavior

A reference sample with one observed category is valid descriptive data but cannot identify a relative span or SD ratio. In that case:

- `span_ratio = None` when the reference span is zero;
- `sd_ratio = None` when the reference SD is zero;
- endpoint/distinct-category evidence is still returned;
- the function does not fabricate `0`, infinity, or a pass/fail verdict.

## Validation and resource bounds

- vectors must be one-dimensional, paired, non-empty, and contain at least two observations;
- `category_count` must be an exact integer in the package-supported category range;
- every label must be an integer in `0..category_count-1`;
- booleans, negative values, fractional values, and out-of-range labels fail closed at the Python boundary;
- Rust validates again and never trusts the wrapper;
- the existing package vector-size/resource conventions apply;
- errors contain no essay text, prompt text, or provider-controlled content.

## Essay reporting boundary

The core evidence API lands first. A subsequent same-issue slice may add the metrics to `EssayValidationEvidenceReport` and route `narrower_observed_support` to human review **only after** the report schema, `ValidationPolicy.metric_ids`, backward compatibility, and interpretation language are reviewed together.

This avoids silently changing every existing validation report merely to land the numerical primitive.

## Testing

Permanent tests must prove:

1. exact hand-calculated metrics for a paired range-compressed scorer;
2. a matched full-range scorer is not signaled;
3. upper-end-only compression is reflected by a positive upper endpoint gap;
4. central tendency requires both endpoints to move inward;
5. same span but fewer internal categories does not satisfy the conservative combined signal;
6. zero-span/zero-SD reference returns unavailable ratios rather than NaN/Inf;
7. invalid shape, length, categories, and category count fail closed;
8. Python output fields exactly equal the raw Rust result;
9. mutating source arrays after the call cannot mutate the result; and
10. added Rust and Python production branches receive complete coverage and public documentation.

## Scientific interpretation boundary

Observed paired range evidence complements rather than replaces the existing MFRM severity estimate. Severity shifts the score location; range restriction concerns category-scale use. The two effects must not be conflated.

The later inferential range-restriction model should follow the rMFRM family with rater-specific transition parameters and must be judged by true-parameter bias/RMSE/coverage, connected sparse designs, predictive fit, and model comparison—not by whether this descriptive signal looks plausible.

## References

Jiao, H., Song, D., & Lee, W.-C. (2026). Evaluating rater effects of large language models in automated essay scoring: GPT, Claude, Gemini, and DeepSeek. *Educational Measurement: Issues and Practice, 45*(2), e70018. https://doi.org/10.1111/emip.70018

Uto, M., & Ueno, M. (2020). A generalized many-facet Rasch model and its Bayesian estimation using Hamiltonian Monte Carlo. *Behaviormetrika, 47*, 469–496. https://doi.org/10.1007/s41237-020-00115-7

Zhang, Y., et al. (2026). Comparing GPT and human raters in essay assessment: Variability, bias, and the potential of LLM-based scoring. *Computers and Education Open, 10*, 100341. https://doi.org/10.1016/j.caeo.2026.100341

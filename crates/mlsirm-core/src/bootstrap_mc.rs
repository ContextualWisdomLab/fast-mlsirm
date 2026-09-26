//! Monte Carlo percentile and binomial order-statistic diagnostics.
//!
//! A binomial rank interval describes simulation uncertainty in a percentile
//! endpoint. It is distinct from an Andrews–Buchinsky `(pdb, tau)` rule for
//! choosing the number of bootstrap repetitions.

use crate::equating::quantile_type7;
use crate::fitstats::ln_gamma;

/// Maximum number of Monte Carlo draws accepted by the numerical and Python APIs.
pub const MAX_BOOTSTRAP_MC_DRAWS: usize = 1_000_000;

#[derive(Clone, Debug, PartialEq)]
pub struct McRankInterval {
    pub confidence: f64,
    pub count_low: usize,
    pub count_high: usize,
    pub rank_low_zero_based: usize,
    pub rank_high_zero_based: usize,
    pub attained_coverage: f64,
    pub lo: f64,
    pub hi: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct McPercentileIntervalPrecision {
    pub lower_endpoint: f64,
    pub upper_endpoint: f64,
    pub interval_halfwidth: f64,
    pub lower_rank: McRankInterval,
    pub upper_rank: McRankInterval,
    pub lower_error_bound: f64,
    pub upper_error_bound: f64,
    pub worst_error_fraction: f64,
    pub allowed_fraction: f64,
    pub meets_tolerance: bool,
}

fn validate_binomial(n: usize, p: f64) -> Result<(), String> {
    if n > MAX_BOOTSTRAP_MC_DRAWS {
        return Err(format!("n must not exceed {MAX_BOOTSTRAP_MC_DRAWS}"));
    }
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err("p must be finite and in [0, 1]".into());
    }
    Ok(())
}

fn binomial_mass(n: usize, p: f64, k: usize) -> f64 {
    if p == 0.0 {
        return f64::from(k == 0);
    }
    if p == 1.0 {
        return f64::from(k == n);
    }
    let nf = n as f64;
    let kf = k as f64;
    (ln_gamma(nf + 1.0) - ln_gamma(kf + 1.0) - ln_gamma(nf - kf + 1.0)
        + kf * p.ln()
        + (nf - kf) * (-p).ln_1p())
    .exp()
}

/// Smallest `k` with `P[Binomial(n,p) <= k] >= probability`.
pub fn binomial_quantile(n: usize, p: f64, probability: f64) -> Result<usize, String> {
    validate_binomial(n, p)?;
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err("probability must be finite and in [0, 1]".into());
    }
    if probability == 0.0 || p == 0.0 {
        return Ok(0);
    }
    if probability == 1.0 || p == 1.0 {
        return Ok(n);
    }
    let mut cdf = 0.0;
    for k in 0..=n {
        cdf += binomial_mass(n, p, k);
        if cdf >= probability {
            return Ok(k);
        }
    }
    Ok(n)
}

/// Inclusive probability `P[low <= Binomial(n,p) <= high]`.
pub fn binomial_interval_coverage(
    n: usize,
    p: f64,
    low: usize,
    high: usize,
) -> Result<f64, String> {
    validate_binomial(n, p)?;
    if low > high || high > n {
        return Err("require 0 <= low <= high <= n".into());
    }
    Ok((low..=high)
        .map(|k| binomial_mass(n, p, k))
        .sum::<f64>()
        .min(1.0))
}

/// Linear-interpolated Type-7 percentile of finite draws.
pub fn linear_percentile(values: &[f64], p: f64) -> Result<f64, String> {
    if values.is_empty()
        || values.len() > MAX_BOOTSTRAP_MC_DRAWS
        || values.iter().any(|v| !v.is_finite())
    {
        return Err(format!(
            "values must contain 1..={MAX_BOOTSTRAP_MC_DRAWS} finite draws"
        ));
    }
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err("p must be finite and in [0, 1]".into());
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    Ok(quantile_type7(&ordered, p))
}

/// Central binomial count interval mapped to zero-based order-statistic ranks.
///
/// For a target percentile `p`, `K ~ Binomial(B,p)`. The central count limits
/// `L,U` map to observed order statistics at ranks `L-1,U`. If either rank
/// falls outside the observed draws, no finite interval has the requested
/// coverage and the function asks the caller for more draws.
pub fn mc_rank_interval(
    values: &[f64],
    percentile: f64,
    confidence: f64,
) -> Result<McRankInterval, String> {
    let n = values.len();
    if n < 2 || n > MAX_BOOTSTRAP_MC_DRAWS || values.iter().any(|v| !v.is_finite()) {
        return Err(format!(
            "values must contain 2..={MAX_BOOTSTRAP_MC_DRAWS} finite draws"
        ));
    }
    if !confidence.is_finite() || !(0.0..1.0).contains(&confidence) {
        return Err("confidence must be finite and in (0, 1)".into());
    }
    if !percentile.is_finite() || !(0.0..1.0).contains(&percentile) {
        return Err("percentile must be finite and in (0, 1)".into());
    }
    let alpha = (1.0 - confidence) / 2.0;
    let count_low = binomial_quantile(n, percentile, alpha)?;
    let upper_probability = 1.0 - alpha;
    let count_high = if upper_probability == 1.0 {
        let mut upper_tail = 0.0;
        let mut upper_count = n;
        for k in (0..n).rev() {
            upper_tail += binomial_mass(n, percentile, k + 1);
            if upper_tail > alpha {
                break;
            }
            upper_count = k;
        }
        upper_count
    } else {
        binomial_quantile(n, percentile, upper_probability)?
    };
    if count_low == 0 || count_high == n {
        return Err("rank interval extends beyond observed draws; increase B".into());
    }
    let rank_low_zero_based = count_low - 1;
    let rank_high_zero_based = count_high;
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    Ok(McRankInterval {
        confidence,
        count_low,
        count_high,
        rank_low_zero_based,
        rank_high_zero_based,
        attained_coverage: binomial_interval_coverage(n, percentile, count_low, count_high)?,
        lo: ordered[rank_low_zero_based],
        hi: ordered[rank_high_zero_based],
    })
}

/// Bound Monte Carlo error of both percentile endpoints relative to interval half-width.
///
/// Rank bounds use the binomial order-statistic construction for independent
/// draws (Lu, 2020, NIST TN 2119, sec. 5.3). `confidence` is per endpoint;
/// callers choose any simultaneous-coverage adjustment and tolerance.
/// This calculation does not assess bootstrap-refit validity or sampling error.
pub fn mc_percentile_interval_precision(
    values: &[f64],
    lower_percentile: f64,
    upper_percentile: f64,
    confidence: f64,
    allowed_fraction: f64,
) -> Result<McPercentileIntervalPrecision, String> {
    if !lower_percentile.is_finite()
        || !upper_percentile.is_finite()
        || !(0.0 < lower_percentile
            && lower_percentile < upper_percentile
            && upper_percentile < 1.0)
    {
        return Err("require 0 < lower_percentile < upper_percentile < 1".into());
    }
    if !allowed_fraction.is_finite() || allowed_fraction < 0.0 {
        return Err("allowed_fraction must be finite and nonnegative".into());
    }
    let lower_endpoint = linear_percentile(values, lower_percentile)?;
    let upper_endpoint = linear_percentile(values, upper_percentile)?;
    let lower_rank = mc_rank_interval(values, lower_percentile, confidence)?;
    let upper_rank = mc_rank_interval(values, upper_percentile, confidence)?;
    let width = upper_endpoint - lower_endpoint;
    let interval_halfwidth = if width.is_finite() {
        width / 2.0
    } else {
        upper_endpoint / 2.0 - lower_endpoint / 2.0
    };
    if !interval_halfwidth.is_finite() || interval_halfwidth <= 0.0 {
        return Err("percentile interval half-width must be positive and finite".into());
    }
    let lower_error_bound = (lower_endpoint - lower_rank.lo)
        .abs()
        .max((lower_rank.hi - lower_endpoint).abs());
    let upper_error_bound = (upper_endpoint - upper_rank.lo)
        .abs()
        .max((upper_rank.hi - upper_endpoint).abs());
    let worst_error_fraction = lower_error_bound.max(upper_error_bound) / interval_halfwidth;
    if !worst_error_fraction.is_finite() {
        return Err("endpoint error fraction must be finite".into());
    }
    Ok(McPercentileIntervalPrecision {
        lower_endpoint,
        upper_endpoint,
        interval_halfwidth,
        lower_rank,
        upper_rank,
        lower_error_bound,
        upper_error_bound,
        worst_error_fraction,
        allowed_fraction,
        meets_tolerance: worst_error_fraction <= allowed_fraction,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binomial_small_cases_and_validation() {
        assert_eq!(binomial_quantile(10, 0.5, 0.025).unwrap(), 2);
        assert_eq!(binomial_quantile(10, 0.5, 0.975).unwrap(), 8);
        assert!((binomial_interval_coverage(10, 0.5, 2, 8).unwrap() - 0.978515625).abs() < 1e-12);
        assert!(binomial_quantile(10, f64::NAN, 0.5).is_err());
        assert!(binomial_quantile(10, 0.5, f64::INFINITY).is_err());
    }

    #[test]
    fn ranks_and_type7_percentiles() {
        let values = [5.0, 1.0, 4.0, 2.0, 3.0];
        assert_eq!(linear_percentile(&values, 0.25).unwrap(), 2.0);
        assert_eq!(linear_percentile(&values, 0.125).unwrap(), 1.5);
        assert_eq!(linear_percentile(&[-1.0e308, 1.0e308], 0.5).unwrap(), 0.0);
        let interval = mc_rank_interval(&values, 0.5, 0.8).unwrap();
        assert_eq!((interval.count_low, interval.count_high), (1, 4));
        assert_eq!(
            (interval.rank_low_zero_based, interval.rank_high_zero_based),
            (0, 4)
        );
        assert_eq!((interval.lo, interval.hi), (1.0, 5.0));
        assert!((interval.attained_coverage - 0.9375).abs() < 1e-12);
        assert!(mc_rank_interval(&[1.0, f64::NAN], 0.5, 0.95).is_err());
        assert!(mc_rank_interval(&[1.0, 2.0], 0.5, 0.95)
            .unwrap_err()
            .contains("increase B"));

        let extreme_confidence = f64::from_bits(1.0_f64.to_bits() - 1);
        let many_values: Vec<f64> = (0..100).map(f64::from).collect();
        let extreme_interval = mc_rank_interval(&many_values, 0.5, extreme_confidence).unwrap();
        assert!(extreme_interval.count_high < many_values.len());
        assert_eq!(
            extreme_interval.count_high,
            many_values.len() - extreme_interval.count_low
        );
    }

    #[test]
    fn percentile_precision_requires_finite_observed_rank_bounds() {
        let values: Vec<f64> = (0..100).map(f64::from).collect();
        let loose = mc_percentile_interval_precision(&values, 0.25, 0.75, 0.8, 1.0).unwrap();
        assert_eq!((loose.lower_endpoint, loose.upper_endpoint), (24.75, 74.25));
        assert!(loose.worst_error_fraction > 0.0);
        assert!(loose.meets_tolerance);
        assert!(
            !mc_percentile_interval_precision(&values, 0.25, 0.75, 0.8, 0.0)
                .unwrap()
                .meets_tolerance
        );
        assert!(mc_percentile_interval_precision(&values, 0.75, 0.25, 0.8, 1.0).is_err());
        assert!(mc_percentile_interval_precision(&[1.0; 100], 0.25, 0.75, 0.8, 1.0).is_err());
        assert!(
            mc_percentile_interval_precision(&values[..5], 0.025, 0.975, 0.995, 1.0)
                .unwrap_err()
                .contains("increase B")
        );
    }
}

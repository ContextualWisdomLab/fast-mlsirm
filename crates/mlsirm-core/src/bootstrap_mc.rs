//! Monte Carlo percentile and binomial order-statistic diagnostics.
//!
//! A binomial rank interval describes simulation uncertainty in a percentile
//! endpoint. It is distinct from an Andrews–Buchinsky `(pdb, tau)` rule for
//! choosing the number of bootstrap repetitions.

use crate::equating::quantile_type7;
use crate::fitstats::ln_gamma;

const MAX_DRAWS: usize = 1_000_000;

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

fn validate_binomial(n: usize, p: f64) -> Result<(), String> {
    if n > MAX_DRAWS {
        return Err(format!("n must not exceed {MAX_DRAWS}"));
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
    if values.is_empty() || values.len() > MAX_DRAWS || values.iter().any(|v| !v.is_finite()) {
        return Err(format!("values must contain 1..={MAX_DRAWS} finite draws"));
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
    if n < 2 || n > MAX_DRAWS || values.iter().any(|v| !v.is_finite()) {
        return Err(format!("values must contain 2..={MAX_DRAWS} finite draws"));
    }
    if !confidence.is_finite() || !(0.0..1.0).contains(&confidence) {
        return Err("confidence must be finite and in (0, 1)".into());
    }
    if !percentile.is_finite() || !(0.0..1.0).contains(&percentile) {
        return Err("percentile must be finite and in (0, 1)".into());
    }
    let alpha = (1.0 - confidence) / 2.0;
    let count_low = binomial_quantile(n, percentile, alpha)?;
    let count_high = binomial_quantile(n, percentile, 1.0 - alpha)?;
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
    }
}

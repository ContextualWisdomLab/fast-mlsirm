//! Rotation-criterion registry and analytic gradients.
//!
//! The formulas follow the public GPArotation criterion contract: every
//! criterion returns a scalar that is minimized and the derivative with respect
//! to the rotated pattern matrix. The optimizer is deliberately separate from
//! this registry, so new criteria can be added without duplicating manifold or
//! multi-start logic.

use super::matrix::{crossprod, matmul, positive_logdet_inverse};
use super::RotationMode;

/// A built-in exploratory factor-rotation criterion.
#[derive(Clone, Debug)]
pub enum RotationCriterion {
    /// Crawford-Ferguson family; named special cases are selected by `kappa`.
    CrawfordFerguson { kappa: f64 },
    /// Orthomax family, with quartimax at `gamma=0` and varimax at `gamma=1`.
    Orthomax { gamma: f64 },
    /// Direct oblimin family, with quartimin at `gamma=0`.
    Oblimin { gamma: f64 },
    /// Geomin criterion.
    Geomin { delta: f64 },
    /// Complete or partially specified target criterion.
    ///
    /// `NaN` target cells are ignored and weights must be exactly zero or one,
    /// matching GPArotation PST semantics. Continuous weights belong to a
    /// separately named extension and are not accepted by this variant.
    Target { target: Vec<f64>, weights: Vec<f64> },
    /// Minimum entropy.
    Entropy,
    /// Infomax.
    Infomax,
    /// McCammon minimum entropy ratio.
    McCammon,
    /// Simplimax, targeting the requested number of smallest squared loadings.
    Simplimax { zeros: usize },
    /// Jennrich-Bentler biquartimin criterion (first column is general).
    Bifactor,
    /// Jennrich-Bentler bi-geomin criterion (first column is general).
    BiGeomin { delta: f64 },
    /// Tandem criterion I.
    TandemI,
    /// Tandem criterion II.
    TandemII,
    /// Oblimax.
    Oblimax,
    /// Bentler invariant pattern simplicity.
    Bentler,
    /// Orthogonal quartimax.
    Quartimax,
    /// Orthogonal varimax.
    Varimax,
    /// Orthogonal varimin.
    Varimin,
    /// Weighted L2 criterion used by iterative Lp/forced-simple-structure methods.
    LpWls { weights: Vec<f64> },
}

/// Scalar value and loading-space gradient for one criterion evaluation.
#[derive(Clone, Debug)]
pub struct CriterionEvaluation {
    /// Criterion value to minimize.
    pub value: f64,
    /// Row-major derivative with respect to the pattern matrix.
    pub gradient: Vec<f64>,
}

impl RotationCriterion {
    /// Stable public identifier.
    pub fn name(&self) -> &'static str {
        match self {
            Self::CrawfordFerguson { .. } => "crawford_ferguson",
            Self::Orthomax { .. } => "orthomax",
            Self::Oblimin { .. } => "oblimin",
            Self::Geomin { .. } => "geomin",
            Self::Target { .. } => "target",
            Self::Entropy => "entropy",
            Self::Infomax => "infomax",
            Self::McCammon => "mccammon",
            Self::Simplimax { .. } => "simplimax",
            Self::Bifactor => "bifactor",
            Self::BiGeomin { .. } => "bigeomin",
            Self::TandemI => "tandem_i",
            Self::TandemII => "tandem_ii",
            Self::Oblimax => "oblimax",
            Self::Bentler => "bentler",
            Self::Quartimax => "quartimax",
            Self::Varimax => "varimax",
            Self::Varimin => "varimin",
            Self::LpWls { .. } => "lp_wls",
        }
    }

    /// Whether a criterion is defined for the requested rotation manifold.
    pub fn supports(&self, mode: RotationMode) -> bool {
        match self {
            Self::Orthomax { .. }
            | Self::Quartimax
            | Self::Varimax
            | Self::Varimin
            | Self::McCammon
            | Self::TandemI
            | Self::TandemII => mode == RotationMode::Orthogonal,
            Self::Oblimin { .. } | Self::Simplimax { .. } | Self::Oblimax => {
                mode == RotationMode::Oblique
            }
            _ => true,
        }
    }

    /// Whether factor column identity is part of the criterion specification.
    pub(crate) fn has_labelled_columns(&self) -> bool {
        matches!(self, Self::Target { .. } | Self::LpWls { .. })
    }

    /// Whether column zero is a fixed general-factor role.
    pub(crate) fn fixes_general_factor(&self) -> bool {
        matches!(self, Self::Bifactor | Self::BiGeomin { .. })
    }

    /// Validate dimensions and criterion-specific hyperparameters.
    pub(crate) fn validate(&self, rows: usize, factors: usize) -> Result<(), String> {
        match self {
            Self::CrawfordFerguson { kappa } => {
                if !kappa.is_finite() || !(0.0..=1.0).contains(kappa) {
                    return Err("crawford_ferguson kappa must be finite and in [0, 1]".into());
                }
            }
            Self::Orthomax { gamma } | Self::Oblimin { gamma } => {
                if !gamma.is_finite() {
                    return Err("rotation gamma must be finite".into());
                }
            }
            Self::Geomin { delta } => {
                if !delta.is_finite() || *delta <= 0.0 {
                    return Err("geomin delta must be finite and positive".into());
                }
            }
            Self::BiGeomin { delta } => {
                if !delta.is_finite() || *delta <= 0.0 {
                    return Err("geomin delta must be finite and positive".into());
                }
                if factors < 3 {
                    return Err("bifactor rotation requires at least three factors".into());
                }
            }
            Self::Target { target, weights } => {
                if target.len() != rows * factors || weights.len() != rows * factors {
                    return Err("target and weights must match the loading-matrix shape".into());
                }
                let mut specified = 0_usize;
                for (target_value, weight) in target.iter().zip(weights) {
                    if !weight.is_finite() || !matches!(*weight, 0.0 | 1.0) {
                        return Err("target weights must be binary zero-or-one values".into());
                    }
                    if target_value.is_finite() && *weight == 1.0 {
                        specified += 1;
                    } else if !target_value.is_nan() && !target_value.is_finite() {
                        return Err("target cells must be finite or NaN".into());
                    }
                }
                if specified == 0 {
                    return Err(
                        "target rotation requires at least one specified weighted cell".into(),
                    );
                }
            }
            Self::Simplimax { zeros } => {
                if *zeros == 0 || *zeros > rows * factors {
                    return Err("simplimax zeros must be in 1..=rows*factors".into());
                }
            }
            Self::Bifactor => {
                if factors < 3 {
                    return Err("bifactor rotation requires at least three factors".into());
                }
            }
            Self::LpWls { weights } => {
                if weights.len() != rows * factors {
                    return Err("lp_wls weights must match the loading-matrix shape".into());
                }
                if weights
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
                {
                    return Err("lp_wls weights must be finite and non-negative".into());
                }
            }
            Self::Entropy
            | Self::Infomax
            | Self::McCammon
            | Self::TandemI
            | Self::TandemII
            | Self::Oblimax
            | Self::Bentler
            | Self::Quartimax
            | Self::Varimax
            | Self::Varimin => {}
        }
        Ok(())
    }

    /// Evaluate the criterion and its analytic loading-space gradient.
    pub fn evaluate(
        &self,
        loadings: &[f64],
        rows: usize,
        factors: usize,
    ) -> Result<CriterionEvaluation, String> {
        if loadings.len() != rows * factors || loadings.iter().any(|x| !x.is_finite()) {
            return Err("criterion loadings must be a finite rows x factors matrix".into());
        }
        self.validate(rows, factors)?;
        let result = match self {
            Self::CrawfordFerguson { kappa } => cf(loadings, rows, factors, *kappa),
            Self::Orthomax { gamma } => orthomax(loadings, rows, factors, *gamma),
            Self::Oblimin { gamma } => oblimin(loadings, rows, factors, *gamma),
            Self::Geomin { delta } => geomin(loadings, rows, factors, *delta, 0),
            Self::Target {
                target: target_values,
                weights,
            } => target(loadings, target_values, weights),
            Self::Entropy => entropy(loadings),
            Self::Infomax => infomax(loadings, rows, factors),
            Self::McCammon => mccammon(loadings, rows, factors)?,
            Self::Simplimax { zeros } => simplimax(loadings, *zeros),
            Self::Bifactor => bifactor(loadings, rows, factors),
            Self::BiGeomin { delta } => geomin(loadings, rows, factors, *delta, 1),
            Self::TandemI => tandem(loadings, rows, factors, true),
            Self::TandemII => tandem(loadings, rows, factors, false),
            Self::Oblimax => oblimax(loadings)?,
            Self::Bentler => bentler(loadings, rows, factors)?,
            Self::Quartimax => quartimax(loadings),
            Self::Varimax => varimax(loadings, rows, factors, false),
            Self::Varimin => varimax(loadings, rows, factors, true),
            Self::LpWls { weights } => lp_wls(loadings, weights, rows),
        };
        if !result.value.is_finite() || result.gradient.iter().any(|x| !x.is_finite()) {
            return Err(format!(
                "{} produced a non-finite value or gradient",
                self.name()
            ));
        }
        Ok(result)
    }
}

fn cf(l: &[f64], rows: usize, factors: usize, kappa: f64) -> CriterionEvaluation {
    let squared: Vec<f64> = l.iter().map(|x| x * x).collect();
    let mut row_sum = vec![0.0; rows];
    let mut col_sum = vec![0.0; factors];
    for i in 0..rows {
        for j in 0..factors {
            let value = squared[i * factors + j];
            row_sum[i] += value;
            col_sum[j] += value;
        }
    }
    let mut value = 0.0;
    let mut gradient = vec![0.0; l.len()];
    for i in 0..rows {
        for j in 0..factors {
            let idx = i * factors + j;
            let row_other = row_sum[i] - squared[idx];
            let col_other = col_sum[j] - squared[idx];
            value += 0.25 * squared[idx] * ((1.0 - kappa) * row_other + kappa * col_other);
            gradient[idx] = l[idx] * ((1.0 - kappa) * row_other + kappa * col_other);
        }
    }
    CriterionEvaluation { value, gradient }
}

fn orthomax(l: &[f64], rows: usize, factors: usize, gamma: f64) -> CriterionEvaluation {
    let mut col_sum = vec![0.0; factors];
    let mut fourth = 0.0;
    for i in 0..rows {
        for j in 0..factors {
            let square = l[i * factors + j] * l[i * factors + j];
            col_sum[j] += square;
            fourth += square * square;
        }
    }
    let correction: f64 = col_sum.iter().map(|x| x * x).sum();
    let value = -0.25 * (fourth - gamma * correction / rows as f64);
    let mut gradient = vec![0.0; l.len()];
    for i in 0..rows {
        for j in 0..factors {
            let idx = i * factors + j;
            gradient[idx] = -l[idx] * (l[idx] * l[idx] - gamma * col_sum[j] / rows as f64);
        }
    }
    CriterionEvaluation { value, gradient }
}

fn oblimin(l: &[f64], rows: usize, factors: usize, gamma: f64) -> CriterionEvaluation {
    let squared: Vec<f64> = l.iter().map(|x| x * x).collect();
    let mut x = vec![0.0; l.len()];
    for i in 0..rows {
        let row_sum: f64 = squared[i * factors..(i + 1) * factors].iter().sum();
        for j in 0..factors {
            x[i * factors + j] = row_sum - squared[i * factors + j];
        }
    }
    if gamma != 0.0 {
        for j in 0..factors {
            let mut column_sum = 0.0;
            for i in 0..rows {
                column_sum += x[i * factors + j];
            }
            let adjustment = gamma * column_sum / rows as f64;
            for i in 0..rows {
                x[i * factors + j] -= adjustment;
            }
        }
    }
    let value = 0.25 * squared.iter().zip(&x).map(|(a, b)| a * b).sum::<f64>();
    let gradient = l.iter().zip(x).map(|(a, b)| a * b).collect();
    CriterionEvaluation { value, gradient }
}

fn geomin(
    l: &[f64],
    rows: usize,
    factors: usize,
    delta: f64,
    skip_columns: usize,
) -> CriterionEvaluation {
    let active = factors - skip_columns;
    let mut value = 0.0;
    let mut gradient = vec![0.0; l.len()];
    for i in 0..rows {
        let mut log_mean = 0.0;
        for j in skip_columns..factors {
            let x = l[i * factors + j];
            log_mean += (x * x + delta).ln();
        }
        let product = (log_mean / active as f64).exp();
        value += product;
        for j in skip_columns..factors {
            let idx = i * factors + j;
            gradient[idx] = 2.0 * l[idx] * product / (active as f64 * (l[idx] * l[idx] + delta));
        }
    }
    CriterionEvaluation { value, gradient }
}

fn target(l: &[f64], target: &[f64], weights: &[f64]) -> CriterionEvaluation {
    let mut value = 0.0;
    let mut gradient = vec![0.0; l.len()];
    for idx in 0..l.len() {
        if target[idx].is_nan() || weights[idx] == 0.0 {
            continue;
        }
        let residual = l[idx] - target[idx];
        value += weights[idx] * residual * residual;
        gradient[idx] = 2.0 * weights[idx] * residual;
    }
    CriterionEvaluation { value, gradient }
}

fn entropy(l: &[f64]) -> CriterionEvaluation {
    let eps = f64::EPSILON;
    let mut value = 0.0;
    let mut gradient = vec![0.0; l.len()];
    for (idx, loading) in l.iter().copied().enumerate() {
        let square = loading * loading;
        let log_square = (square + eps).ln();
        value -= 0.5 * square * log_square;
        gradient[idx] = -(loading * log_square + loading);
    }
    CriterionEvaluation { value, gradient }
}

fn infomax(l: &[f64], rows: usize, factors: usize) -> CriterionEvaluation {
    let eps = f64::EPSILON;
    let squared: Vec<f64> = l.iter().map(|x| x * x).collect();
    let total: f64 = squared.iter().sum();
    let mut row_sum = vec![0.0; rows];
    let mut col_sum = vec![0.0; factors];
    for i in 0..rows {
        for j in 0..factors {
            let x = squared[i * factors + j];
            row_sum[i] += x;
            col_sum[j] += x;
        }
    }
    let e: Vec<f64> = squared.iter().map(|x| x / total).collect();
    let e1: Vec<f64> = row_sum.iter().map(|x| x / total).collect();
    let e2: Vec<f64> = col_sum.iter().map(|x| x / total).collect();
    let q0 = -e.iter().map(|x| x * (x + eps).ln()).sum::<f64>();
    let q1 = -e1.iter().map(|x| x * (x + eps).ln()).sum::<f64>();
    let q2 = -e2.iter().map(|x| x * (x + eps).ln()).sum::<f64>();
    let h: Vec<f64> = e.iter().map(|x| -((x + eps).ln() + 1.0)).collect();
    let h1: Vec<f64> = e1.iter().map(|x| -((x + eps).ln() + 1.0)).collect();
    let h2: Vec<f64> = e2.iter().map(|x| -((x + eps).ln() + 1.0)).collect();
    let center0 = squared.iter().zip(&h).map(|(x, y)| x * y).sum::<f64>() / total;
    let center1 = row_sum.iter().zip(&h1).map(|(x, y)| x * y).sum::<f64>() / total;
    let center2 = col_sum.iter().zip(&h2).map(|(x, y)| x * y).sum::<f64>() / total;
    let mut gradient = vec![0.0; l.len()];
    for i in 0..rows {
        for j in 0..factors {
            let idx = i * factors + j;
            let g0 = (h[idx] - center0) / total;
            let g1 = (h1[i] - center1) / total;
            let g2 = (h2[j] - center2) / total;
            gradient[idx] = 2.0 * l[idx] * (g0 - g1 - g2);
        }
    }
    CriterionEvaluation {
        value: (factors as f64).ln() + q0 - q1 - q2,
        gradient,
    }
}

fn mccammon(l: &[f64], rows: usize, factors: usize) -> Result<CriterionEvaluation, String> {
    let squared: Vec<f64> = l.iter().map(|x| x * x).collect();
    let mut col_sum = vec![0.0; factors];
    for i in 0..rows {
        for j in 0..factors {
            col_sum[j] += squared[i * factors + j];
        }
    }
    let total: f64 = col_sum.iter().sum();
    if total <= 0.0 || col_sum.iter().any(|x| *x <= 0.0) {
        return Err("mccammon requires nonzero variance in every factor".into());
    }
    let mut p = vec![0.0; l.len()];
    let mut log_p = vec![0.0; l.len()];
    for i in 0..rows {
        for j in 0..factors {
            let idx = i * factors + j;
            p[idx] = squared[idx] / col_sum[j];
            log_p[idx] = if p[idx] > 0.0 { p[idx].ln() } else { 0.0 };
        }
    }
    let p2: Vec<f64> = col_sum.iter().map(|x| x / total).collect();
    let log_p2: Vec<f64> = p2
        .iter()
        .map(|x| if *x > 0.0 { x.ln() } else { 0.0 })
        .collect();
    let q1 = -p.iter().zip(&log_p).map(|(x, y)| x * y).sum::<f64>();
    let q2 = -p2.iter().zip(&log_p2).map(|(x, y)| x * y).sum::<f64>();
    if q1 <= 1e-15 || q2 <= 1e-15 {
        return Err("mccammon entropy is degenerate".into());
    }
    let h1: Vec<f64> = log_p.iter().map(|x| -(x + 1.0)).collect();
    let h2: Vec<f64> = log_p2.iter().map(|x| -(x + 1.0)).collect();
    let mut alpha1 = vec![0.0; factors];
    for i in 0..rows {
        for j in 0..factors {
            let idx = i * factors + j;
            alpha1[j] += p[idx] * h1[idx];
        }
    }
    let alpha2: f64 = p2.iter().zip(&h2).map(|(x, y)| x * y).sum();
    let mut gradient = vec![0.0; l.len()];
    for i in 0..rows {
        for j in 0..factors {
            let idx = i * factors + j;
            let g1 = (h1[idx] - alpha1[j]) / col_sum[j];
            let g2 = (h2[j] - alpha2) / total;
            gradient[idx] = 2.0 * l[idx] * (g1 / q1 - g2 / q2);
        }
    }
    Ok(CriterionEvaluation {
        value: q1.ln() - q2.ln(),
        gradient,
    })
}

fn simplimax(l: &[f64], zeros: usize) -> CriterionEvaluation {
    let mut ordered: Vec<(usize, f64)> =
        l.iter().enumerate().map(|(idx, x)| (idx, x * x)).collect();
    ordered.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("finite loadings"));
    let mut value = 0.0;
    let mut gradient = vec![0.0; l.len()];
    for (idx, square) in ordered.into_iter().take(zeros) {
        value += square;
        gradient[idx] = 2.0 * l[idx];
    }
    CriterionEvaluation { value, gradient }
}

fn bifactor(l: &[f64], rows: usize, factors: usize) -> CriterionEvaluation {
    let mut value = 0.0;
    let mut gradient = vec![0.0; l.len()];
    for i in 0..rows {
        let mut row_sum = 0.0;
        for j in 1..factors {
            let square = l[i * factors + j] * l[i * factors + j];
            row_sum += square;
        }
        value += row_sum * row_sum;
        for j in 1..factors {
            let idx = i * factors + j;
            let square = l[idx] * l[idx];
            value -= square * square;
            gradient[idx] = 4.0 * l[idx] * (row_sum - square);
        }
    }
    CriterionEvaluation { value, gradient }
}

fn tandem(l: &[f64], rows: usize, factors: usize, first: bool) -> CriterionEvaluation {
    let squared: Vec<f64> = l.iter().map(|x| x * x).collect();
    let l_transpose = super::matrix::transpose(l, rows, factors);
    let ll = matmul(l, rows, factors, &l_transpose, rows);
    let ll_squared: Vec<f64> = ll.iter().map(|x| x * x).collect();
    let kernel: Vec<f64> = if first {
        ll_squared.clone()
    } else {
        ll_squared.iter().map(|x| 1.0 - x).collect()
    };
    let product = matmul(&kernel, rows, rows, &squared, factors);
    let mut value: f64 = squared.iter().zip(&product).map(|(x, y)| x * y).sum();
    if first {
        value = -value;
    }
    let mut gradient1 = vec![0.0; l.len()];
    for idx in 0..l.len() {
        gradient1[idx] = 4.0 * l[idx] * product[idx];
    }
    let squared_transpose = super::matrix::transpose(&squared, rows, factors);
    let squared_gram = matmul(&squared, rows, factors, &squared_transpose, rows);
    let hadamard: Vec<f64> = ll.iter().zip(squared_gram).map(|(x, y)| x * y).collect();
    let mut gradient2 = matmul(&hadamard, rows, rows, l, factors);
    for value in &mut gradient2 {
        *value *= 4.0;
    }
    let gradient = if first {
        gradient1
            .iter()
            .zip(gradient2)
            .map(|(a, b)| -(a + b))
            .collect()
    } else {
        gradient1
            .iter()
            .zip(gradient2)
            .map(|(a, b)| a - b)
            .collect()
    };
    CriterionEvaluation { value, gradient }
}

/// Evaluate the natural logarithm using a fixed binary64 operator sequence.
///
/// The input must be finite and strictly positive. The implementation normalizes
/// the IEEE-754 significand with exact power-of-two scaling and evaluates the
/// `atanh` series with a fixed 24-term Kahan-compensated reduction. Unlike
/// `f64::ln`, this criterion-local reference does not delegate to a
/// platform-dependent transcendental implementation.
fn deterministic_ln_positive(value: f64) -> Option<f64> {
    const FRACTION_MASK: u64 = 0x000f_ffff_ffff_ffff;
    const LN_2: f64 = f64::from_bits(0x3fe6_2e42_fefa_39ef);
    const SQRT_2: f64 = f64::from_bits(0x3ff6_a09e_667f_3bcd);

    if !value.is_finite() || value <= 0.0 {
        return None;
    }

    let bits = value.to_bits();
    let exponent_field = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & FRACTION_MASK;
    let (mut mantissa, mut exponent) = if exponent_field == 0 {
        let bit_index = 63_i32 - fraction.leading_zeros() as i32;
        let shift = 52_i32 - bit_index;
        let normalized_fraction = (fraction << shift as u32) & FRACTION_MASK;
        (
            f64::from_bits((1023_u64 << 52) | normalized_fraction),
            bit_index - 1074,
        )
    } else {
        (
            f64::from_bits((1023_u64 << 52) | fraction),
            exponent_field - 1023,
        )
    };

    if mantissa > SQRT_2 {
        mantissa *= 0.5;
        exponent += 1;
    }

    let y = (mantissa - 1.0) / (mantissa + 1.0);
    let y_squared = y * y;
    let mut term = y;
    let mut sum = 0.0;
    let mut correction = 0.0;
    let mut denominator = 1.0;
    for _ in 0..24 {
        let addend = term / denominator;
        let adjusted = addend - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
        term *= y_squared;
        denominator += 2.0;
    }

    Some(2.0 * sum + exponent as f64 * LN_2)
}

fn oblimax(l: &[f64]) -> Result<CriterionEvaluation, String> {
    let mut sum2 = 0.0;
    let mut sum4 = 0.0;
    for x in l.iter().copied() {
        let square = x * x;
        sum2 += square;
        sum4 += square * square;
    }
    if sum2 <= 0.0 || sum4 <= 0.0 {
        return Err("oblimax requires nonzero loadings".into());
    }
    let log_sum2 = deterministic_ln_positive(sum2)
        .ok_or_else(|| "oblimax second moment must remain finite and positive".to_string())?;
    let log_sum4 = deterministic_ln_positive(sum4)
        .ok_or_else(|| "oblimax fourth moment must remain finite and positive".to_string())?;
    let gradient = l
        .iter()
        .map(|x| {
            let cube = (x * x) * x;
            -(4.0 * cube / sum4 - 4.0 * x / sum2)
        })
        .collect();
    Ok(CriterionEvaluation {
        value: -(log_sum4 - 2.0 * log_sum2),
        gradient,
    })
}

fn bentler(l: &[f64], rows: usize, factors: usize) -> Result<CriterionEvaluation, String> {
    let squared: Vec<f64> = l.iter().map(|x| x * x).collect();
    let gram = crossprod(&squared, &squared, rows, factors, factors);
    let (logdet, inverse_gram) = positive_logdet_inverse(&gram, factors)?;
    let mut logdiag = 0.0;
    let mut difference = inverse_gram;
    for j in 0..factors {
        let diagonal = gram[j * factors + j];
        if diagonal <= 0.0 {
            return Err("bentler requires positive diagonal fourth moments".into());
        }
        logdiag += diagonal.ln();
        difference[j * factors + j] -= diagonal.recip();
    }
    let product = matmul(&squared, rows, factors, &difference, factors);
    let gradient = l.iter().zip(product).map(|(x, y)| -x * y).collect();
    Ok(CriterionEvaluation {
        value: -0.25 * (logdet - logdiag),
        gradient,
    })
}

fn quartimax(l: &[f64]) -> CriterionEvaluation {
    CriterionEvaluation {
        value: -0.25 * l.iter().map(|x| x.powi(4)).sum::<f64>(),
        gradient: l.iter().map(|x| -x.powi(3)).collect(),
    }
}

fn varimax(l: &[f64], rows: usize, factors: usize, reverse: bool) -> CriterionEvaluation {
    let mut means = vec![0.0; factors];
    for i in 0..rows {
        for j in 0..factors {
            means[j] += l[i * factors + j] * l[i * factors + j] / rows as f64;
        }
    }
    let sign = if reverse { 1.0 } else { -1.0 };
    let mut value = 0.0;
    let mut gradient = vec![0.0; l.len()];
    for i in 0..rows {
        for j in 0..factors {
            let idx = i * factors + j;
            let centered = l[idx] * l[idx] - means[j];
            value += 0.25 * sign * centered * centered;
            gradient[idx] = sign * l[idx] * centered;
        }
    }
    CriterionEvaluation { value, gradient }
}

fn lp_wls(l: &[f64], weights: &[f64], rows: usize) -> CriterionEvaluation {
    let scale = rows as f64;
    CriterionEvaluation {
        value: l.iter().zip(weights).map(|(x, w)| w * x * x / scale).sum(),
        gradient: l
            .iter()
            .zip(weights)
            .map(|(x, w)| 2.0 * w * x / scale)
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix() -> Vec<f64> {
        vec![0.8, 0.2, 0.1, 0.7, 0.5, -0.4, 0.3, 0.6]
    }

    fn finite_difference(criterion: &RotationCriterion, index: usize) -> f64 {
        let mut plus = matrix();
        let mut minus = matrix();
        let h = 1e-6;
        plus[index] += h;
        minus[index] -= h;
        (criterion.evaluate(&plus, 4, 2).unwrap().value
            - criterion.evaluate(&minus, 4, 2).unwrap().value)
            / (2.0 * h)
    }

    #[test]
    fn analytic_gradients_match_finite_differences() {
        let target_values = vec![0.7, 0.0, 0.0, 0.7, f64::NAN, f64::NAN, 0.0, 0.5];
        let criteria = vec![
            RotationCriterion::CrawfordFerguson { kappa: 0.2 },
            RotationCriterion::Orthomax { gamma: 0.7 },
            RotationCriterion::Oblimin { gamma: 0.3 },
            RotationCriterion::Geomin { delta: 0.01 },
            RotationCriterion::Target {
                target: target_values,
                weights: vec![1.0; 8],
            },
            RotationCriterion::Entropy,
            RotationCriterion::Infomax,
            RotationCriterion::McCammon,
            RotationCriterion::Simplimax { zeros: 3 },
            RotationCriterion::TandemI,
            RotationCriterion::TandemII,
            RotationCriterion::Oblimax,
            RotationCriterion::Bentler,
            RotationCriterion::Quartimax,
            RotationCriterion::Varimax,
            RotationCriterion::Varimin,
            RotationCriterion::LpWls {
                weights: vec![1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0],
            },
        ];
        for criterion in criteria {
            let evaluation = criterion.evaluate(&matrix(), 4, 2).unwrap();
            let numeric = finite_difference(&criterion, 2);
            let tolerance = if matches!(criterion, RotationCriterion::Entropy) {
                1e-5
            } else {
                3e-4
            };
            assert!(
                (evaluation.gradient[2] - numeric).abs() < tolerance,
                "{} analytic={} numeric={}",
                criterion.name(),
                evaluation.gradient[2],
                numeric
            );
        }
    }

    #[test]
    fn deterministic_log_handles_domain_and_binary64_edges() {
        assert_eq!(deterministic_ln_positive(1.0), Some(0.0));
        assert_eq!(
            deterministic_ln_positive(2.0).map(f64::to_bits),
            Some(0x3fe6_2e42_fefa_39ef)
        );
        assert!(deterministic_ln_positive(f64::from_bits(1))
            .expect("smallest positive subnormal is in-domain")
            .is_finite());
        for value in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(deterministic_ln_positive(value), None);
        }
    }

    #[test]
    fn binary_target_weights_have_unsquared_weight_semantics() {
        let criterion = RotationCriterion::Target {
            target: vec![0.0; 8],
            weights: vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        };
        let evaluation = criterion.evaluate(&matrix(), 4, 2).unwrap();
        let expected = 0.8_f64.powi(2) + 0.1_f64.powi(2) + 0.5_f64.powi(2) + 0.3_f64.powi(2);
        assert!((evaluation.value - expected).abs() < 1e-12);
        assert_eq!(evaluation.gradient[1], 0.0);
    }

    #[test]
    fn bifactor_gradients_and_metadata_are_exercised() {
        let loadings = vec![0.7, 0.5, 0.1, 0.6, 0.1, 0.5, 0.8, 0.4, 0.2, 0.6, 0.2, 0.4];
        for criterion in [
            RotationCriterion::Bifactor,
            RotationCriterion::BiGeomin { delta: 0.01 },
        ] {
            let evaluation = criterion.evaluate(&loadings, 4, 3).unwrap();
            assert_eq!(evaluation.gradient[0], 0.0);
            assert!(criterion.fixes_general_factor());
            assert!(!criterion.has_labelled_columns());
            assert!(criterion.supports(RotationMode::Orthogonal));
            assert!(criterion.supports(RotationMode::Oblique));
        }
        assert!(RotationCriterion::Target {
            target: vec![0.0; 8],
            weights: vec![1.0; 8]
        }
        .has_labelled_columns());
    }

    #[test]
    fn criterion_validation_rejects_invalid_contracts() {
        assert!(RotationCriterion::CrawfordFerguson { kappa: -0.1 }
            .validate(4, 2)
            .is_err());
        assert!(RotationCriterion::Orthomax { gamma: f64::NAN }
            .validate(4, 2)
            .is_err());
        assert!(RotationCriterion::Geomin { delta: 0.0 }
            .validate(4, 2)
            .is_err());
        assert!(RotationCriterion::Target {
            target: vec![0.0],
            weights: vec![1.0]
        }
        .validate(4, 2)
        .is_err());
        assert!(RotationCriterion::Target {
            target: vec![f64::NAN; 8],
            weights: vec![1.0; 8]
        }
        .validate(4, 2)
        .is_err());
        let mut bad_target = vec![0.0; 8];
        bad_target[0] = f64::INFINITY;
        assert!(RotationCriterion::Target {
            target: bad_target,
            weights: vec![1.0; 8]
        }
        .validate(4, 2)
        .is_err());
        assert!(RotationCriterion::Target {
            target: vec![0.0; 8],
            weights: vec![-1.0; 8]
        }
        .validate(4, 2)
        .is_err());
        let mut continuous_weights = vec![1.0; 8];
        continuous_weights[0] = 0.25;
        assert!(RotationCriterion::Target {
            target: vec![0.0; 8],
            weights: continuous_weights
        }
        .validate(4, 2)
        .is_err());
        assert!(RotationCriterion::Simplimax { zeros: 0 }
            .validate(4, 2)
            .is_err());
        assert!(RotationCriterion::Bifactor.validate(4, 2).is_err());
        assert!(RotationCriterion::BiGeomin { delta: 0.01 }
            .validate(4, 2)
            .is_err());
        assert!(RotationCriterion::LpWls { weights: vec![1.0] }
            .validate(4, 2)
            .is_err());
        assert!(RotationCriterion::LpWls {
            weights: vec![f64::NAN; 8]
        }
        .validate(4, 2)
        .is_err());
        assert!(RotationCriterion::Varimax
            .evaluate(&[f64::NAN; 8], 4, 2)
            .is_err());
        assert!(!RotationCriterion::Varimax.supports(RotationMode::Oblique));
        assert!(!RotationCriterion::Oblimin { gamma: 0.0 }.supports(RotationMode::Orthogonal));
    }

    #[test]
    fn degenerate_criteria_fail_closed() {
        assert!(RotationCriterion::McCammon
            .evaluate(&[0.0; 8], 4, 2)
            .is_err());
        assert!(RotationCriterion::Oblimax
            .evaluate(&[0.0; 8], 4, 2)
            .is_err());
        let duplicated = vec![0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2];
        assert!(RotationCriterion::Bentler
            .evaluate(&duplicated, 4, 2)
            .is_err());
    }
}

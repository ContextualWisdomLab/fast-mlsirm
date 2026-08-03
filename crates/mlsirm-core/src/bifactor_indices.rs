//! Scoreability diagnostics for standardized orthogonal bifactor loadings.
//!
//! The kernel answers a different question from model-selection statistics:
//! after a bifactor structure has been selected and calibrated, do the fitted
//! loadings support interpretation of a general total score and residual
//! factor-specific scores? It computes explained-common-variance (ECV)
//! variants, item ECV, percentage of uncontaminated correlations (PUC), omega
//! total, omega hierarchical, and construct replicability `H`.
//!
//! # Source governance
//!
//! Implementation formulas were independently transcribed and verified from
//! the CRAN `BifactorIndicesCalculator` 0.2.2 source files
//! `R/ECV_Indices.R`, `R/Omega_Indices.R`, and `R/Other_Indices.R` (read in
//! full for the implemented continuous-indicator functions). Rodriguez,
//! Reise, and Haviland (2016) is cited as the methodological origin described
//! by that package; the article itself was not read in full for this change.
//! The implementation uses formula facts only and does not copy expressive R
//! source structure.
//!
//! # Metric contract
//!
//! Let `Lambda` be an item-by-factor standardized loading matrix, `Theta` the
//! vector of item uniquenesses, and `g` the explicitly supplied general-factor
//! column. Structural membership is `|lambda_if| > zero_tolerance`; numerical
//! sums retain the original loading values, including values below that
//! structural tolerance.
//!
//! For factor `f`, with membership indicator `I_if`:
//!
//! - `ECV_SS_f = sum_i I_if lambda_if^2 / sum_i I_if sum_h lambda_ih^2`.
//! - `ECV_SG_f = sum_i I_if lambda_if^2 / sum_i sum_h lambda_ih^2`.
//! - `ECV_GS_f = sum_i I_if lambda_ig^2 / sum_i I_if sum_h lambda_ih^2`.
//! - `I_ECV_i = lambda_ig^2 / sum_h lambda_ih^2`.
//! - `omega_total_f` uses all common-factor column sums over items in factor
//!   `f` in the numerator and adds their uniquenesses to the denominator.
//! - `omega_hierarchical_f` replaces that numerator by the squared loading sum
//!   for factor `f` alone, retaining the same denominator.
//! - `H_f = 1 / (1 + 1 / sum_i[lambda_if^2/(1-lambda_if^2)])`.
//!
//! PUC is defined only when every item loads on the general factor and on at
//! most one specific factor. For other loading patterns, the result returns
//! `None` rather than silently applying a strict-bifactor formula to a
//! two-tier or cross-loaded model.
//!
//! # References (APA 7th ed.)
//!
//! Dueber, D. M. (2021). *BifactorIndicesCalculator: Bifactor indices
//! calculator* (Version 0.2.2) [R package].
//! https://CRAN.R-project.org/package=BifactorIndicesCalculator
//!
//! Rodriguez, A., Reise, S. P., & Haviland, M. G. (2016). Evaluating bifactor
//! models: Calculating and interpreting statistical indices. *Psychological
//! Methods, 21*(2), 137-150. https://doi.org/10.1037/met0000045

use crate::checked_mul_usize;

/// Shape and structural-zero controls for [`bifactor_indices`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BifactorIndicesConfig {
    /// Number of item rows in the row-major loading matrix.
    pub n_items: usize,
    /// Number of factor columns in the row-major loading matrix.
    pub n_factors: usize,
    /// Zero-based column containing the general factor.
    pub general_factor: usize,
    /// Loadings with absolute value at or below this value are structural zeroes.
    pub zero_tolerance: f64,
}

impl BifactorIndicesConfig {
    /// Construct a configuration using exact zeroes for loading membership.
    pub const fn new(n_items: usize, n_factors: usize, general_factor: usize) -> Self {
        Self {
            n_items,
            n_factors,
            general_factor,
            zero_tolerance: 0.0,
        }
    }
}

/// Scoreability indices computed from one standardized loading solution.
#[derive(Clone, Debug, PartialEq)]
pub struct BifactorIndicesResult {
    /// Number of structurally non-zero item loadings on each factor.
    pub factor_item_counts: Vec<usize>,
    /// Whether every item loads on the general factor and at most one specific factor.
    pub is_strict_bifactor: bool,
    /// Percentage of uncontaminated correlations for a strict bifactor pattern.
    pub puc: Option<f64>,
    /// ECV of each factor with respect to the items loading on that factor.
    pub ecv_ss: Vec<f64>,
    /// ECV of each factor with respect to common variance across all items.
    pub ecv_sg: Vec<f64>,
    /// General-factor ECV within the item domain of each factor.
    pub ecv_gs: Vec<f64>,
    /// Proportion of each item's common variance attributable to the general factor.
    pub item_ecv: Vec<f64>,
    /// Omega total for the item set loading on each factor.
    pub omega_total: Vec<f64>,
    /// Omega hierarchical for the target factor within each factor's item set.
    pub omega_hierarchical: Vec<f64>,
    /// Construct replicability `H` for each factor.
    pub construct_replicability: Vec<f64>,
}

/// Compute continuous-indicator bifactor scoreability indices in Rust.
///
/// `loadings` is row-major with shape `n_items x n_factors` and must contain
/// finite standardized loadings with absolute value below one. `uniquenesses`
/// contains one finite non-negative item residual variance per row. The factors
/// are assumed orthogonal; callers must standardize IRT slopes before using
/// this function.
///
/// This function deliberately returns descriptive indices, not a pass/fail
/// verdict. Model selection should use likelihood, predictive, recovery, and
/// non-nested comparison evidence before interpreting these diagnostics.
pub fn bifactor_indices(
    loadings: &[f64],
    uniquenesses: &[f64],
    config: BifactorIndicesConfig,
) -> Result<BifactorIndicesResult, String> {
    validate_inputs(loadings, uniquenesses, config)?;

    let BifactorIndicesConfig {
        n_items,
        n_factors,
        general_factor,
        zero_tolerance,
    } = config;
    let active: Vec<bool> = loadings
        .iter()
        .map(|loading| loading.abs() > zero_tolerance)
        .collect();
    let squared: Vec<f64> = loadings.iter().map(|loading| loading * loading).collect();

    let mut factor_item_counts = vec![0usize; n_factors];
    let mut item_common_variance = vec![0.0_f64; n_items];
    for item in 0..n_items {
        for factor in 0..n_factors {
            let index = item * n_factors + factor;
            if active[index] && squared[index] == 0.0 {
                return Err(format!(
                    "loading magnitude is too small for stable squaring at item {item}, factor {factor}"
                ));
            }
            item_common_variance[item] += squared[index];
            if active[index] {
                factor_item_counts[factor] += 1;
            }
        }
        if !active[item * n_factors..(item + 1) * n_factors]
            .iter()
            .any(|is_active| *is_active)
        {
            return Err(format!(
                "every item must have at least one loading above zero_tolerance; item {item} has none"
            ));
        }
    }
    if let Some(factor) = factor_item_counts.iter().position(|count| *count == 0) {
        return Err(format!(
            "every factor must have at least one loading above zero_tolerance; factor {factor} has none"
        ));
    }

    let total_common_variance: f64 = item_common_variance.iter().sum();
    let mut ecv_ss = Vec::with_capacity(n_factors);
    let mut ecv_sg = Vec::with_capacity(n_factors);
    let mut ecv_gs = Vec::with_capacity(n_factors);
    let mut omega_total = Vec::with_capacity(n_factors);
    let mut omega_hierarchical = Vec::with_capacity(n_factors);
    let mut construct_replicability = Vec::with_capacity(n_factors);

    for factor in 0..n_factors {
        let mut target_variance = 0.0_f64;
        let mut general_variance = 0.0_f64;
        let mut domain_common_variance = 0.0_f64;
        let mut uniqueness_sum = 0.0_f64;
        let mut column_sums = vec![0.0_f64; n_factors];
        let mut h_information = 0.0_f64;

        for item in 0..n_items {
            let factor_index = item * n_factors + factor;
            let target_squared = squared[factor_index];
            h_information += target_squared / (1.0 - target_squared);

            if !active[factor_index] {
                continue;
            }
            target_variance += target_squared;
            general_variance += squared[item * n_factors + general_factor];
            domain_common_variance += item_common_variance[item];
            uniqueness_sum += uniquenesses[item];
            for column in 0..n_factors {
                column_sums[column] += loadings[item * n_factors + column];
            }
        }

        ecv_ss.push(target_variance / domain_common_variance);
        ecv_sg.push(target_variance / total_common_variance);
        ecv_gs.push(general_variance / domain_common_variance);

        let common_sum_variance: f64 = column_sums.iter().map(|sum| sum * sum).sum();
        let omega_denominator = common_sum_variance + uniqueness_sum;
        if omega_denominator <= 0.0 {
            return Err(format!(
                "omega denominator must be positive for factor {factor}"
            ));
        }
        omega_total.push(common_sum_variance / omega_denominator);
        omega_hierarchical.push(column_sums[factor].powi(2) / omega_denominator);
        construct_replicability.push(1.0 / (1.0 + 1.0 / h_information));
    }

    let item_ecv = (0..n_items)
        .map(|item| {
            squared[item * n_factors + general_factor] / item_common_variance[item]
        })
        .collect();

    let is_strict_bifactor = (0..n_items).all(|item| {
        let general_active = active[item * n_factors + general_factor];
        let specific_count = (0..n_factors)
            .filter(|factor| *factor != general_factor)
            .filter(|factor| active[item * n_factors + *factor])
            .count();
        general_active && specific_count <= 1
    });
    let puc = is_strict_bifactor.then(|| {
        strict_bifactor_puc(&factor_item_counts, n_items, general_factor)
    });

    Ok(BifactorIndicesResult {
        factor_item_counts,
        is_strict_bifactor,
        puc,
        ecv_ss,
        ecv_sg,
        ecv_gs,
        item_ecv,
        omega_total,
        omega_hierarchical,
        construct_replicability,
    })
}

fn validate_inputs(
    loadings: &[f64],
    uniquenesses: &[f64],
    config: BifactorIndicesConfig,
) -> Result<(), String> {
    if config.n_items < 2 {
        return Err("bifactor indices require at least two items".into());
    }
    if config.n_factors < 2 {
        return Err("bifactor indices require at least two factors".into());
    }
    if config.general_factor >= config.n_factors {
        return Err(format!(
            "general_factor must be in 0..{}",
            config.n_factors
        ));
    }
    if !(config.zero_tolerance.is_finite() && config.zero_tolerance >= 0.0) {
        return Err("zero_tolerance must be finite and non-negative".into());
    }
    let expected = checked_mul_usize(
        config.n_items,
        config.n_factors,
        "loading matrix dimensions overflow usize",
    )?;
    if loadings.len() != expected {
        return Err(format!(
            "loading matrix length must be n_items * n_factors ({expected}); got {}",
            loadings.len()
        ));
    }
    if uniquenesses.len() != config.n_items {
        return Err(format!(
            "uniquenesses length must equal n_items ({}); got {}",
            config.n_items,
            uniquenesses.len()
        ));
    }
    if loadings.iter().any(|loading| !loading.is_finite()) {
        return Err("standardized loadings must be finite".into());
    }
    if loadings.iter().any(|loading| loading.abs() >= 1.0) {
        return Err("standardized loadings must have absolute value below 1".into());
    }
    if uniquenesses.iter().any(|value| !value.is_finite()) {
        return Err("uniquenesses must be finite".into());
    }
    if uniquenesses.iter().any(|value| *value < 0.0) {
        return Err("uniquenesses must be non-negative".into());
    }
    Ok(())
}

fn strict_bifactor_puc(
    factor_item_counts: &[usize],
    n_items: usize,
    general_factor: usize,
) -> f64 {
    let total_pairs = pair_count(n_items);
    let contaminated_pairs: f64 = factor_item_counts
        .iter()
        .enumerate()
        .filter(|(factor, _count)| *factor != general_factor)
        .map(|(_factor, &count)| pair_count(count))
        .sum();
    1.0 - contaminated_pairs / total_pairs
}

fn pair_count(value: usize) -> f64 {
    if value < 2 {
        0.0
    } else {
        value as f64 * (value - 1) as f64 / 2.0
    }
}

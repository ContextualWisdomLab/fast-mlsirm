//! Finite-population proportion sampling designs.
//!
//! The normal-approximation sample-size equation and finite-population
//! correction follow the NIST/SEMATECH e-Handbook. Stratified allocation
//! follows the Australian Bureau of Statistics proportional and optimum
//! (Neyman, equal-cost) allocation descriptions. Callers must supply the
//! expected proportion for every stratum; this module never invents one.

use std::cmp::Ordering;

use crate::nodes::inv_normal_cdf;
use sha2::{Digest, Sha256};

/// Wire/schema identity for the Rust and PyO3 sampling-design result.
pub const SAMPLING_DESIGN_SCHEMA_VERSION: &str = "fast-mlsirm.sampling-design.v1";
/// Stable identity of the Rust implementation that owns the artifact.
pub const SAMPLING_DESIGN_SOURCE_IDENTITY: &str = "fast-mlsirm.mlsirm-core.sampling-design";
/// Version of the sample-size, FPC, allocation, and integerization algorithm.
pub const SAMPLING_DESIGN_ALGORITHM_VERSION: &str = "1.1.0";
/// Wire/schema identity for an achieved one-stratum proportion result.
pub const ACHIEVED_PROPORTION_SCHEMA_VERSION: &str = "fast-mlsirm.achieved-proportion.v1";
/// Version of the SRSWOR estimator, variance, and Wang/Konijn interval.
pub const ACHIEVED_PROPORTION_ALGORITHM_VERSION: &str = "1.0.0";
const MAX_EXACT_F64_INTEGER: usize = 1_usize << 53;
const MAX_STRATA: usize = 100_000;

/// Supported evidence-grounded allocation rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationMethod {
    /// Allocate in proportion to stratum population size.
    Proportional,
    /// Equal-cost Neyman allocation using `N_h sqrt(p_h (1-p_h))`.
    Neyman,
}

impl AllocationMethod {
    /// Parse the exact public allocation identifier.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "proportional" => Some(Self::Proportional),
            "neyman" => Some(Self::Neyman),
            _ => None,
        }
    }

    /// Return the exact public allocation identifier.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Proportional => "proportional",
            Self::Neyman => "neyman",
        }
    }
}

/// One disjoint finite-population stratum and its prior/pilot proportion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingStratum {
    /// Count of population units in the stratum.
    pub population_size: usize,
    /// Prior- or pilot-derived expected proportion, strictly between zero and one.
    pub expected_proportion: f64,
}

/// Versioned sampling-design result returned by Rust and PyO3.
#[derive(Clone, Debug, PartialEq)]
pub struct ProportionSamplingDesign {
    /// Exact wire/schema version.
    pub schema_version: &'static str,
    /// Stable package-owned source identity.
    pub source_identity: &'static str,
    /// SHA-256 of this exact Rust source file at build time.
    pub source_sha256: String,
    /// Version of the complete Rust-owned algorithm.
    pub algorithm_version: &'static str,
    /// Finite population size.
    pub population_size: usize,
    /// Population-weighted expected proportion derived from the strata.
    pub expected_proportion: f64,
    /// Caller-declared two-sided confidence level.
    pub confidence_level: f64,
    /// Standard-normal critical value for the two-sided confidence level.
    pub critical_value: f64,
    /// Caller-declared absolute margin of error.
    pub margin_of_error: f64,
    /// Infinite-population normal-approximation sample size before rounding.
    pub uncorrected_sample_size: f64,
    /// Finite-population corrected sample size, rounded upward.
    pub sample_size: usize,
    /// Standard-error multiplier `sqrt((N - n) / (N - 1))`.
    pub finite_population_correction: f64,
    /// Allocation rule used for the ordered strata.
    pub allocation_method: AllocationMethod,
    /// Canonical ordered inputs retained for independent replay.
    pub strata: Vec<SamplingStratum>,
    /// Integer sample count for each input stratum, in input order.
    pub stratum_sample_sizes: Vec<usize>,
    /// Exact first-order inclusion probability `n_h / N_h` for each stratum.
    pub stratum_inclusion_probability_ratios: Vec<(usize, usize)>,
    /// SHA-256 of the canonical input encoding.
    pub input_sha256: String,
    /// SHA-256 of the canonical computed-output encoding.
    pub output_sha256: String,
    /// SHA-256 binding schema, source, algorithm, input, and output identities.
    pub artifact_sha256: String,
}

/// Terminal estimate and exact interval for one completed SRSWOR sample.
#[derive(Clone, Debug, PartialEq)]
pub struct AchievedProportion {
    /// Exact wire/schema version.
    pub schema_version: &'static str,
    /// Stable package-owned source identity.
    pub source_identity: &'static str,
    /// SHA-256 of this exact Rust source file at build time.
    pub source_sha256: String,
    /// Version of the complete Rust-owned algorithm.
    pub algorithm_version: &'static str,
    /// Sampling-design artifact this result terminates.
    pub design_artifact_sha256: String,
    /// Finite population size.
    pub population_size: usize,
    /// Completed sample size.
    pub sample_size: usize,
    /// Count of sampled units possessing the declared attribute.
    pub success_count: usize,
    /// Sample proportion estimator.
    pub estimated_proportion: f64,
    /// Unbiased SRSWOR design-variance estimate for the sample proportion.
    pub design_variance: f64,
    /// Caller-declared two-sided confidence level.
    pub confidence_level: f64,
    /// Exact interval method identifier.
    pub interval_method: &'static str,
    /// Inclusive lower bound for the finite-population success count.
    pub lower_success_count: usize,
    /// Inclusive upper bound for the finite-population success count.
    pub upper_success_count: usize,
    /// Lower proportion bound on the finite-population grid.
    pub lower_proportion: f64,
    /// Upper proportion bound on the finite-population grid.
    pub upper_proportion: f64,
    /// SHA-256 of the canonical input encoding.
    pub input_sha256: String,
    /// SHA-256 of the canonical computed-output encoding.
    pub output_sha256: String,
    /// SHA-256 binding schema, source, algorithm, design, input, and output.
    pub artifact_sha256: String,
}

fn put_text(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn input_identity(
    population_size: usize,
    confidence_level: f64,
    margin_of_error: f64,
    strata: &[SamplingStratum],
    allocation_method: AllocationMethod,
) -> String {
    let mut bytes = Vec::new();
    put_text(&mut bytes, SAMPLING_DESIGN_SCHEMA_VERSION);
    put_text(&mut bytes, SAMPLING_DESIGN_SOURCE_IDENTITY);
    put_text(&mut bytes, SAMPLING_DESIGN_ALGORITHM_VERSION);
    bytes.extend_from_slice(&(population_size as u64).to_be_bytes());
    bytes.extend_from_slice(&confidence_level.to_bits().to_be_bytes());
    bytes.extend_from_slice(&margin_of_error.to_bits().to_be_bytes());
    put_text(&mut bytes, allocation_method.as_str());
    bytes.extend_from_slice(&(strata.len() as u64).to_be_bytes());
    for stratum in strata {
        bytes.extend_from_slice(&(stratum.population_size as u64).to_be_bytes());
        bytes.extend_from_slice(&stratum.expected_proportion.to_bits().to_be_bytes());
    }
    sha256_hex(&bytes)
}

fn output_identity(
    expected_proportion: f64,
    critical_value: f64,
    uncorrected_sample_size: f64,
    sample_size: usize,
    finite_population_correction: f64,
    stratum_sample_sizes: &[usize],
    stratum_inclusion_probability_ratios: &[(usize, usize)],
) -> String {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&expected_proportion.to_bits().to_be_bytes());
    bytes.extend_from_slice(&critical_value.to_bits().to_be_bytes());
    bytes.extend_from_slice(&uncorrected_sample_size.to_bits().to_be_bytes());
    bytes.extend_from_slice(&(sample_size as u64).to_be_bytes());
    bytes.extend_from_slice(&finite_population_correction.to_bits().to_be_bytes());
    bytes.extend_from_slice(&(stratum_sample_sizes.len() as u64).to_be_bytes());
    for count in stratum_sample_sizes {
        bytes.extend_from_slice(&(*count as u64).to_be_bytes());
    }
    bytes.extend_from_slice(&(stratum_inclusion_probability_ratios.len() as u64).to_be_bytes());
    for (numerator, denominator) in stratum_inclusion_probability_ratios {
        bytes.extend_from_slice(&(*numerator as u64).to_be_bytes());
        bytes.extend_from_slice(&(*denominator as u64).to_be_bytes());
    }
    sha256_hex(&bytes)
}

fn artifact_identity(source_sha256: &str, input_sha256: &str, output_sha256: &str) -> String {
    let mut bytes = Vec::new();
    put_text(&mut bytes, SAMPLING_DESIGN_SCHEMA_VERSION);
    put_text(&mut bytes, SAMPLING_DESIGN_SOURCE_IDENTITY);
    put_text(&mut bytes, source_sha256);
    put_text(&mut bytes, SAMPLING_DESIGN_ALGORITHM_VERSION);
    put_text(&mut bytes, input_sha256);
    put_text(&mut bytes, output_sha256);
    sha256_hex(&bytes)
}

fn ln_choose(total: usize, selected: usize) -> f64 {
    if selected > total {
        return f64::NEG_INFINITY;
    }
    let selected = selected.min(total - selected);
    (1..=selected).fold(0.0, |sum, index| {
        sum + ((total - selected + index) as f64).ln() - (index as f64).ln()
    })
}

fn log_add_exp(left: f64, right: f64) -> f64 {
    if left == f64::NEG_INFINITY {
        return right;
    }
    let maximum = left.max(right);
    maximum + ((left - maximum).exp() + (right - maximum).exp()).ln()
}

fn hypergeometric_log_cdf(
    population_size: usize,
    population_successes: usize,
    sample_size: usize,
    cutoff: usize,
) -> f64 {
    let support_lower = sample_size.saturating_sub(population_size - population_successes);
    let support_upper = sample_size.min(population_successes);
    if cutoff < support_lower {
        return f64::NEG_INFINITY;
    }
    if cutoff >= support_upper {
        return 0.0;
    }
    let mut count = support_lower;
    let mut log_probability = ln_choose(population_successes, count)
        + ln_choose(population_size - population_successes, sample_size - count)
        - ln_choose(population_size, sample_size);
    let mut log_sum = log_probability;
    while count < cutoff {
        let numerator_left = population_successes - count;
        let numerator_right = sample_size - count;
        let denominator_left = count + 1;
        let denominator_right =
            (population_size - population_successes) - (sample_size - count) + 1;
        log_probability += (numerator_left as f64).ln() + (numerator_right as f64).ln()
            - (denominator_left as f64).ln()
            - (denominator_right as f64).ln();
        log_sum = log_add_exp(log_sum, log_probability);
        count += 1;
    }
    log_sum.min(0.0)
}

fn wang_konijn_lower_bound(
    population_size: usize,
    sample_size: usize,
    success_count: usize,
    lower_tail_probability: f64,
) -> usize {
    if success_count == 0 {
        return 0;
    }
    let log_threshold = (1.0 - lower_tail_probability).ln();
    let mut lower = success_count;
    let mut upper = population_size - sample_size + success_count;
    while lower < upper {
        let candidate = lower + (upper - lower).div_ceil(2);
        let log_cdf = hypergeometric_log_cdf(
            population_size,
            candidate - 1,
            sample_size,
            success_count - 1,
        );
        if log_cdf >= log_threshold {
            lower = candidate;
        } else {
            upper = candidate - 1;
        }
    }
    lower
}

/// Estimate a finite-population proportion after one complete SRSWOR sample.
pub fn finite_population_achieved_proportion(
    design_artifact_sha256: &str,
    population_size: usize,
    sample_size: usize,
    success_count: usize,
    confidence_level: f64,
) -> Result<AchievedProportion, String> {
    if design_artifact_sha256.len() != 64
        || !design_artifact_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("design_artifact_sha256 must be 64 lowercase hexadecimal characters".into());
    }
    if population_size == 0 || population_size > MAX_EXACT_F64_INTEGER {
        return Err("population_size must be between 1 and 2^53".into());
    }
    if sample_size == 0 || sample_size > population_size {
        return Err("sample_size must be between 1 and population_size".into());
    }
    if success_count > sample_size {
        return Err("success_count must be between zero and sample_size".into());
    }
    if !(0.0 < confidence_level && confidence_level < 1.0) {
        return Err("confidence_level must be strictly between zero and one".into());
    }
    if sample_size == 1 && population_size > 1 {
        return Err("a non-census design variance requires at least two sampled units".into());
    }

    let estimated_proportion = success_count as f64 / sample_size as f64;
    let design_variance = if sample_size == population_size {
        0.0
    } else {
        (population_size - sample_size) as f64 / population_size as f64
            * estimated_proportion
            * (1.0 - estimated_proportion)
            / (sample_size - 1) as f64
    };
    let tail_probability = (1.0 - confidence_level) / 2.0;
    let lower_success_count = wang_konijn_lower_bound(
        population_size,
        sample_size,
        success_count,
        tail_probability,
    );
    let upper_success_count = population_size
        - wang_konijn_lower_bound(
            population_size,
            sample_size,
            sample_size - success_count,
            tail_probability,
        );
    let lower_proportion = lower_success_count as f64 / population_size as f64;
    let upper_proportion = upper_success_count as f64 / population_size as f64;
    let source_sha256 = sha256_hex(include_bytes!("sampling_design.rs"));

    let mut input_bytes = Vec::new();
    put_text(&mut input_bytes, ACHIEVED_PROPORTION_SCHEMA_VERSION);
    put_text(&mut input_bytes, SAMPLING_DESIGN_SOURCE_IDENTITY);
    put_text(&mut input_bytes, ACHIEVED_PROPORTION_ALGORITHM_VERSION);
    put_text(&mut input_bytes, design_artifact_sha256);
    input_bytes.extend_from_slice(&(population_size as u64).to_be_bytes());
    input_bytes.extend_from_slice(&(sample_size as u64).to_be_bytes());
    input_bytes.extend_from_slice(&(success_count as u64).to_be_bytes());
    input_bytes.extend_from_slice(&confidence_level.to_bits().to_be_bytes());
    let input_sha256 = sha256_hex(&input_bytes);

    let mut output_bytes = Vec::new();
    output_bytes.extend_from_slice(&estimated_proportion.to_bits().to_be_bytes());
    output_bytes.extend_from_slice(&design_variance.to_bits().to_be_bytes());
    put_text(&mut output_bytes, "wang_konijn_equal_tailed");
    output_bytes.extend_from_slice(&(lower_success_count as u64).to_be_bytes());
    output_bytes.extend_from_slice(&(upper_success_count as u64).to_be_bytes());
    output_bytes.extend_from_slice(&lower_proportion.to_bits().to_be_bytes());
    output_bytes.extend_from_slice(&upper_proportion.to_bits().to_be_bytes());
    let output_sha256 = sha256_hex(&output_bytes);

    let mut artifact_bytes = Vec::new();
    put_text(&mut artifact_bytes, ACHIEVED_PROPORTION_SCHEMA_VERSION);
    put_text(&mut artifact_bytes, SAMPLING_DESIGN_SOURCE_IDENTITY);
    put_text(&mut artifact_bytes, &source_sha256);
    put_text(&mut artifact_bytes, ACHIEVED_PROPORTION_ALGORITHM_VERSION);
    put_text(&mut artifact_bytes, design_artifact_sha256);
    put_text(&mut artifact_bytes, &input_sha256);
    put_text(&mut artifact_bytes, &output_sha256);
    let artifact_sha256 = sha256_hex(&artifact_bytes);

    Ok(AchievedProportion {
        schema_version: ACHIEVED_PROPORTION_SCHEMA_VERSION,
        source_identity: SAMPLING_DESIGN_SOURCE_IDENTITY,
        source_sha256,
        algorithm_version: ACHIEVED_PROPORTION_ALGORITHM_VERSION,
        design_artifact_sha256: design_artifact_sha256.to_owned(),
        population_size,
        sample_size,
        success_count,
        estimated_proportion,
        design_variance,
        confidence_level,
        interval_method: "wang_konijn_equal_tailed",
        lower_success_count,
        upper_success_count,
        lower_proportion,
        upper_proportion,
        input_sha256,
        output_sha256,
        artifact_sha256,
    })
}

/// Design a two-sided proportion estimate for sampling without replacement.
pub fn finite_population_proportion_design(
    population_size: usize,
    confidence_level: f64,
    margin_of_error: f64,
    strata: &[SamplingStratum],
    allocation_method: AllocationMethod,
) -> Result<ProportionSamplingDesign, String> {
    if population_size == 0 || population_size > MAX_EXACT_F64_INTEGER {
        return Err("population_size must be between 1 and 2^53".into());
    }
    if !(0.0 < confidence_level && confidence_level < 1.0) {
        return Err("confidence_level must be strictly between zero and one".into());
    }
    if !(0.0 < margin_of_error && margin_of_error < 1.0) {
        return Err("margin_of_error must be strictly between zero and one".into());
    }
    if strata.is_empty() || strata.len() > MAX_STRATA {
        return Err(format!(
            "strata must contain between 1 and {MAX_STRATA} entries"
        ));
    }

    let mut stratum_population_sum = 0_usize;
    let mut expected_numerator = 0.0_f64;
    for stratum in strata {
        if stratum.population_size == 0 {
            return Err("every stratum population_size must be positive".into());
        }
        if !(0.0 < stratum.expected_proportion && stratum.expected_proportion < 1.0) {
            return Err(
                "every expected_proportion must be finite and strictly between zero and one".into(),
            );
        }
        stratum_population_sum = stratum_population_sum
            .checked_add(stratum.population_size)
            .ok_or_else(|| "stratum population sum overflows".to_string())?;
        expected_numerator += stratum.population_size as f64 * stratum.expected_proportion;
    }
    if stratum_population_sum != population_size {
        return Err("stratum population sizes must sum to population_size".into());
    }

    let expected_proportion = expected_numerator / population_size as f64;
    let critical_value = inv_normal_cdf(0.5 + confidence_level / 2.0);
    let uncorrected_sample_size =
        critical_value * critical_value * expected_proportion * (1.0 - expected_proportion)
            / (margin_of_error * margin_of_error);
    if !uncorrected_sample_size.is_finite() || uncorrected_sample_size <= 0.0 {
        return Err("sampling design did not produce a finite positive sample size".into());
    }
    let finite_sample_size = population_size as f64 * uncorrected_sample_size
        / (population_size as f64 + uncorrected_sample_size - 1.0);
    let sample_size = (finite_sample_size.ceil() as usize).min(population_size);
    let finite_population_correction = if sample_size == population_size {
        0.0
    } else {
        ((population_size - sample_size) as f64 / (population_size - 1) as f64).sqrt()
    };
    let stratum_sample_sizes = allocate_strata(sample_size, strata, allocation_method)?;
    let stratum_inclusion_probability_ratios = stratum_sample_sizes
        .iter()
        .zip(strata)
        .map(|(sample_count, stratum)| (*sample_count, stratum.population_size))
        .collect::<Vec<_>>();
    let source_sha256 = sha256_hex(include_bytes!("sampling_design.rs"));
    let input_sha256 = input_identity(
        population_size,
        confidence_level,
        margin_of_error,
        strata,
        allocation_method,
    );
    let output_sha256 = output_identity(
        expected_proportion,
        critical_value,
        uncorrected_sample_size,
        sample_size,
        finite_population_correction,
        &stratum_sample_sizes,
        &stratum_inclusion_probability_ratios,
    );
    let artifact_sha256 = artifact_identity(&source_sha256, &input_sha256, &output_sha256);

    Ok(ProportionSamplingDesign {
        schema_version: SAMPLING_DESIGN_SCHEMA_VERSION,
        source_identity: SAMPLING_DESIGN_SOURCE_IDENTITY,
        source_sha256,
        algorithm_version: SAMPLING_DESIGN_ALGORITHM_VERSION,
        population_size,
        expected_proportion,
        confidence_level,
        critical_value,
        margin_of_error,
        uncorrected_sample_size,
        sample_size,
        finite_population_correction,
        allocation_method,
        strata: strata.to_vec(),
        stratum_sample_sizes,
        stratum_inclusion_probability_ratios,
        input_sha256,
        output_sha256,
        artifact_sha256,
    })
}

fn allocate_strata(
    sample_size: usize,
    strata: &[SamplingStratum],
    method: AllocationMethod,
) -> Result<Vec<usize>, String> {
    allocate_strata_with_cap_probe(sample_size, strata, method, || {})
}

fn allocate_strata_with_cap_probe<F>(
    sample_size: usize,
    strata: &[SamplingStratum],
    method: AllocationMethod,
    mut inspect_cap_candidate: F,
) -> Result<Vec<usize>, String>
where
    F: FnMut(),
{
    let weights: Vec<f64> = strata
        .iter()
        .map(|stratum| match method {
            AllocationMethod::Proportional => stratum.population_size as f64,
            AllocationMethod::Neyman => {
                stratum.population_size as f64
                    * (stratum.expected_proportion * (1.0 - stratum.expected_proportion)).sqrt()
            }
        })
        .collect();
    let mut allocated = vec![0_usize; strata.len()];
    let mut remaining = sample_size;
    let mut weight_sum: f64 = weights.iter().sum();
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err("stratum allocation has no positive finite weight".into());
    }

    // The capped proportional-allocation solution is a water-filling problem.
    // For an active stratum i, capping occurs when
    // remaining / weight_sum >= population_i / weight_i. Sorting those fixed
    // thresholds once lets the cap phase inspect every stratum at most once;
    // the prior implementation rescanned the whole active set after each cap.
    let mut cap_order: Vec<usize> = (0..strata.len()).collect();
    cap_order.sort_by(|left, right| {
        let left_threshold = strata[*left].population_size as f64 / weights[*left];
        let right_threshold = strata[*right].population_size as f64 / weights[*right];
        left_threshold
            .partial_cmp(&right_threshold)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(right))
    });

    for index in cap_order {
        if remaining == 0 {
            break;
        }
        inspect_cap_candidate();
        let quota = remaining as f64 * weights[index] / weight_sum;
        if quota < strata[index].population_size as f64 {
            break;
        }
        allocated[index] = strata[index].population_size;
        remaining -= allocated[index];
        weight_sum -= weights[index];
        if remaining > 0 && (!weight_sum.is_finite() || weight_sum <= 0.0) {
            return Err("stratum allocation has no positive finite weight".into());
        }
    }

    if remaining > 0 {
        let mut fractions = Vec::new();
        for index in 0..strata.len() {
            if allocated[index] != 0 {
                continue;
            }
            let quota = remaining as f64 * weights[index] / weight_sum;
            let base = quota.floor() as usize;
            allocated[index] = base;
            fractions.push((index, quota - base as f64));
        }
        let assigned: usize = allocated.iter().sum();
        fractions.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });
        for (index, _) in fractions.into_iter().take(sample_size - assigned) {
            allocated[index] += 1;
        }
    }

    if allocated.contains(&0) {
        return Err(
            "the requested precision cannot allocate at least one unit to every stratum".into(),
        );
    }
    if allocated.iter().sum::<usize>() != sample_size {
        return Err("stratum allocation does not sum to the required sample size".into());
    }
    Ok(allocated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_the_finite_population_correction_and_rounds_up() {
        let design = finite_population_proportion_design(
            1_000,
            0.95,
            0.05,
            &[SamplingStratum {
                population_size: 1_000,
                expected_proportion: 0.5,
            }],
            AllocationMethod::Proportional,
        )
        .unwrap();
        assert_eq!(design.schema_version, SAMPLING_DESIGN_SCHEMA_VERSION);
        assert_eq!(design.sample_size, 278);
        assert_eq!(design.stratum_sample_sizes, vec![278]);
        assert_eq!(
            design.stratum_inclusion_probability_ratios,
            vec![(278, 1_000)]
        );
        assert_eq!(design.strata[0].population_size, 1_000);
        assert_eq!(design.source_identity, SAMPLING_DESIGN_SOURCE_IDENTITY);
        assert_eq!(design.source_sha256.len(), 64);
        assert_eq!(design.algorithm_version, SAMPLING_DESIGN_ALGORITHM_VERSION);
        assert_eq!(design.input_sha256.len(), 64);
        assert_eq!(design.output_sha256.len(), 64);
        assert_eq!(design.artifact_sha256.len(), 64);
        assert!((design.uncorrected_sample_size - 384.145_882).abs() < 1e-5);
        assert!((design.finite_population_correction - (722.0_f64 / 999.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn content_identity_is_deterministic_and_binds_inputs_and_outputs() {
        let build = |proportion| {
            finite_population_proportion_design(
                100,
                0.95,
                0.1,
                &[SamplingStratum {
                    population_size: 100,
                    expected_proportion: proportion,
                }],
                AllocationMethod::Proportional,
            )
            .unwrap()
        };
        let first = build(0.5);
        let replay = build(0.5);
        let changed = build(0.4);
        assert_eq!(first.input_sha256, replay.input_sha256);
        assert_eq!(first.output_sha256, replay.output_sha256);
        assert_eq!(first.artifact_sha256, replay.artifact_sha256);
        assert_eq!(
            first.input_sha256,
            "8d4b0c374e9c6c39fa872e8bf1e195c4d39c91d1a9ce8991836a5a66ee3bd269"
        );
        assert_eq!(
            first.output_sha256,
            "bb9bc3ca52b6f2642226b80b5ea2f41b83eac225ee8dc448c8f9e29de13c68b9"
        );
        assert_ne!(first.input_sha256, changed.input_sha256);
        assert_ne!(first.artifact_sha256, changed.artifact_sha256);
    }

    #[test]
    fn allocates_strata_by_declared_proportional_or_neyman_evidence() {
        let strata = [
            SamplingStratum {
                population_size: 60,
                expected_proportion: 0.1,
            },
            SamplingStratum {
                population_size: 40,
                expected_proportion: 0.5,
            },
        ];
        let proportional = finite_population_proportion_design(
            100,
            0.95,
            0.1,
            &strata,
            AllocationMethod::Proportional,
        )
        .unwrap();
        let neyman =
            finite_population_proportion_design(100, 0.95, 0.1, &strata, AllocationMethod::Neyman)
                .unwrap();
        assert_eq!(proportional.stratum_sample_sizes, vec![26, 17]);
        assert_eq!(neyman.stratum_sample_sizes, vec![20, 23]);
        assert_eq!(
            proportional.stratum_inclusion_probability_ratios,
            vec![(26, 60), (17, 40)]
        );
        assert_eq!(
            neyman.stratum_inclusion_probability_ratios,
            vec![(20, 60), (23, 40)]
        );
        assert_eq!(proportional.sample_size, 43);
        assert_eq!(neyman.sample_size, 43);
    }

    #[test]
    fn census_cap_phase_inspects_each_stratum_at_most_once() {
        let strata = vec![
            SamplingStratum {
                population_size: 1,
                expected_proportion: 0.5,
            };
            MAX_STRATA
        ];
        let mut cap_checks = 0_usize;
        let allocated = allocate_strata_with_cap_probe(
            MAX_STRATA,
            &strata,
            AllocationMethod::Proportional,
            || cap_checks += 1,
        )
        .unwrap();
        assert_eq!(cap_checks, MAX_STRATA);
        assert!(allocated.iter().all(|count| *count == 1));
    }

    #[test]
    fn refuses_invented_or_infeasible_stratum_evidence() {
        let invalid = SamplingStratum {
            population_size: 10,
            expected_proportion: 0.0,
        };
        assert!(finite_population_proportion_design(
            10,
            0.95,
            0.1,
            &[invalid],
            AllocationMethod::Neyman,
        )
        .unwrap_err()
        .contains("expected_proportion"));

        let tiny = [
            SamplingStratum {
                population_size: 1,
                expected_proportion: 0.5,
            },
            SamplingStratum {
                population_size: 1,
                expected_proportion: 0.5,
            },
            SamplingStratum {
                population_size: 98,
                expected_proportion: 0.5,
            },
        ];
        assert!(finite_population_proportion_design(
            100,
            0.80,
            0.49,
            &tiny,
            AllocationMethod::Proportional,
        )
        .unwrap_err()
        .contains("at least one"));
    }

    #[test]
    fn refuses_mismatched_population_and_unknown_method() {
        assert_eq!(
            AllocationMethod::parse("proportional"),
            Some(AllocationMethod::Proportional)
        );
        assert_eq!(
            AllocationMethod::parse("neyman"),
            Some(AllocationMethod::Neyman)
        );
        assert_eq!(AllocationMethod::parse("equal"), None);
        let error = finite_population_proportion_design(
            11,
            0.95,
            0.1,
            &[SamplingStratum {
                population_size: 10,
                expected_proportion: 0.5,
            }],
            AllocationMethod::Proportional,
        )
        .unwrap_err();
        assert!(error.contains("sum"));
    }

    #[test]
    fn matches_wang_konijn_published_equal_tailed_belt() {
        let expected_lower = [0, 1, 3, 8, 13, 19, 26, 33, 40, 48, 57];
        let expected_upper = [32, 47, 61, 73, 85, 95, 106, 116, 125, 134, 143];
        for success_count in 0..=10 {
            let result = finite_population_achieved_proportion(
                &"a".repeat(64),
                200,
                20,
                success_count,
                0.95,
            )
            .unwrap();
            assert_eq!(result.lower_success_count, expected_lower[success_count]);
            assert_eq!(result.upper_success_count, expected_upper[success_count]);
        }
    }

    #[test]
    fn exact_interval_meets_exhaustive_finite_population_coverage() {
        for population_size in 2..=20 {
            for sample_size in 2..=population_size {
                let intervals = (0..=sample_size)
                    .map(|success_count| {
                        finite_population_achieved_proportion(
                            &"b".repeat(64),
                            population_size,
                            sample_size,
                            success_count,
                            0.95,
                        )
                        .unwrap()
                    })
                    .collect::<Vec<_>>();
                for pair in intervals.windows(2) {
                    assert!(pair[0].lower_success_count <= pair[1].lower_success_count);
                    assert!(pair[0].upper_success_count <= pair[1].upper_success_count);
                }
                for success_count in 0..=sample_size {
                    let complement = &intervals[sample_size - success_count];
                    assert_eq!(
                        intervals[success_count].lower_success_count,
                        population_size - complement.upper_success_count
                    );
                    assert_eq!(
                        intervals[success_count].upper_success_count,
                        population_size - complement.lower_success_count
                    );
                }
                for population_successes in 0..=population_size {
                    let support_lower =
                        sample_size.saturating_sub(population_size - population_successes);
                    let support_upper = sample_size.min(population_successes);
                    let coverage = (support_lower..=support_upper)
                        .filter(|success_count| {
                            let interval = &intervals[*success_count];
                            interval.lower_success_count <= population_successes
                                && population_successes <= interval.upper_success_count
                        })
                        .map(|success_count| {
                            (ln_choose(population_successes, success_count)
                                + ln_choose(
                                    population_size - population_successes,
                                    sample_size - success_count,
                                )
                                - ln_choose(population_size, sample_size))
                            .exp()
                        })
                        .sum::<f64>();
                    assert!(coverage >= 0.95 - 1e-12, "coverage={coverage}");
                }
            }
        }
    }

    #[test]
    fn achieved_artifact_binds_design_and_preserves_extreme_uncertainty() {
        let result =
            finite_population_achieved_proportion(&"c".repeat(64), 43_814, 100, 100, 0.95).unwrap();
        assert_eq!(result.schema_version, ACHIEVED_PROPORTION_SCHEMA_VERSION);
        assert_eq!(
            result.algorithm_version,
            ACHIEVED_PROPORTION_ALGORITHM_VERSION
        );
        assert_eq!(result.estimated_proportion, 1.0);
        assert_eq!(result.design_variance, 0.0);
        assert!(result.lower_success_count < result.population_size);
        assert_eq!(result.upper_success_count, result.population_size);
        assert!(result.lower_proportion < 1.0);
        assert_eq!(result.upper_proportion, 1.0);
        assert_eq!(result.source_sha256.len(), 64);
        assert_eq!(result.input_sha256.len(), 64);
        assert_eq!(result.output_sha256.len(), 64);
        assert_eq!(result.artifact_sha256.len(), 64);

        let changed =
            finite_population_achieved_proportion(&"d".repeat(64), 43_814, 100, 100, 0.95).unwrap();
        assert_ne!(result.input_sha256, changed.input_sha256);
        assert_ne!(result.artifact_sha256, changed.artifact_sha256);
    }

    #[test]
    fn achieved_artifact_fails_closed_on_incomplete_or_invalid_inputs() {
        let valid_hash = "e".repeat(64);
        let cases = [
            finite_population_achieved_proportion("bad", 10, 2, 1, 0.95),
            finite_population_achieved_proportion(&valid_hash, 0, 2, 1, 0.95),
            finite_population_achieved_proportion(&valid_hash, 10, 0, 0, 0.95),
            finite_population_achieved_proportion(&valid_hash, 10, 11, 1, 0.95),
            finite_population_achieved_proportion(&valid_hash, 10, 2, 3, 0.95),
            finite_population_achieved_proportion(&valid_hash, 10, 2, 1, 1.0),
            finite_population_achieved_proportion(&valid_hash, 10, 1, 1, 0.95),
        ];
        assert!(cases.into_iter().all(|result| result.is_err()));

        let census = finite_population_achieved_proportion(&valid_hash, 1, 1, 1, 0.95).unwrap();
        assert_eq!(census.lower_success_count, 1);
        assert_eq!(census.upper_success_count, 1);
        assert_eq!(census.design_variance, 0.0);
    }
}

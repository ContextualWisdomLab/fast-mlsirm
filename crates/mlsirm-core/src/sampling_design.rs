//! Finite-population proportion sampling designs.
//!
//! The normal-approximation sample-size equation and finite-population
//! correction follow the NIST/SEMATECH e-Handbook. Stratified allocation
//! follows the Australian Bureau of Statistics proportional and optimum
//! (Neyman, equal-cost) allocation descriptions. Callers must supply the
//! expected proportion for every stratum; this module never invents one.

use std::cmp::Ordering;

use crate::nodes::inv_normal_cdf;

/// Wire/schema identity for the Rust and PyO3 sampling-design result.
pub const SAMPLING_DESIGN_SCHEMA_VERSION: &str = "fast-mlsirm.sampling-design.v1";
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
    /// Integer sample count for each input stratum, in input order.
    pub stratum_sample_sizes: Vec<usize>,
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

    Ok(ProportionSamplingDesign {
        schema_version: SAMPLING_DESIGN_SCHEMA_VERSION,
        population_size,
        expected_proportion,
        confidence_level,
        critical_value,
        margin_of_error,
        uncorrected_sample_size,
        sample_size,
        finite_population_correction,
        allocation_method,
        stratum_sample_sizes,
    })
}

fn allocate_strata(
    sample_size: usize,
    strata: &[SamplingStratum],
    method: AllocationMethod,
) -> Result<Vec<usize>, String> {
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
    let mut active = vec![true; strata.len()];
    let mut remaining = sample_size;

    loop {
        let weight_sum: f64 = weights
            .iter()
            .zip(&active)
            .filter(|(_, enabled)| **enabled)
            .map(|(weight, _)| *weight)
            .sum();
        if !weight_sum.is_finite() || weight_sum <= 0.0 {
            return Err("stratum allocation has no positive finite weight".into());
        }
        let capped = active.iter().enumerate().find_map(|(index, enabled)| {
            if !enabled {
                return None;
            }
            let quota = remaining as f64 * weights[index] / weight_sum;
            (quota >= strata[index].population_size as f64).then_some(index)
        });
        match capped {
            Some(index) => {
                allocated[index] = strata[index].population_size;
                remaining -= allocated[index];
                active[index] = false;
                if remaining == 0 {
                    break;
                }
            }
            None => {
                let mut fractions = Vec::new();
                for (index, enabled) in active.iter().enumerate() {
                    if !enabled {
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
                break;
            }
        }
    }

    if allocated.iter().any(|count| *count == 0) {
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
        assert!((design.uncorrected_sample_size - 384.145_882).abs() < 1e-5);
        assert!((design.finite_population_correction - (722.0_f64 / 999.0).sqrt()).abs() < 1e-12);
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
        assert_eq!(proportional.sample_size, 43);
        assert_eq!(neyman.sample_size, 43);
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
}

//! Ordered proficiency summaries from calibrated posterior draws.
//!
//! This module owns only ordinal probability and uncertainty arithmetic. It does
//! not estimate a latent trait, establish cut-score validity, link an assessment
//! to an external framework, average ordinal labels, or authorize a decision.

use std::error::Error;
use std::fmt::{Display, Formatter};

const PROBABILITY_TOLERANCE: f64 = 1.0e-12;
const MAX_POSTERIOR_SAMPLES: usize = 1_000_000;
const MAX_CREDIBLE_INTERVAL_CANDIDATES: usize = 20_000_000;

/// Borrowed inputs for one ordered proficiency-domain summary.
#[derive(Clone, Copy, Debug)]
pub struct OrderedProfileInput<'a> {
    /// Finite posterior draws on one already-calibrated latent scale.
    pub posterior_samples: &'a [f64],
    /// Optional finite, nonnegative draw weights with positive total mass.
    pub sample_weights: Option<&'a [f64]>,
    /// Strictly increasing finite boundaries between adjacent ordered levels.
    pub cut_scores: &'a [f64],
    /// Requested posterior mass for the contiguous credible-level set.
    pub credible_mass: f64,
}

/// Deterministic ordinal summary derived from posterior draws and cut scores.
#[derive(Clone, Debug, PartialEq)]
pub struct OrderedProfileSummary {
    /// Normalized posterior probability for every ordered level.
    pub level_probabilities: Vec<f64>,
    /// Unique modal level when it lies in the credible set, otherwise `None`.
    pub reported_level_index: Option<usize>,
    /// Shortest contiguous ordered-level interval meeting `credible_mass`.
    pub credible_level_indices: Vec<usize>,
    /// Weighted posterior mean on the supplied latent scale.
    pub posterior_mean: f64,
    /// Weighted posterior standard deviation used as score uncertainty.
    pub posterior_standard_deviation: f64,
}

/// Fail-closed validation and numerical errors for ordered profile summaries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OrderedProfileError {
    /// No posterior draw was supplied.
    EmptyPosterior,
    /// The posterior draw count exceeds the package work budget.
    PosteriorWorkLimit { samples: usize },
    /// A posterior draw was NaN or infinite.
    NonFinitePosterior { index: usize },
    /// The optional weight vector did not match the draw count.
    WeightLengthMismatch { expected: usize, actual: usize },
    /// A weight was negative, NaN, or infinite.
    InvalidWeight { index: usize },
    /// All admitted weights were zero.
    ZeroWeightMass,
    /// Finite individual weights overflowed during accumulation.
    NonFiniteWeightMass,
    /// No boundary was supplied for the ordered scale.
    EmptyCutScores,
    /// A cut score was NaN or infinite.
    NonFiniteCutScore { index: usize },
    /// Cut scores were not strictly increasing in their supplied order.
    NonIncreasingCutScores { index: usize },
    /// The requested credible mass was not finite and in `(0, 1]`.
    InvalidCredibleMass,
    /// The worst-case contiguous credible-set search exceeds the package work budget.
    CredibleIntervalWorkLimit { levels: usize },
    /// A posterior moment or normalized probability became non-finite.
    NonFinitePosteriorMoment,
}

impl Display for OrderedProfileError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPosterior => formatter.write_str("posterior samples must be nonempty"),
            Self::PosteriorWorkLimit { samples } => write!(
                formatter,
                "posterior summary for {samples} draws exceeds the package work limit"
            ),
            Self::NonFinitePosterior { index } => {
                write!(formatter, "posterior sample at index {index} must be finite")
            }
            Self::WeightLengthMismatch { expected, actual } => write!(
                formatter,
                "sample weights must contain {expected} values, received {actual}"
            ),
            Self::InvalidWeight { index } => write!(
                formatter,
                "sample weight at index {index} must be finite and nonnegative"
            ),
            Self::ZeroWeightMass => formatter.write_str("sample weights must have positive mass"),
            Self::NonFiniteWeightMass => {
                formatter.write_str("sample weight mass exceeds the finite numeric range")
            }
            Self::EmptyCutScores => formatter.write_str("cut scores must be nonempty"),
            Self::NonFiniteCutScore { index } => {
                write!(formatter, "cut score at index {index} must be finite")
            }
            Self::NonIncreasingCutScores { index } => write!(
                formatter,
                "cut score at index {index} must exceed its predecessor"
            ),
            Self::InvalidCredibleMass => {
                formatter.write_str("credible mass must be finite and in (0, 1]")
            }
            Self::CredibleIntervalWorkLimit { levels } => write!(
                formatter,
                "credible-level interval search for {levels} levels exceeds the package work limit"
            ),
            Self::NonFinitePosteriorMoment => {
                formatter.write_str("posterior summary exceeded the finite numeric range")
            }
        }
    }
}

impl Error for OrderedProfileError {}

/// Summarize one calibrated posterior into ordered level probabilities.
///
/// An exact cut score belongs to the level above that boundary. Inputs are
/// validated in their supplied order; the function never sorts cut scores,
/// repairs weights, or forces a reported level outside the credible set. Posterior
/// draw/weight pairs are canonically ordered internally so a joint permutation
/// produces bit-identical output.
pub fn summarize_ordered_profile(
    input: OrderedProfileInput<'_>,
) -> Result<OrderedProfileSummary, OrderedProfileError> {
    validate_input(input)?;

    let mut pairs = match input.sample_weights {
        Some(weights) => input
            .posterior_samples
            .iter()
            .copied()
            .zip(weights.iter().copied())
            .collect::<Vec<_>>(),
        None => input
            .posterior_samples
            .iter()
            .copied()
            .map(|sample| (sample, 1.0))
            .collect::<Vec<_>>(),
    };
    pairs.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.total_cmp(&right.1))
    });

    let total_weight = compensated_sum(pairs.iter().map(|(_, weight)| *weight));
    if total_weight == 0.0 {
        return Err(OrderedProfileError::ZeroWeightMass);
    }
    if !total_weight.is_finite() {
        return Err(OrderedProfileError::NonFiniteWeightMass);
    }

    let level_count = input.cut_scores.len() + 1;
    let mut level_sums = vec![0.0; level_count];
    let mut level_corrections = vec![0.0; level_count];
    for (sample, weight) in &pairs {
        let level_index = input
            .cut_scores
            .partition_point(|cut_score| *sample >= *cut_score);
        compensated_add(
            &mut level_sums[level_index],
            &mut level_corrections[level_index],
            *weight,
        );
    }

    let mut level_probabilities = level_sums
        .into_iter()
        .zip(level_corrections)
        .map(|(sum, correction)| (sum + correction) / total_weight)
        .collect::<Vec<_>>();
    let probability_mass = compensated_sum(level_probabilities.iter().copied());
    if !probability_mass.is_finite() || probability_mass <= 0.0 {
        return Err(OrderedProfileError::NonFinitePosteriorMoment);
    }
    for probability in &mut level_probabilities {
        *probability /= probability_mass;
    }

    let posterior_mean = compensated_sum(
        pairs
            .iter()
            .map(|(sample, weight)| sample * (weight / total_weight)),
    );
    if !posterior_mean.is_finite() {
        return Err(OrderedProfileError::NonFinitePosteriorMoment);
    }

    let mut variance_terms = Vec::with_capacity(pairs.len());
    for (sample, weight) in &pairs {
        let centered = sample - posterior_mean;
        let term = (weight / total_weight) * centered * centered;
        if !term.is_finite() {
            return Err(OrderedProfileError::NonFinitePosteriorMoment);
        }
        variance_terms.push(term);
    }
    let posterior_variance = compensated_sum(variance_terms);
    if !posterior_variance.is_finite() || posterior_variance < 0.0 {
        return Err(OrderedProfileError::NonFinitePosteriorMoment);
    }

    let credible_level_indices = shortest_contiguous_interval(
        &level_probabilities,
        input.credible_mass,
    );
    let reported_level_index = unique_modal_level(&level_probabilities)
        .filter(|index| credible_level_indices.contains(index));

    Ok(OrderedProfileSummary {
        reported_level_index,
        credible_level_indices,
        level_probabilities,
        posterior_mean,
        posterior_standard_deviation: posterior_variance.sqrt(),
    })
}

fn validate_input(input: OrderedProfileInput<'_>) -> Result<(), OrderedProfileError> {
    if input.posterior_samples.is_empty() {
        return Err(OrderedProfileError::EmptyPosterior);
    }
    if input.posterior_samples.len() > MAX_POSTERIOR_SAMPLES {
        return Err(OrderedProfileError::PosteriorWorkLimit {
            samples: input.posterior_samples.len(),
        });
    }

    if let Some(weights) = input.sample_weights {
        if weights.len() != input.posterior_samples.len() {
            return Err(OrderedProfileError::WeightLengthMismatch {
                expected: input.posterior_samples.len(),
                actual: weights.len(),
            });
        }
    }

    if input.cut_scores.is_empty() {
        return Err(OrderedProfileError::EmptyCutScores);
    }

    if !input.credible_mass.is_finite()
        || input.credible_mass <= 0.0
        || input.credible_mass > 1.0
    {
        return Err(OrderedProfileError::InvalidCredibleMass);
    }

    let level_count = input
        .cut_scores
        .len()
        .checked_add(1)
        .ok_or(OrderedProfileError::CredibleIntervalWorkLimit { levels: usize::MAX })?;
    let adjacent_count = level_count
        .checked_add(1)
        .ok_or(OrderedProfileError::CredibleIntervalWorkLimit { levels: level_count })?;
    let candidate_count = if level_count % 2 == 0 {
        (level_count / 2).checked_mul(adjacent_count)
    } else {
        level_count.checked_mul(adjacent_count / 2)
    }
    .ok_or(OrderedProfileError::CredibleIntervalWorkLimit { levels: level_count })?;
    if candidate_count > MAX_CREDIBLE_INTERVAL_CANDIDATES {
        return Err(OrderedProfileError::CredibleIntervalWorkLimit { levels: level_count });
    }

    if let Some(weights) = input.sample_weights {
        let mut weight_sum = 0.0;
        let mut weight_correction = 0.0;
        for (index, weight) in weights.iter().enumerate() {
            if !weight.is_finite() || *weight < 0.0 {
                return Err(OrderedProfileError::InvalidWeight { index });
            }
            compensated_add(&mut weight_sum, &mut weight_correction, *weight);
        }
        let weight_mass = weight_sum + weight_correction;
        if weight_mass == 0.0 {
            return Err(OrderedProfileError::ZeroWeightMass);
        }
        if !weight_mass.is_finite() {
            return Err(OrderedProfileError::NonFiniteWeightMass);
        }
    }

    for (index, sample) in input.posterior_samples.iter().enumerate() {
        if !sample.is_finite() {
            return Err(OrderedProfileError::NonFinitePosterior { index });
        }
    }

    for (index, cut_score) in input.cut_scores.iter().enumerate() {
        if !cut_score.is_finite() {
            return Err(OrderedProfileError::NonFiniteCutScore { index });
        }
        if index > 0 && *cut_score <= input.cut_scores[index - 1] {
            return Err(OrderedProfileError::NonIncreasingCutScores { index });
        }
    }

    Ok(())
}

fn unique_modal_level(probabilities: &[f64]) -> Option<usize> {
    let maximum = probabilities
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut modal_index = None;
    for (index, probability) in probabilities.iter().enumerate() {
        if (*probability - maximum).abs() <= PROBABILITY_TOLERANCE {
            if modal_index.is_some() {
                return None;
            }
            modal_index = Some(index);
        }
    }
    modal_index
}

fn shortest_contiguous_interval(probabilities: &[f64], target_mass: f64) -> Vec<usize> {
    let mut best: Option<(usize, usize, f64)> = None;
    for start in 0..probabilities.len() {
        let mut mass_sum = 0.0;
        let mut mass_correction = 0.0;
        for (end, probability) in probabilities.iter().enumerate().skip(start) {
            compensated_add(&mut mass_sum, &mut mass_correction, *probability);
            let mass = mass_sum + mass_correction;
            if mass < target_mass {
                continue;
            }
            let candidate_length = end - start + 1;
            let replace = match best {
                None => true,
                Some((best_start, best_end, best_mass)) => {
                    let best_length = best_end - best_start + 1;
                    candidate_length < best_length
                        || (candidate_length == best_length && mass > best_mass)
                }
            };
            if replace {
                best = Some((start, end, mass));
            }
            // Starts are visited in ascending order, so an exact length-and-mass
            // tie already preserves the required lower-start interval.
            break;
        }
    }

    let (start, end, _) = best.unwrap_or((0, probabilities.len() - 1, 1.0));
    (start..=end).collect()
}

fn compensated_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for value in values {
        compensated_add(&mut sum, &mut correction, value);
    }
    sum + correction
}

fn compensated_add(sum: &mut f64, correction: &mut f64, value: f64) {
    let updated = *sum + value;
    if sum.abs() >= value.abs() {
        *correction += (*sum - updated) + value;
    } else {
        *correction += (value - updated) + *sum;
    }
    *sum = updated;
}

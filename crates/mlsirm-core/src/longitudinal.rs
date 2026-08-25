//! Rust-owned longitudinal state estimation for repeated psychometric observations.
//!
//! The estimator is intentionally small and explicit. The compatibility wire
//! label `random_intercept_slope` denotes an independent per-respondent
//! ordinary-least-squares trend; it does not estimate a population random-
//! effects distribution or apply shrinkage. A stationary AR(1) state uses the
//! caller-supplied discrete-occasion coefficient and produces one-step latent
//! predictions. Respondents are independent, so the CPU path shards them
//! across scoped threads and reduces diagnostics in respondent order for
//! deterministic results. The item/factor likelihood remains in the existing
//! CPU/GPU Rust kernels; this module owns only the repeated-measurement state
//! layer described by the multilevel RFC.
//!
//! Missing observations are represented by `NaN` and are excluded from fitting
//! while retaining a predicted state at every declared occasion. Time offsets
//! are exact milliseconds at the boundary and are converted to days only for
//! the OLS design matrix. The AR coefficient is a discrete occasion
//! parameter, not a continuous-time decay parameter.

use std::thread;

const MILLIS_PER_DAY: f64 = 86_400_000.0;
const MAX_ABS_TIME_DAYS: f64 = 10_000_000.0;
const MAX_AR_SEQUENCE_GAP: usize = i32::MAX as usize;

/// Result of a validated independent-OLS-trend or stationary-AR state fit.
#[derive(Clone, Debug, PartialEq)]
pub struct LongitudinalStateFit {
    /// Predicted latent state aligned with the flattened occasion input.
    pub state: Vec<f64>,
    /// Respondent-level intercept estimates, in respondent input order.
    pub intercepts: Vec<f64>,
    /// Respondent-level slope estimates in outcome units per day.
    pub slopes: Vec<f64>,
    /// The fixed or validated AR coefficient. Zero for intercept/slope fits.
    pub ar_coefficient: f64,
    /// RMSE over observed values or one-step AR predictions.
    pub rmse: f64,
    /// Number of finite outcome observations used in fitting.
    pub observed_count: usize,
    /// Number of observed respondent transitions used by the AR diagnostic.
    pub transition_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StateKind {
    IndependentRespondentOlsTrend,
    StationaryAutoregressive,
}

#[derive(Clone, Debug)]
struct RespondentFit {
    state: Vec<f64>,
    intercept: f64,
    slope: f64,
    squared_error: f64,
    observed_count: usize,
    transition_count: usize,
}

fn parse_state_kind(value: &str) -> Result<StateKind, String> {
    match value {
        "random_intercept_slope" => Ok(StateKind::IndependentRespondentOlsTrend),
        "stationary_autoregressive" => Ok(StateKind::StationaryAutoregressive),
        _ => Err(
            "state_kind must be random_intercept_slope or stationary_autoregressive".to_string(),
        ),
    }
}

fn validate_offsets(row_offsets: &[usize], n_values: usize) -> Result<usize, String> {
    if row_offsets.is_empty() || row_offsets[0] != 0 {
        return Err("row_offsets must be non-empty and start at zero".to_string());
    }
    if row_offsets.windows(2).any(|window| window[1] < window[0]) {
        return Err("row_offsets must be non-decreasing".to_string());
    }
    if row_offsets.last().copied() != Some(n_values) {
        return Err("row_offsets must end at the value count".to_string());
    }
    Ok(row_offsets.len() - 1)
}

fn checked_ar_gap(gap: usize) -> Result<i32, String> {
    i32::try_from(gap)
        .map_err(|_| "accumulated AR occasion gap exceeds the supported range".to_string())
}

fn respondent_sequence_span(sequences: &[usize]) -> Result<usize, String> {
    let Some(first) = sequences.first().copied() else {
        return Ok(0);
    };
    sequences[sequences.len() - 1]
        .checked_sub(first)
        .ok_or_else(|| "sequence indices must increase within the supported AR gap".to_string())
}

fn validate_inputs(
    row_offsets: &[usize],
    sequence_indices: &[usize],
    time_offsets_milliseconds: &[i64],
    values: &[f64],
    state_kind: StateKind,
    ar_coefficient: Option<f64>,
) -> Result<(usize, f64), String> {
    if sequence_indices.len() != values.len() {
        return Err("sequence_indices and values must have equal length".to_string());
    }
    if time_offsets_milliseconds.len() != values.len() {
        return Err("time_offsets_milliseconds and values must have equal length".to_string());
    }
    let respondents = validate_offsets(row_offsets, values.len())?;
    for row in 0..respondents {
        let start = row_offsets[row];
        let end = row_offsets[row + 1];
        let sequences = &sequence_indices[start..end];
        if sequences
            .windows(2)
            .any(|window| window[1] <= window[0] || window[1] - window[0] > MAX_AR_SEQUENCE_GAP)
        {
            return Err("sequence indices must increase within the supported AR gap".to_string());
        }
        if !sequences.is_empty() {
            checked_ar_gap(respondent_sequence_span(sequences)?)?;
        }
        if time_offsets_milliseconds[start..end]
            .windows(2)
            .any(|window| window[1] <= window[0])
        {
            return Err("time offsets must increase strictly within each respondent".to_string());
        }
    }
    let phi = match state_kind {
        StateKind::IndependentRespondentOlsTrend => {
            if ar_coefficient.is_some() {
                return Err("random_intercept_slope does not accept an AR coefficient".to_string());
            }
            0.0
        }
        StateKind::StationaryAutoregressive => {
            let value = ar_coefficient.ok_or_else(|| {
                "stationary_autoregressive requires an AR coefficient".to_string()
            })?;
            if !value.is_finite() || !(-1.0 < value && value < 1.0) {
                return Err(
                    "AR coefficient must be finite and strictly between -1 and 1".to_string(),
                );
            }
            value
        }
    };
    for &value in values {
        if !value.is_finite() && !value.is_nan() {
            return Err("values must be finite or NaN for missing observations".to_string());
        }
    }
    for &offset in time_offsets_milliseconds {
        let days = offset as f64 / MILLIS_PER_DAY;
        if !days.is_finite() || days.abs() > MAX_ABS_TIME_DAYS {
            return Err("time offsets exceed the supported finite range".to_string());
        }
    }
    Ok((respondents, phi))
}

fn slope_is_identified(denominator: f64, max_abs_deviation: f64) -> bool {
    denominator > f64::EPSILON * max_abs_deviation * max_abs_deviation
}

fn fit_intercept_slope(times: &[i64], values: &[f64]) -> Result<RespondentFit, String> {
    let first_time = times.first().copied().unwrap_or(0);
    let x: Vec<f64> = times
        .iter()
        .map(|value| ((*value - first_time) as f64) / MILLIS_PER_DAY)
        .collect();
    let observed: Vec<(f64, f64)> = x
        .iter()
        .zip(values)
        .filter_map(|(time, value)| value.is_finite().then_some((*time, *value)))
        .collect();
    if observed.is_empty() {
        return Ok(RespondentFit {
            state: vec![0.0; values.len()],
            intercept: 0.0,
            slope: 0.0,
            squared_error: 0.0,
            observed_count: 0,
            transition_count: 0,
        });
    }
    let count = observed.len() as f64;
    let mean_x = observed.iter().map(|(time, _)| time).sum::<f64>() / count;
    let mean_y = observed.iter().map(|(_, value)| value).sum::<f64>() / count;
    let denominator = observed
        .iter()
        .map(|(time, _)| (time - mean_x).powi(2))
        .sum::<f64>();
    let max_abs_deviation = observed
        .iter()
        .map(|(time, _)| (time - mean_x).abs())
        .fold(0.0_f64, f64::max);
    let numerator = observed
        .iter()
        .map(|(time, value)| (time - mean_x) * (value - mean_y))
        .sum::<f64>();
    // Time offsets are strictly increasing before this private fitter runs.
    // Degeneracy is therefore either an identified intercept-only case
    // (fewer than two finite observations) or a genuine scale-relative
    // collapse that must fail closed rather than invent a zero slope.
    let slope = if observed.len() < 2 {
        0.0
    } else if slope_is_identified(denominator, max_abs_deviation) {
        numerator / denominator
    } else {
        return Err("OLS slope is degenerate relative to the time scale".to_string());
    };
    let intercept = mean_y - slope * mean_x;
    let state: Vec<f64> = x.iter().map(|time| intercept + slope * time).collect();
    let squared_error = observed
        .iter()
        .map(|(time, value)| (value - (intercept + slope * time)).powi(2))
        .sum::<f64>();
    Ok(RespondentFit {
        state,
        intercept,
        slope,
        squared_error,
        observed_count: observed.len(),
        transition_count: 0,
    })
}

fn fit_ar(sequence_indices: &[usize], values: &[f64], phi: f64) -> Result<RespondentFit, String> {
    let mut state = vec![0.0; values.len()];
    let observed: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let Some(first) = values.iter().position(|value| value.is_finite()) else {
        return Ok(RespondentFit {
            state,
            intercept: 0.0,
            slope: 0.0,
            squared_error: 0.0,
            observed_count: 0,
            transition_count: 0,
        });
    };
    // Psychometric latent scores use the identified zero-centered origin. The
    // state specification has no free intercept, so estimating a respondent
    // mean here would reintroduce an unanchored location parameter.
    let mean = 0.0;
    let mut previous_index = first;
    let mut previous_observed = values[first];
    state[first] = previous_observed;
    let mut squared_error = 0.0;
    let mut transition_count = 0;
    for index in (first + 1)..values.len() {
        let gap = sequence_indices[index]
            .checked_sub(sequence_indices[previous_index])
            .ok_or_else(|| "AR sequence indices must increase".to_string())?;
        let exponent = checked_ar_gap(gap)?;
        let prediction = mean + phi.powi(exponent) * (previous_observed - mean);
        state[index] = prediction;
        if values[index].is_finite() {
            squared_error += (values[index] - prediction).powi(2);
            transition_count += 1;
            previous_index = index;
            previous_observed = values[index];
        }
    }
    Ok(RespondentFit {
        state,
        intercept: mean,
        slope: 0.0,
        squared_error,
        observed_count: observed.len(),
        transition_count,
    })
}

fn map_worker_join<T>(joined: thread::Result<T>) -> Result<T, String> {
    joined.map_err(|_| "longitudinal worker failed".to_string())
}

fn require_respondent_fit(fit: Option<RespondentFit>) -> Result<RespondentFit, String> {
    fit.ok_or_else(|| "a respondent state fit is missing".to_string())
}

/// Fit respondent-level repeated-measurement states on the Rust CPU path.
///
/// `row_offsets` partitions flattened occasions by respondent. The caller
/// supplies a validated state specification from the Python contract, while
/// this function independently rechecks all array and numeric invariants at
/// the trust boundary. `worker_count` controls deterministic respondent
/// sharding; a value larger than the respondent count is capped.
pub fn fit_longitudinal_state(
    row_offsets: &[usize],
    sequence_indices: &[usize],
    time_offsets_milliseconds: &[i64],
    values: &[f64],
    state_kind: &str,
    ar_coefficient: Option<f64>,
    worker_count: usize,
) -> Result<LongitudinalStateFit, String> {
    if worker_count == 0 {
        return Err("worker_count must be at least one".to_string());
    }
    let kind = parse_state_kind(state_kind)?;
    let (respondent_count, phi) = validate_inputs(
        row_offsets,
        sequence_indices,
        time_offsets_milliseconds,
        values,
        kind,
        ar_coefficient,
    )?;
    let mut fits: Vec<Option<RespondentFit>> = (0..respondent_count).map(|_| None).collect();
    if respondent_count > 0 {
        let workers = worker_count.min(respondent_count);
        let chunk = respondent_count.div_ceil(workers);
        let joined: Result<(), String> = thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers);
            for worker in 0..workers {
                let start = worker * chunk;
                let end = (start + chunk).min(respondent_count);
                if start >= end {
                    continue;
                }
                handles.push(scope.spawn(move || {
                    (start..end)
                        .map(|row| {
                            let value_start = row_offsets[row];
                            let value_end = row_offsets[row + 1];
                            let sequences = &sequence_indices[value_start..value_end];
                            let times = &time_offsets_milliseconds[value_start..value_end];
                            let row_values = &values[value_start..value_end];
                            let fit = match kind {
                                StateKind::IndependentRespondentOlsTrend => {
                                    fit_intercept_slope(times, row_values)
                                }
                                StateKind::StationaryAutoregressive => {
                                    fit_ar(sequences, row_values, phi)
                                }
                            }?;
                            Ok((row, fit))
                        })
                        .collect::<Result<Vec<_>, String>>()
                }));
            }
            for handle in handles {
                let rows = map_worker_join(handle.join())??;
                for (row, fit) in rows {
                    fits[row] = Some(fit);
                }
            }
            Ok(())
        });
        joined?;
    }
    let mut state = Vec::with_capacity(values.len());
    let mut intercepts = Vec::with_capacity(respondent_count);
    let mut slopes = Vec::with_capacity(respondent_count);
    let mut squared_error = 0.0;
    let mut observed_count = 0;
    let mut transition_count = 0;
    for fit in fits {
        let fit = require_respondent_fit(fit)?;
        state.extend(fit.state);
        intercepts.push(fit.intercept);
        slopes.push(fit.slope);
        squared_error += fit.squared_error;
        observed_count += fit.observed_count;
        transition_count += fit.transition_count;
    }
    let denominator = if kind == StateKind::StationaryAutoregressive {
        transition_count
    } else {
        observed_count
    };
    Ok(LongitudinalStateFit {
        state,
        intercepts,
        slopes,
        ar_coefficient: phi,
        rmse: if denominator == 0 {
            0.0
        } else {
            (squared_error / denominator as f64).sqrt()
        },
        observed_count,
        transition_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_two_respondent_intercept_slopes_and_is_worker_deterministic() {
        let offsets = [0, 3, 6];
        let sequences = [0, 1, 2, 0, 1, 2];
        let times = [0, 86_400_000, 172_800_000, 0, 86_400_000, 172_800_000];
        let values = [2.0, 3.5, 5.0, -1.0, -3.0, -5.0];
        let one = fit_longitudinal_state(
            &offsets,
            &sequences,
            &times,
            &values,
            "random_intercept_slope",
            None,
            1,
        )
        .unwrap();
        let many = fit_longitudinal_state(
            &offsets,
            &sequences,
            &times,
            &values,
            "random_intercept_slope",
            None,
            8,
        )
        .unwrap();
        assert_eq!(one, many);
        assert_eq!(one.intercepts, vec![2.0, -1.0]);
        assert_eq!(one.slopes, vec![1.5, -2.0]);
        assert!(one.rmse < 1e-12);
    }

    #[test]
    fn recovers_noisy_ols_trends_with_bounded_rmse() {
        let respondents = 12_usize;
        let occasions = 6_usize;
        let mut offsets = vec![0];
        let mut sequences = Vec::new();
        let mut times = Vec::new();
        let mut values = Vec::new();
        let mut true_intercepts = Vec::new();
        let mut true_slopes = Vec::new();
        for respondent in 0..respondents {
            let intercept = (respondent as f64) * 0.25 - 1.0;
            let slope = 0.5 - (respondent as f64) * 0.05;
            true_intercepts.push(intercept);
            true_slopes.push(slope);
            for occasion in 0..occasions {
                let days = occasion as f64;
                sequences.push(occasion);
                times.push((days * MILLIS_PER_DAY) as i64);
                let noise = if occasion % 2 == 0 { 0.01 } else { -0.01 };
                values.push(intercept + slope * days + noise);
            }
            offsets.push(values.len());
        }
        let fit = fit_longitudinal_state(
            &offsets,
            &sequences,
            &times,
            &values,
            "random_intercept_slope",
            None,
            4,
        )
        .unwrap();
        let intercept_rmse = true_intercepts
            .iter()
            .zip(&fit.intercepts)
            .map(|(truth, estimate)| (truth - estimate).powi(2))
            .sum::<f64>()
            .sqrt()
            / (respondents as f64).sqrt();
        let slope_rmse = true_slopes
            .iter()
            .zip(&fit.slopes)
            .map(|(truth, estimate)| (truth - estimate).powi(2))
            .sum::<f64>()
            .sqrt()
            / (respondents as f64).sqrt();
        assert!(intercept_rmse < 0.02, "{intercept_rmse}");
        assert!(slope_rmse < 0.02, "{slope_rmse}");
        assert!(fit.rmse < 0.02, "{}", fit.rmse);
        assert_eq!(fit.observed_count, respondents * occasions);
    }

    #[test]
    fn ar_state_starts_from_first_finite_observation() {
        let fit = fit_longitudinal_state(
            &[0, 3],
            &[0, 1, 2],
            &[0, 86_400_000, 172_800_000],
            &[f64::NAN, 1.0, 0.4],
            "stationary_autoregressive",
            Some(0.4),
            1,
        )
        .unwrap();
        assert_eq!(fit.state[0], 0.0);
        assert_eq!(fit.state[1], 1.0);
        assert!((fit.state[2] - 0.4).abs() < 1e-12);
        assert_eq!(fit.observed_count, 2);
        assert_eq!(fit.transition_count, 1);
        assert!(fit.rmse < 1e-12);
    }

    #[test]
    fn ar_state_preserves_missing_occasion_and_recovers_prediction_error() {
        let offsets = [0, 4];
        let sequences = [0, 1, 2, 3];
        let times = [0, 86_400_000, 172_800_000, 259_200_000];
        let values = [1.0, f64::NAN, 0.25, 0.125];
        let fit = fit_longitudinal_state(
            &offsets,
            &sequences,
            &times,
            &values,
            "stationary_autoregressive",
            Some(0.5),
            2,
        )
        .unwrap();
        assert_eq!(fit.ar_coefficient, 0.5);
        assert_eq!(fit.observed_count, 3);
        assert_eq!(fit.transition_count, 2);
        assert_eq!(fit.state[1], 0.5);
        assert_eq!(fit.state[2], 0.25);
        assert_eq!(fit.state[3], 0.125);
        assert!(fit.rmse < 1e-12);
    }

    #[test]
    fn recovers_caller_supplied_ar_series_with_bounded_rmse() {
        let phi = 0.6;
        let start = 1.25;
        let mut values = vec![start];
        for _ in 0..7 {
            let next = phi * values[values.len() - 1];
            values.push(next);
        }
        let sequences: Vec<usize> = (0..values.len()).collect();
        let times: Vec<i64> = sequences
            .iter()
            .map(|step| (*step as i64) * 86_400_000)
            .collect();
        let fit = fit_longitudinal_state(
            &[0, values.len()],
            &sequences,
            &times,
            &values,
            "stationary_autoregressive",
            Some(phi),
            1,
        )
        .unwrap();
        assert_eq!(fit.ar_coefficient, phi);
        assert_eq!(fit.transition_count, values.len() - 1);
        assert!(fit.rmse < 1e-12, "{}", fit.rmse);
        assert!((fit.state[0] - start).abs() < 1e-12);
    }

    #[test]
    fn intercept_only_and_all_missing_rows_are_identified() {
        let empty =
            fit_longitudinal_state(&[0], &[], &[], &[], "random_intercept_slope", None, 3).unwrap();
        assert!(empty.state.is_empty());
        assert_eq!(empty.observed_count, 0);
        assert_eq!(empty.rmse, 0.0);

        let intercept_only = fit_longitudinal_state(
            &[0, 1, 1],
            &[0],
            &[0],
            &[4.0],
            "random_intercept_slope",
            None,
            2,
        )
        .unwrap();
        assert_eq!(intercept_only.intercepts, vec![4.0, 0.0]);
        assert_eq!(intercept_only.slopes, vec![0.0, 0.0]);
        assert_eq!(intercept_only.observed_count, 1);
        assert_eq!(intercept_only.rmse, 0.0);

        let missing_ar = fit_longitudinal_state(
            &[0, 2],
            &[0, 1],
            &[0, 1],
            &[f64::NAN, f64::NAN],
            "stationary_autoregressive",
            Some(0.3),
            1,
        )
        .unwrap();
        assert_eq!(missing_ar.observed_count, 0);
        assert_eq!(missing_ar.transition_count, 0);
        assert_eq!(missing_ar.rmse, 0.0);
        assert_eq!(missing_ar.state, vec![0.0, 0.0]);
    }

    #[test]
    fn rejects_invalid_contracts_without_panicking() {
        let values = [1.0, 2.0];
        let sequences = [0, 1];
        let times = [0, 1];
        for (offsets, kind, phi, message) in [
            (&[1, 2][..], "random_intercept_slope", None, "row_offsets"),
            (&[0, 2][..], "stationary_autoregressive", None, "requires"),
            (
                &[0, 2][..],
                "stationary_autoregressive",
                Some(1.0),
                "strictly",
            ),
        ] {
            let error = fit_longitudinal_state(offsets, &sequences, &times, &values, kind, phi, 1)
                .unwrap_err();
            assert!(error.contains(message), "{error}");
        }
        assert!(
            fit_longitudinal_state(&[0, 2], &sequences, &times, &values, "unknown", None, 1)
                .unwrap_err()
                .contains("state_kind")
        );
        assert!(fit_longitudinal_state(
            &[0, 2],
            &sequences,
            &times,
            &values,
            "random_intercept_slope",
            None,
            0
        )
        .unwrap_err()
        .contains("worker_count"));
        assert!(fit_longitudinal_state(
            &[0, 2],
            &[0, MAX_AR_SEQUENCE_GAP + 1],
            &times,
            &values,
            "stationary_autoregressive",
            Some(0.5),
            1
        )
        .unwrap_err()
        .contains("supported AR gap"));
        assert!(fit_longitudinal_state(
            &[0, 2],
            &[0],
            &times,
            &values,
            "random_intercept_slope",
            None,
            1
        )
        .unwrap_err()
        .contains("equal length"));
        assert!(fit_longitudinal_state(
            &[0, 2],
            &sequences,
            &[0],
            &values,
            "random_intercept_slope",
            None,
            1
        )
        .unwrap_err()
        .contains("time_offsets_milliseconds"));
        assert!(validate_offsets(&[0, 1, 0], 0)
            .unwrap_err()
            .contains("non-decreasing"));
        assert!(validate_offsets(&[0, 1], 2)
            .unwrap_err()
            .contains("end at the value count"));
        assert!(fit_longitudinal_state(
            &[0, 2],
            &[1, 0],
            &times,
            &values,
            "random_intercept_slope",
            None,
            1
        )
        .unwrap_err()
        .contains("sequence indices"));
        assert!(fit_longitudinal_state(
            &[0, 2],
            &sequences,
            &[2, 1],
            &values,
            "random_intercept_slope",
            None,
            1
        )
        .unwrap_err()
        .contains("time offsets"));
        assert!(fit_longitudinal_state(
            &[0, 2],
            &sequences,
            &times,
            &values,
            "random_intercept_slope",
            Some(0.1),
            1
        )
        .unwrap_err()
        .contains("does not accept"));
        assert!(fit_longitudinal_state(
            &[0, 2],
            &sequences,
            &times,
            &values,
            "stationary_autoregressive",
            Some(f64::NAN),
            1
        )
        .unwrap_err()
        .contains("strictly"));
        assert!(fit_longitudinal_state(
            &[0, 2],
            &sequences,
            &times,
            &[1.0, f64::INFINITY],
            "random_intercept_slope",
            None,
            1
        )
        .unwrap_err()
        .contains("finite or NaN"));
        assert!(fit_longitudinal_state(
            &[0, 2],
            &sequences,
            &[0, 86_400_000_i64.saturating_mul(10_000_001 * 2)],
            &values,
            "random_intercept_slope",
            None,
            1
        )
        .unwrap_err()
        .contains("supported finite range"));
    }

    #[test]
    fn unused_worker_chunk_is_skipped_without_changing_estimates() {
        // Four respondents and three workers yield chunk=2, so the third
        // worker starts at index 4 and is skipped. The remaining shards still
        // produce the independent intercept-only estimates.
        let offsets = [0, 1, 2, 3, 4];
        let sequences = [0, 0, 0, 0];
        let times = [0, 0, 0, 0];
        let values = [1.0, 2.0, 3.0, 4.0];
        let fit = fit_longitudinal_state(
            &offsets,
            &sequences,
            &times,
            &values,
            "random_intercept_slope",
            None,
            3,
        )
        .unwrap();
        assert_eq!(fit.intercepts, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(fit.slopes, vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(fit.observed_count, 4);
        assert_eq!(fit.rmse, 0.0);
    }

    #[test]
    fn package_owned_join_and_missing_fit_errors_are_stable() {
        let join_error = map_worker_join::<()>(Err(Box::new("boom")));
        assert_eq!(join_error.unwrap_err(), "longitudinal worker failed");
        assert_eq!(
            require_respondent_fit(None).unwrap_err(),
            "a respondent state fit is missing"
        );
        assert!(checked_ar_gap(MAX_AR_SEQUENCE_GAP + 1)
            .unwrap_err()
            .contains("accumulated AR occasion gap"));
        assert_eq!(respondent_sequence_span(&[]).unwrap(), 0);
        assert_eq!(respondent_sequence_span(&[5]).unwrap(), 0);
        assert_eq!(respondent_sequence_span(&[0, 3]).unwrap(), 3);
        assert!(respondent_sequence_span(&[5, 2])
            .unwrap_err()
            .contains("sequence indices must increase"));
        assert!(!slope_is_identified(0.0, 0.0));
        assert!(slope_is_identified(1e-16, 1e-8));
    }

    #[test]
    fn fit_ar_rejects_decreasing_sequence_after_validation_bypass() {
        let error = fit_ar(&[2, 1], &[1.0, 0.5], 0.4).unwrap_err();
        assert!(
            error.contains("AR sequence indices must increase"),
            "{error}"
        );
    }

    #[test]
    fn intercept_slope_fails_closed_on_scale_relative_degeneracy() {
        let error = fit_intercept_slope(&[0, 0], &[1.0, 2.0]).unwrap_err();
        assert!(error.contains("degenerate"), "{error}");
        let empty = fit_intercept_slope(&[], &[]).unwrap();
        assert_eq!(empty.observed_count, 0);
        assert_eq!(empty.slope, 0.0);
    }
}

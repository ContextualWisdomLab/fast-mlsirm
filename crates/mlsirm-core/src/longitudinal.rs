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
//! the random-slope design matrix. The AR coefficient is a discrete occasion
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
        if sequence_indices[start..end]
            .windows(2)
            .any(|window| {
                window[1] <= window[0]
                    || window[1] - window[0] > MAX_AR_SEQUENCE_GAP
            })
        {
            return Err("sequence indices must increase within the supported AR gap".to_string());
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

fn fit_intercept_slope(times: &[i64], values: &[f64]) -> RespondentFit {
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
        return RespondentFit {
            state: vec![0.0; values.len()],
            intercept: 0.0,
            slope: 0.0,
            squared_error: 0.0,
            observed_count: 0,
            transition_count: 0,
        };
    }
    let count = observed.len() as f64;
    let mean_x = observed.iter().map(|(time, _)| time).sum::<f64>() / count;
    let mean_y = observed.iter().map(|(_, value)| value).sum::<f64>() / count;
    let denominator = observed
        .iter()
        .map(|(time, _)| (time - mean_x).powi(2))
        .sum::<f64>();
    let numerator = observed
        .iter()
        .map(|(time, value)| (time - mean_x) * (value - mean_y))
        .sum::<f64>();
    // Time offsets are strictly increasing before this private fitter runs. A
    // zero centered sum of squares therefore represents an actually
    // unidentified one-observation trend, not a scale-dependent epsilon
    // threshold. Sub-day and millisecond intervals remain legitimate slopes.
    let slope = if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    };
    let intercept = mean_y - slope * mean_x;
    let state: Vec<f64> = x.iter().map(|time| intercept + slope * time).collect();
    let squared_error = observed
        .iter()
        .map(|(time, value)| (value - (intercept + slope * time)).powi(2))
        .sum::<f64>();
    RespondentFit {
        state,
        intercept,
        slope,
        squared_error,
        observed_count: observed.len(),
        transition_count: 0,
    }
}

fn fit_ar(
    sequence_indices: &[usize],
    values: &[f64],
    phi: f64,
) -> Result<RespondentFit, String> {
    let mut state = vec![0.0; values.len()];
    let observed: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if observed.is_empty() {
        return Ok(RespondentFit {
            state,
            intercept: 0.0,
            slope: 0.0,
            squared_error: 0.0,
            observed_count: 0,
            transition_count: 0,
        });
    }
    // Psychometric latent scores use the identified zero-centered origin. The
    // state specification has no free intercept, so estimating a respondent
    // mean here would reintroduce an unanchored location parameter.
    let mean = 0.0;
    let first = values
        .iter()
        .position(|value| value.is_finite())
        .ok_or_else(|| "AR state fit has no finite observation".to_string())?;
    let mut previous_index = first;
    let mut previous_observed = values[first];
    state[first] = previous_observed;
    let mut squared_error = 0.0;
    let mut transition_count = 0;
    for index in (first + 1)..values.len() {
        let gap = sequence_indices[index]
            .checked_sub(sequence_indices[previous_index])
            .ok_or_else(|| "AR sequence indices must increase".to_string())?;
        let exponent = i32::try_from(gap).map_err(|_| {
            "accumulated AR occasion gap exceeds the supported range".to_string()
        })?;
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
                                    Ok(fit_intercept_slope(times, row_values))
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
                let rows = handle
                    .join()
                    .map_err(|_| "longitudinal worker failed".to_string())??;
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
        let fit = fit.ok_or_else(|| "a respondent state fit is missing".to_string())?;
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
    }
}

//! Provider-neutral expected-utility and value-of-information arithmetic.
//!
//! The decision table is intentionally explicit: callers provide a prior over
//! finite states, utility for every action/state pair, intervention costs, and
//! optionally a joint state/signal distribution.  This module does not infer
//! utilities, causal effects, or probabilities from text or model scores.
//!
//! For action `a` and state `s`, the net intervention value is
//! `U(a, s) - U(no_action, s) - cost(a)`.  The selected action maximizes its
//! prior expected net value.  With a joint state/signal distribution `p(s,z)`,
//! sample information is evaluated without unstable posterior division as
//! `sum_z max_a sum_s p(s,z) net(a,s)`.
//!
//! The expected-value identities follow Howard (1966) and Raiffa and
//! Schlaifer (1961).  They are decision-analytic quantities over caller-
//! supplied distributions and utilities, not estimates of causal effects.

/// Maximum number of explicit actions in one decision table.
pub const MAX_DECISION_ACTIONS: usize = 1024;
/// Maximum number of explicit uncertain states in one decision table.
pub const MAX_DECISION_STATES: usize = 4096;
/// Maximum number of explicit signals in one sample-information table.
pub const MAX_DECISION_SIGNALS: usize = 1024;
/// Maximum dense action/state or signal/state cells admitted per evaluation.
pub const MAX_DECISION_CELLS: usize = 1_000_000;

const PROBABILITY_TOLERANCE: f64 = 1e-12;

/// Rust-owned result for one finite decision table.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionSupportResult {
    /// Prior expected net intervention value for each action, in input order.
    pub action_expected_net_values: Vec<f64>,
    /// Index of the prior-optimal action; exact ties select the lowest index.
    pub selected_action: usize,
    /// Prior expected net intervention value of [`Self::selected_action`].
    pub expected_net_intervention_value: f64,
    /// Expected value of perfect state information before information cost.
    pub expected_value_perfect_information: f64,
    /// Expected value of the supplied sample information, when supplied.
    pub expected_value_sample_information: Option<f64>,
    /// Expected value of the supplied sample information less its cost.
    pub net_expected_value_sample_information: Option<f64>,
}

/// Evaluate expected net intervention value, EVPI, and optional EVSI.
///
/// `action_utilities` is row-major with shape `(action_count, state_count)`.
/// `signal_joint_probabilities`, when present, is row-major with shape
/// `(signal_count, state_count)` and represents the joint distribution
/// `P(signal, state)`, not an arbitrary collection of posterior vectors.  Its
/// state marginal must agree with `state_probabilities` within the documented
/// binary64 probability tolerance.
pub fn evaluate_decision_support(
    state_probabilities: &[f64],
    action_utilities: &[f64],
    action_count: usize,
    state_count: usize,
    intervention_costs: &[f64],
    no_action_index: usize,
    signal_joint_probabilities: Option<&[f64]>,
    signal_count: usize,
    information_cost: f64,
) -> Result<DecisionSupportResult, String> {
    validate_dimensions(
        state_probabilities,
        action_utilities,
        action_count,
        state_count,
        intervention_costs,
        no_action_index,
        signal_joint_probabilities,
        signal_count,
    )?;
    if !information_cost.is_finite() || information_cost < 0.0 {
        return Err("information_cost must be finite and non-negative".to_owned());
    }

    let state_mass = validate_probability_vector(state_probabilities, "state_probabilities")?;
    validate_utilities(
        action_utilities,
        action_count,
        state_count,
        intervention_costs,
        no_action_index,
    )?;
    let action_expected_net_values = action_expected_values(
        state_probabilities,
        action_utilities,
        action_count,
        state_count,
        intervention_costs,
        no_action_index,
    )?;
    let selected_action = argmax_first(&action_expected_net_values);
    let expected_net_intervention_value = action_expected_net_values[selected_action];
    let perfect_information_value = perfect_information_value(
        state_probabilities,
        action_utilities,
        action_count,
        state_count,
        intervention_costs,
        no_action_index,
    )?;
    let expected_value_perfect_information =
        perfect_information_value - expected_net_intervention_value;
    if !expected_value_perfect_information.is_finite() {
        return Err("expected_value_perfect_information is non-finite".to_owned());
    }

    let (expected_value_sample_information, net_expected_value_sample_information) =
        match signal_joint_probabilities {
            None => (None, None),
            Some(joint) => {
                let joint_mass = validate_probability_vector(joint, "signal_joint_probabilities")?;
                validate_signal_marginal(
                    joint,
                    signal_count,
                    state_count,
                    state_probabilities,
                    state_mass,
                    joint_mass,
                )?;
                let with_information = sample_information_value(
                    joint,
                    signal_count,
                    state_count,
                    action_count,
                    action_utilities,
                    intervention_costs,
                    no_action_index,
                )?;
                let sample_value = with_information - expected_net_intervention_value;
                let net_sample_value = sample_value - information_cost;
                if !sample_value.is_finite() || !net_sample_value.is_finite() {
                    return Err("sample-information value is non-finite".to_owned());
                }
                (Some(sample_value), Some(net_sample_value))
            }
        };

    Ok(DecisionSupportResult {
        action_expected_net_values,
        selected_action,
        expected_net_intervention_value,
        expected_value_perfect_information,
        expected_value_sample_information,
        net_expected_value_sample_information,
    })
}

fn validate_dimensions(
    state_probabilities: &[f64],
    action_utilities: &[f64],
    action_count: usize,
    state_count: usize,
    intervention_costs: &[f64],
    no_action_index: usize,
    signal_joint_probabilities: Option<&[f64]>,
    signal_count: usize,
) -> Result<(), String> {
    if action_count == 0 || action_count > MAX_DECISION_ACTIONS {
        return Err(format!(
            "action_count must be between 1 and {MAX_DECISION_ACTIONS}"
        ));
    }
    if state_count == 0 || state_count > MAX_DECISION_STATES {
        return Err(format!(
            "state_count must be between 1 and {MAX_DECISION_STATES}"
        ));
    }
    let utility_cells = action_count
        .checked_mul(state_count)
        .ok_or_else(|| "action utility table dimensions overflow".to_owned())?;
    if utility_cells > MAX_DECISION_CELLS {
        return Err(format!(
            "action utility table exceeds {MAX_DECISION_CELLS} cells"
        ));
    }
    if action_utilities.len() != utility_cells {
        return Err("action_utilities length does not match its declared shape".to_owned());
    }
    if state_probabilities.len() != state_count {
        return Err("state_probabilities length does not match state_count".to_owned());
    }
    if intervention_costs.len() != action_count {
        return Err("intervention_costs length does not match action_count".to_owned());
    }
    if no_action_index >= action_count {
        return Err("no_action_index must identify one action".to_owned());
    }
    match signal_joint_probabilities {
        None if signal_count != 0 => {
            Err("signal_count must be zero when sample information is absent".to_owned())
        }
        None => Ok(()),
        Some(joint) => {
            if signal_count == 0 || signal_count > MAX_DECISION_SIGNALS {
                return Err(format!(
                    "signal_count must be between 1 and {MAX_DECISION_SIGNALS}"
                ));
            }
            let signal_cells = signal_count
                .checked_mul(state_count)
                .ok_or_else(|| "signal joint table dimensions overflow".to_owned())?;
            if signal_cells > MAX_DECISION_CELLS {
                return Err(format!(
                    "signal joint table exceeds {MAX_DECISION_CELLS} cells"
                ));
            }
            if joint.len() != signal_cells {
                return Err(
                    "signal_joint_probabilities length does not match its declared shape"
                        .to_owned(),
                );
            }
            Ok(())
        }
    }
}

fn validate_probability_vector(values: &[f64], name: &str) -> Result<f64, String> {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(format!("{name}[{index}] must be finite and non-negative"));
        }
        compensated_add(&mut sum, &mut correction, value);
    }
    let mass = sum + correction;
    if !mass.is_finite() || mass <= 0.0 {
        return Err(format!("{name} must have finite positive total mass"));
    }
    if (mass - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(format!("{name} must sum to one within 1e-12"));
    }
    Ok(mass)
}

fn validate_utilities(
    action_utilities: &[f64],
    action_count: usize,
    state_count: usize,
    intervention_costs: &[f64],
    no_action_index: usize,
) -> Result<(), String> {
    for (index, &value) in action_utilities.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("action_utilities[{index}] must be finite"));
        }
    }
    for (index, &cost) in intervention_costs.iter().enumerate() {
        if !cost.is_finite() || cost < 0.0 {
            return Err(format!(
                "intervention_costs[{index}] must be finite and non-negative"
            ));
        }
    }
    if intervention_costs[no_action_index] != 0.0 {
        return Err("no-action intervention cost must be zero".to_owned());
    }
    for action in 0..action_count {
        for state in 0..state_count {
            let baseline = action_utilities[no_action_index * state_count + state];
            let contrast = action_utilities[action * state_count + state] - baseline;
            let net = contrast - intervention_costs[action];
            if !contrast.is_finite() || !net.is_finite() {
                return Err(format!(
                    "action utility contrast is non-finite at action {action}, state {state}"
                ));
            }
        }
    }
    Ok(())
}

fn action_expected_values(
    state_probabilities: &[f64],
    action_utilities: &[f64],
    action_count: usize,
    state_count: usize,
    intervention_costs: &[f64],
    no_action_index: usize,
) -> Result<Vec<f64>, String> {
    let mut result = Vec::with_capacity(action_count);
    for action in 0..action_count {
        let mut sum = 0.0;
        let mut correction = 0.0;
        for state in 0..state_count {
            let net = net_utility(
                action,
                state,
                state_count,
                action_utilities,
                intervention_costs,
                no_action_index,
            );
            let term = state_probabilities[state] * net;
            if !term.is_finite() {
                return Err("expected net utility is non-finite".to_owned());
            }
            compensated_add(&mut sum, &mut correction, term);
        }
        let value = sum + correction;
        if !value.is_finite() {
            return Err("expected net utility is non-finite".to_owned());
        }
        result.push(value);
    }
    Ok(result)
}

fn perfect_information_value(
    state_probabilities: &[f64],
    action_utilities: &[f64],
    action_count: usize,
    state_count: usize,
    intervention_costs: &[f64],
    no_action_index: usize,
) -> Result<f64, String> {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for state in 0..state_count {
        let mut best = net_utility(
            0,
            state,
            state_count,
            action_utilities,
            intervention_costs,
            no_action_index,
        );
        for action in 1..action_count {
            let value = net_utility(
                action,
                state,
                state_count,
                action_utilities,
                intervention_costs,
                no_action_index,
            );
            if value > best {
                best = value;
            }
        }
        let term = state_probabilities[state] * best;
        if !term.is_finite() {
            return Err("perfect-information value is non-finite".to_owned());
        }
        compensated_add(&mut sum, &mut correction, term);
    }
    let value = sum + correction;
    if value.is_finite() {
        Ok(value)
    } else {
        Err("perfect-information value is non-finite".to_owned())
    }
}

fn validate_signal_marginal(
    joint: &[f64],
    signal_count: usize,
    state_count: usize,
    state_probabilities: &[f64],
    state_mass: f64,
    joint_mass: f64,
) -> Result<(), String> {
    if (joint_mass - state_mass).abs() > PROBABILITY_TOLERANCE {
        return Err("state and signal joint probabilities have different total mass".to_owned());
    }
    for state in 0..state_count {
        let mut sum = 0.0;
        let mut correction = 0.0;
        for signal in 0..signal_count {
            compensated_add(
                &mut sum,
                &mut correction,
                joint[signal * state_count + state],
            );
        }
        let marginal = sum + correction;
        if (marginal - state_probabilities[state]).abs() > PROBABILITY_TOLERANCE {
            return Err(format!(
                "signal joint state marginal does not match state_probabilities[{state}]"
            ));
        }
    }
    Ok(())
}

fn sample_information_value(
    joint: &[f64],
    signal_count: usize,
    state_count: usize,
    action_count: usize,
    action_utilities: &[f64],
    intervention_costs: &[f64],
    no_action_index: usize,
) -> Result<f64, String> {
    let mut total = 0.0;
    let mut total_correction = 0.0;
    for signal in 0..signal_count {
        let mut best = f64::NEG_INFINITY;
        for action in 0..action_count {
            let mut sum = 0.0;
            let mut correction = 0.0;
            for state in 0..state_count {
                let net = net_utility(
                    action,
                    state,
                    state_count,
                    action_utilities,
                    intervention_costs,
                    no_action_index,
                );
                let term = joint[signal * state_count + state] * net;
                if !term.is_finite() {
                    return Err("sample-information value is non-finite".to_owned());
                }
                compensated_add(&mut sum, &mut correction, term);
            }
            let value = sum + correction;
            if value > best {
                best = value;
            }
        }
        compensated_add(&mut total, &mut total_correction, best);
    }
    let value = total + total_correction;
    if value.is_finite() {
        Ok(value)
    } else {
        Err("sample-information value is non-finite".to_owned())
    }
}

fn net_utility(
    action: usize,
    state: usize,
    state_count: usize,
    action_utilities: &[f64],
    intervention_costs: &[f64],
    no_action_index: usize,
) -> f64 {
    action_utilities[action * state_count + state]
        - action_utilities[no_action_index * state_count + state]
        - intervention_costs[action]
}

fn argmax_first(values: &[f64]) -> usize {
    let mut best_index = 0;
    for index in 1..values.len() {
        if values[index] > values[best_index] {
            best_index = index;
        }
    }
    best_index
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

#[cfg(test)]
#[path = "../../../tests/unit/decision_support_tests.rs"]
mod tests;

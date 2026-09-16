use super::*;

fn evaluate(
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
    evaluate_decision_support(
        state_probabilities,
        action_utilities,
        action_count,
        state_count,
        intervention_costs,
        no_action_index,
        signal_joint_probabilities,
        signal_count,
        information_cost,
    )
}

#[test]
fn computes_expected_net_value_and_perfect_information() {
    let result = evaluate(
        &[0.75, 0.25],
        &[0.0, 0.0, 10.0, -10.0],
        2,
        2,
        &[0.0, 1.0],
        0,
        None,
        0,
        0.0,
    )
    .expect("decision table");

    assert_eq!(result.selected_action, 1);
    assert_eq!(result.action_expected_net_values.len(), 2);
    assert!((result.action_expected_net_values[1] - 4.0).abs() < 1e-12);
    assert!((result.expected_net_intervention_value - 4.0).abs() < 1e-12);
    assert!((result.expected_value_perfect_information - 2.75).abs() < 1e-12);
    assert_eq!(result.expected_value_sample_information, None);
    assert_eq!(result.net_expected_value_sample_information, None);
}

#[test]
fn computes_sample_information_from_a_joint_distribution() {
    let result = evaluate(
        &[0.75, 0.25],
        &[0.0, 0.0, 10.0, -10.0],
        2,
        2,
        &[0.0, 1.0],
        0,
        Some(&[0.75, 0.0, 0.0, 0.25]),
        2,
        0.5,
    )
    .expect("decision table");

    assert!((result.expected_value_sample_information.unwrap() - 2.75).abs() < 1e-12);
    assert!((result.net_expected_value_sample_information.unwrap() - 2.25).abs() < 1e-12);
}

#[test]
fn uses_declared_no_action_row_and_first_index_tie_break() {
    let result =
        evaluate(&[1.0], &[10.0, 0.0], 2, 1, &[0.0, 0.0], 1, None, 0, 0.0).expect("decision table");
    assert_eq!(result.selected_action, 0);
    assert_eq!(result.action_expected_net_values, vec![10.0, 0.0]);

    let tie = evaluate(
        &[0.5, 0.5],
        &[0.0, 0.0, 1.0, -1.0],
        2,
        2,
        &[0.0, 0.0],
        0,
        None,
        0,
        0.0,
    )
    .expect("tie decision table");
    assert_eq!(tie.selected_action, 0);
}

#[test]
fn zero_probability_signal_is_valid_and_has_no_value() {
    let result = evaluate(
        &[1.0, 0.0],
        &[0.0, 0.0, 4.0, -4.0],
        2,
        2,
        &[0.0, 0.0],
        0,
        Some(&[1.0, 0.0, 0.0, 0.0]),
        2,
        0.0,
    )
    .expect("zero-probability signal");
    assert_eq!(result.expected_value_sample_information, Some(0.0));
    assert_eq!(result.net_expected_value_sample_information, Some(0.0));
}

#[test]
fn rejects_invalid_dimensions_and_probability_contracts() {
    assert!(evaluate(&[], &[0.0], 1, 0, &[0.0], 0, None, 0, 0.0)
        .unwrap_err()
        .contains("state_count"));
    assert!(evaluate(&[1.0], &[], 1, 1, &[0.0], 0, None, 0, 0.0)
        .unwrap_err()
        .contains("action_utilities length"));
    assert!(evaluate(&[1.0], &[0.0], 0, 1, &[], 0, None, 0, 0.0)
        .unwrap_err()
        .contains("action_count"));
    assert!(evaluate(&[1.0], &[0.0], 1, 1, &[0.0], 1, None, 0, 0.0)
        .unwrap_err()
        .contains("no_action_index"));
    assert!(evaluate(&[1.0], &[0.0], 1, 1, &[0.0], 0, None, 1, 0.0)
        .unwrap_err()
        .contains("signal_count"));
    assert!(
        evaluate(&[1.0], &[0.0], 1, 1, &[0.0], 0, Some(&[1.0]), 0, 0.0)
            .unwrap_err()
            .contains("signal_count")
    );
    assert!(evaluate(&[1.0], &[0.0], 1, 1, &[0.0], 0, Some(&[]), 1, 0.0)
        .unwrap_err()
        .contains("signal_joint_probabilities length"));
    assert!(evaluate(&[0.5], &[0.0], 1, 1, &[0.0], 0, None, 0, 0.0)
        .unwrap_err()
        .contains("sum to one"));
    assert!(evaluate(&[-1.0], &[0.0], 1, 1, &[0.0], 0, None, 0, 0.0)
        .unwrap_err()
        .contains("state_probabilities[0]"));
    assert!(evaluate(&[1.0], &[0.0], 1, 1, &[-1.0], 0, None, 0, 0.0)
        .unwrap_err()
        .contains("intervention_costs[0]"));
    assert!(evaluate(&[1.0], &[0.0], 1, 1, &[1.0], 0, None, 0, 0.0)
        .unwrap_err()
        .contains("no-action intervention cost"));
    assert!(evaluate(&[1.0], &[0.0], 1, 1, &[0.0], 0, None, 0, -1.0)
        .unwrap_err()
        .contains("information_cost"));
    assert!(
        evaluate(&[1.0], &[f64::NAN], 1, 1, &[0.0], 0, None, 0, 0.0,)
            .unwrap_err()
            .contains("action_utilities[0]")
    );
    assert!(evaluate(
        &[1.0 - 9e-13],
        &[0.0],
        1,
        1,
        &[0.0],
        0,
        Some(&[1.0 + 9e-13]),
        1,
        0.0,
    )
    .unwrap_err()
    .contains("different total mass"));
    assert!(evaluate(
        &[0.75, 0.25],
        &[0.0, 0.0],
        1,
        2,
        &[0.0],
        0,
        Some(&[1.0, 0.0]),
        1,
        0.0,
    )
    .unwrap_err()
    .contains("state marginal"));
}

#[test]
fn rejects_nonfinite_utility_contrasts() {
    assert!(evaluate(
        &[1.0],
        &[f64::MAX, -f64::MAX],
        2,
        1,
        &[0.0, 0.0],
        1,
        None,
        0,
        0.0,
    )
    .unwrap_err()
    .contains("utility contrast"));
}

#[test]
fn rejects_oversized_evsi_work_before_value_validation() {
    let action_count = 500;
    let state_count = 100;
    let signal_count = 500;
    let state_probabilities = vec![1.0 / state_count as f64; state_count];
    let mut action_utilities = vec![0.0; action_count * state_count];
    action_utilities[0] = f64::NAN;
    let intervention_costs = vec![0.0; action_count];
    let signal_joint_probabilities =
        vec![1.0 / (signal_count * state_count) as f64; signal_count * state_count];

    let error = evaluate(
        &state_probabilities,
        &action_utilities,
        action_count,
        state_count,
        &intervention_costs,
        0,
        Some(&signal_joint_probabilities),
        signal_count,
        0.0,
    )
    .unwrap_err();

    assert!(error.contains("EVSI work exceeds"), "{error}");
}

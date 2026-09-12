use mlsirm_core::personfit_np::person_fit_np;

fn cumulative(values: &[f64]) -> Vec<f64> {
    let mut acc = 0.0;
    values
        .iter()
        .map(|&value| {
            acc += value;
            acc
        })
        .collect()
}

fn legacy_zu3(responses: &[Vec<f64>]) -> Vec<f64> {
    let n_persons = responses.len();
    let n_items = responses[0].len();
    let n_persons_f64 = n_persons as f64;

    let scores: Vec<f64> = responses.iter().map(|row| row.iter().sum()).collect();
    let probabilities: Vec<f64> = (0..n_items)
        .map(|item| {
            responses
                .iter()
                .map(|row| row[item])
                .sum::<f64>()
                / n_persons_f64
        })
        .collect();
    let log_odds: Vec<f64> = probabilities
        .iter()
        .map(|&probability| {
            let value = (probability / (1.0 - probability)).ln();
            if value.is_finite() { value } else { 0.0 }
        })
        .collect();

    let mut log_odds_descending = log_odds.clone();
    log_odds_descending.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let mut log_odds_ascending = log_odds.clone();
    log_odds_ascending.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cumulative_descending = cumulative(&log_odds_descending);
    let cumulative_ascending = cumulative(&log_odds_ascending);

    // Preserve the five independent reductions that preceded the production
    // single-pass fold. This helper is intentionally test-only legacy evidence.
    let s1: f64 = probabilities
        .iter()
        .zip(&log_odds)
        .map(|(&probability, &logit)| probability * logit)
        .sum();
    let s2: f64 = probabilities
        .iter()
        .zip(&log_odds)
        .map(|(&probability, &logit)| probability * (1.0 - probability) * logit)
        .sum();
    let s3: f64 = probabilities.iter().sum();
    let s4: f64 = probabilities
        .iter()
        .map(|&probability| probability * (1.0 - probability))
        .sum();
    let beta = probabilities
        .iter()
        .zip(&log_odds)
        .map(|(&probability, &logit)| {
            probability * (1.0 - probability) * logit * logit
        })
        .sum::<f64>()
        - s2 * s2 / s4;

    scores
        .iter()
        .enumerate()
        .map(|(person, &score)| {
            if score == 0.0 || score == n_items as f64 {
                return f64::NAN;
            }
            let k = score as usize;
            let first = cumulative_descending[k - 1];
            let last = cumulative_ascending[k - 1];
            let observed_log_odds: f64 = responses[person]
                .iter()
                .zip(&log_odds)
                .map(|(&response, &logit)| response * logit)
                .sum();
            let u3 = (first - observed_log_odds) / (first - last);
            let alpha = s1 + s2 * (score - s3) / s4;
            let expected = (first - alpha) / (first - last);
            let variance = beta / ((first - last) * (first - last));
            let value = (u3 - expected) / variance.sqrt();
            if value.is_finite() { value } else { f64::NAN }
        })
        .collect()
}

fn assert_same_f64(actual: f64, expected: f64, fixture: usize, person: usize) {
    if actual.is_nan() && expected.is_nan() {
        return;
    }
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "ZU3 legacy parity failure for fixture {fixture}, person {person}: actual={actual:?}, expected={expected:?}"
    );
}

#[test]
fn production_zu3_matches_pre_fold_reference() {
    let fixtures = [
        vec![
            vec![1.0, 1.0, 1.0, 0.0, 0.0],
            vec![1.0, 1.0, 0.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        vec![
            vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            vec![1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
    ];

    for (fixture_index, responses) in fixtures.iter().enumerate() {
        let expected = legacy_zu3(responses);
        let actual = person_fit_np(responses).expect("fixture must satisfy public data admission");
        assert_eq!(actual.zu3.len(), expected.len());
        for (person, (&actual_zu3, &expected_zu3)) in
            actual.zu3.iter().zip(&expected).enumerate()
        {
            assert_same_f64(actual_zu3, expected_zu3, fixture_index, person);
        }
    }
}

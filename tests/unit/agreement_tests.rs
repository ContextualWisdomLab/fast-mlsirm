use super::*;

#[test]
fn kappa_hand_computed_2x2() {
    // table: a\b -> [[20, 5], [10, 65]], n = 100
    let mut a = Vec::new();
    let mut b = Vec::new();
    for (x, y, count) in [(0, 0, 20), (0, 1, 5), (1, 0, 10), (1, 1, 65)] {
        for _ in 0..count {
            a.push(x);
            b.push(y);
        }
    }
    // po = .85; pe = .25*.30 + .75*.70 = .60; kappa = .25/.40 = .625
    let k = cohen_kappa(&a, &b, 2).unwrap();
    assert!((k - 0.625).abs() < 1e-9, "kappa {k}");
    // binary QWK equals unweighted kappa
    let qwk = quadratic_weighted_kappa(&a, &b, 2).unwrap();
    assert!((qwk - k).abs() < 1e-9);
    let (exact, adjacent) = agreement_rates(&a, &b).unwrap();
    assert!((exact - 0.85).abs() < 1e-9);
    assert!(
        (adjacent - 1.0).abs() < 1e-9,
        "binary adjacent is degenerate at 1"
    );
}

#[test]
fn smd_and_r_hand_computed() {
    let human = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
    let auto = [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    // p_h = .625, sd_h = sqrt(.625*.375); p_a = .75
    let expect = (0.75 - 0.625) / (0.625_f64 * 0.375).sqrt();
    assert!((smd(&auto, &human).unwrap() - expect).abs() < 1e-9);
    let r = pearson_r(&auto, &human).unwrap();
    assert!(r > 0.6 && r < 1.0);
}

#[test]
fn verdict_gates_flag_degradation() {
    // auto-human agreement clearly worse than human-human
    let human: Vec<u32> = (0..200).map(|i| (i % 2) as u32).collect();
    let auto: Vec<u32> = (0..200)
        .map(|i| {
            if i % 5 == 0 {
                1 - (i % 2) as u32
            } else {
                (i % 2) as u32
            }
        })
        .collect();
    let h2: Vec<u32> = human.clone(); // perfect human-human baseline
    let verdict = validate_scoring(&auto, &human, 2, Some((&human, &h2)), None).unwrap();
    let degr = verdict
        .gates
        .iter()
        .find(|g| g.name == "degradation")
        .unwrap();
    assert!(
        !degr.pass,
        "20% flips vs perfect baseline must flag degradation"
    );
    assert!(verdict.exact_agreement < 1.0);
}

#[test]
fn subgroup_smd_catches_biased_slice() {
    // group 1 systematically over-scored by the auto rater
    let mut auto = Vec::new();
    let mut human = Vec::new();
    let mut grp = Vec::new();
    let mut state = 9u64;
    let mut unif = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for i in 0..400 {
        let g = (i % 2) as u32;
        let h = if unif() < 0.5 { 1u32 } else { 0 };
        let a = if g == 1 && h == 0 && unif() < 0.5 {
            1
        } else {
            h
        };
        auto.push(a);
        human.push(h);
        grp.push(g);
    }
    let verdict = validate_scoring(&auto, &human, 2, None, Some(&grp)).unwrap();
    let sg = verdict
        .gates
        .iter()
        .find(|g| g.name == "subgroup_smd")
        .unwrap();
    assert!(
        !sg.pass,
        "inflated group-1 scores must flag the subgroup SMD gate"
    );
}

#[test]
fn rejects_degenerate_inputs() {
    assert!(cohen_kappa(&[0, 1], &[0], 2).is_err());
    assert!(quadratic_weighted_kappa(&[0, 0], &[0, 0], 2).is_err());
    assert!(pearson_r(&[1.0, 1.0], &[0.0, 1.0]).is_err());
    assert!(smd(&[1.0, 1.0], &[1.0, 1.0]).is_err());
    assert!(quadratic_weighted_kappa(&[0, 3], &[0, 1], 2).is_err());
    assert!(quadratic_weighted_kappa(&[0, 1], &[0, 1], 1).is_err());
    assert!(cohen_kappa(&[0, 0], &[0, 0], 2).is_err());
    assert!(pearson_r(&[1.0], &[1.0]).is_err());
    assert!(smd(&[1.0], &[1.0]).is_err());
    assert!(agreement_rates(&[], &[]).is_err());

    let auto = [0, 1, 0, 1];
    let human = [0, 1, 1, 0];
    assert!(validate_scoring(&auto, &human, 2, None, Some(&[0, 1])).is_err());
    let singleton = validate_scoring(&auto, &human, 2, None, Some(&[0, 1, 1, 1])).unwrap();
    assert!(singleton
        .gates
        .iter()
        .any(|gate| gate.name == "subgroup_smd"));
    let zero_variance_group =
        validate_scoring(&auto, &human, 2, None, Some(&[0, 0, 1, 1])).unwrap();
    assert!(zero_variance_group
        .gates
        .iter()
        .any(|gate| gate.name == "subgroup_smd"));
    let subgroup_human_zero_variance =
        validate_scoring(&auto, &[0, 0, 0, 1], 2, None, Some(&[0, 0, 1, 1])).unwrap();
    assert!(subgroup_human_zero_variance
        .gates
        .iter()
        .any(|gate| gate.name == "subgroup_smd"));
}

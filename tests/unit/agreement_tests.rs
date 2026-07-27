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

// ---------------------------------------------------------------------------
// fleiss_kappa (irr 0.85 kappam.fleiss; oracle anchors FK1-FK5 with exact
// Fractions). Every assert below reads crate outputs (FleissKappaResult
// fields returned by fleiss_kappa); pins are hand-derived rationals.
// z pins are pure arithmetic (rel 1e-12); p-value pins go through the
// crate's Numerical-Recipes erfc (|error| < 1.2e-7), hence abs 5e-7.
// ---------------------------------------------------------------------------

/// FK fixture: ns=5, nr=4, k=3 (asymmetric; 5x4 also breaks stride
/// transposition and row-vs-column chanceP confusion).
fn fk_fixture() -> Vec<i64> {
    vec![
        0, 0, 0, 1, // S1
        0, 1, 1, 1, // S2
        2, 2, 2, 2, // S3
        0, 0, 2, 2, // S4
        1, 1, 1, 0, // S5
    ]
}

fn rel_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1.0)
}

#[test]
fn fk_anchor_fk1_classic() {
    // Kills MU1 (agreeP drops -nr centering: kappa would be 113/133),
    // MU2 (row-sum chanceP: 1/5 -> kappa 11/24), and via the z pin MU4
    // (variance second-term sign: Sum p q (q-p) = 441/2000 != 0).
    let res = fleiss_kappa(&fk_fixture(), 5, 4, 3, false).unwrap();
    assert!(
        rel_eq(res.kappa, 139.0 / 399.0, 1e-15),
        "kappa {}",
        res.kappa
    );
    assert_eq!(res.subjects_used, 5);
    // var = 181/10830; z = kappa/sqrt(var) (oracle 2.694739854085488).
    assert!(
        rel_eq(res.z, (139.0 / 399.0) / (181.0_f64 / 10830.0).sqrt(), 1e-12),
        "z {}",
        res.z
    );
    assert!(
        (res.p_value - 0.007044360582468963).abs() < 5e-7,
        "p {}",
        res.p_value
    );
}

#[test]
fn fk_exact_fk2() {
    // Kills MU3 (exact == classic): Conger correction Sum s2_j = 3/50 gives
    // chanceP 8/25 and kappa 37/102 != 139/399.
    let res = fleiss_kappa(&fk_fixture(), 5, 4, 3, true).unwrap();
    assert!(
        rel_eq(res.kappa, 37.0 / 102.0, 1e-15),
        "exact kappa {}",
        res.kappa
    );
    assert_eq!(res.subjects_used, 5);
    // Exact mode returns no test statistic or detail (irr returns neither).
    assert!(res.z.is_nan() && res.p_value.is_nan());
    assert!(res.category_kappa.is_empty());
    assert!(res.category_z.is_empty());
    assert!(res.category_p.is_empty());
}

#[test]
fn fk_missing_drop_fk3() {
    // Kills MU5 (missing code counted as a category instead of listwise row
    // drop): the prepended row must vanish, leaving FK1 exactly.
    let mut ratings = vec![0, -1, 1, 2];
    ratings.extend(fk_fixture());
    let res = fleiss_kappa(&ratings, 6, 4, 3, false).unwrap();
    assert_eq!(res.subjects_used, 5);
    let base = fleiss_kappa(&fk_fixture(), 5, 4, 3, false).unwrap();
    assert_eq!(res.kappa, base.kappa, "drop must reproduce FK1 bitwise");
    assert!(rel_eq(res.kappa, 139.0 / 399.0, 1e-15));
}

#[test]
fn fk_category_detail_fk4() {
    // Kills MU6 (pjk drops the m*nr*p_j subtraction: category kappas would
    // be [51/91, 233/273, 73/63]).
    let res = fleiss_kappa(&fk_fixture(), 5, 4, 3, false).unwrap();
    let expect = [1.0 / 21.0, 31.0 / 91.0, 43.0 / 63.0];
    assert_eq!(res.category_kappa.len(), 3);
    for j in 0..3 {
        assert!(
            rel_eq(res.category_kappa[j], expect[j], 1e-15),
            "kappa_{j} {}",
            res.category_kappa[j]
        );
        // var_j = 1/30 for all j -> z_j = kappa_j * sqrt(30).
        assert!(
            rel_eq(res.category_z[j], expect[j] * 30.0_f64.sqrt(), 1e-12),
            "z_{j} {}",
            res.category_z[j]
        );
    }
    // Oracle p-values (through math.erfc; crate erfc abs 5e-7).
    let expect_p = [
        0.7942311156261253,
        0.062059828219847915,
        0.00018517759699332675,
    ];
    for j in 0..3 {
        assert!(
            (res.category_p[j] - expect_p[j]).abs() < 5e-7,
            "p_{j} {}",
            res.category_p[j]
        );
    }
}

#[test]
fn fk_empty_category_fk5() {
    // k=4 with no code 3: R's 0/0 -> NaN preserved; overall kappa unchanged
    // (C_3 = 0 contributes nothing to chanceP).
    let res = fleiss_kappa(&fk_fixture(), 5, 4, 4, false).unwrap();
    assert!(rel_eq(res.kappa, 139.0 / 399.0, 1e-15));
    assert_eq!(res.category_kappa.len(), 4);
    assert!(res.category_kappa[3].is_nan());
    assert!(res.category_z[3].is_nan());
    assert!(res.category_p[3].is_nan());
    // Non-empty categories unaffected by the extra level.
    assert!(rel_eq(res.category_kappa[0], 1.0 / 21.0, 1e-15));
}

#[test]
fn fk_error_contract() {
    let fk = fk_fixture();
    assert!(fleiss_kappa(&fk, 0, 4, 3, false).is_err(), "ns == 0");
    assert!(fleiss_kappa(&fk, 5, 1, 3, false).is_err(), "nr < 2");
    assert!(fleiss_kappa(&fk, 5, 4, 1, false).is_err(), "k < 2");
    assert!(
        fleiss_kappa(&[], 2_000_000, 2, 2, false).is_err(),
        "ns cap before ns*nr"
    );
    assert!(fleiss_kappa(&fk, 5, 4, 4, false).is_ok());
    assert!(
        fleiss_kappa(&fk[..19], 5, 4, 3, false).is_err(),
        "length mismatch"
    );
    assert!(
        fleiss_kappa(&[0, 1, 3, 0, 1, 2], 3, 2, 3, false).is_err(),
        "code >= k"
    );
    assert!(
        fleiss_kappa(&[-1, 0, 1, -1], 2, 2, 2, false).is_err(),
        "all rows dropped"
    );
    assert!(
        fleiss_kappa(&[1, 1, 1, 1, 1, 1], 3, 2, 2, false).is_err(),
        "degenerate chanceP == 1"
    );
}

/// 500-rep invariance MC: kappa (classic and exact), z, and category detail
/// are invariant under subject-row permutation; kappa and z are invariant
/// under rater-column permutation (ttab is unchanged by either; rtab is
/// permuted across raters, leaving Sum_j s2_j unchanged). Asserts compare
/// two crate outputs, so any asymmetry introduced into the aggregation
/// breaks them. Disclosure: this cannot detect a wrong-but-symmetric
/// formula; the FK1-FK5 value pins above are the discriminating anchors.
#[test]
#[ignore]
fn fk_mc_500_permutation_invariance() {
    struct Lcg(u64);
    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next_u64() >> 33) as usize % n
        }
    }
    let mut rng = Lcg(0x5eed_f1e5_5000_0001);
    let (ns, nr, k) = (8usize, 5usize, 4usize);
    for rep in 0..500 {
        let ratings: Vec<i64> = (0..ns * nr).map(|_| rng.below(k) as i64).collect();
        let Ok(base) = fleiss_kappa(&ratings, ns, nr, k, false) else {
            continue; // degenerate draw
        };
        let base_ex = fleiss_kappa(&ratings, ns, nr, k, true).unwrap();
        // Fisher-Yates over subject rows.
        let mut order: Vec<usize> = (0..ns).collect();
        for i in (1..ns).rev() {
            order.swap(i, rng.below(i + 1));
        }
        let by_subj: Vec<i64> = order
            .iter()
            .flat_map(|&i| ratings[i * nr..(i + 1) * nr].iter().copied())
            .collect();
        let perm = fleiss_kappa(&by_subj, ns, nr, k, false).unwrap();
        let perm_ex = fleiss_kappa(&by_subj, ns, nr, k, true).unwrap();
        assert!(
            rel_eq(perm.kappa, base.kappa, 1e-12),
            "rep {rep} subject-perm kappa"
        );
        assert!(rel_eq(perm.z, base.z, 1e-12), "rep {rep} subject-perm z");
        assert!(
            rel_eq(perm_ex.kappa, base_ex.kappa, 1e-12),
            "rep {rep} subject-perm exact"
        );
        for j in 0..k {
            let (a, b) = (perm.category_kappa[j], base.category_kappa[j]);
            assert!(
                (a.is_nan() && b.is_nan()) || rel_eq(a, b, 1e-12),
                "rep {rep} cat {j}"
            );
        }
        // Fisher-Yates over rater columns.
        let mut rorder: Vec<usize> = (0..nr).collect();
        for i in (1..nr).rev() {
            rorder.swap(i, rng.below(i + 1));
        }
        let by_rater: Vec<i64> = (0..ns)
            .flat_map(|i| {
                let ratings = &ratings;
                rorder.iter().map(move |&r| ratings[i * nr + r])
            })
            .collect();
        let rperm = fleiss_kappa(&by_rater, ns, nr, k, true).unwrap();
        assert!(
            rel_eq(rperm.kappa, base_ex.kappa, 1e-12),
            "rep {rep} rater-perm exact kappa"
        );
    }
}

use super::*;

#[test]
fn test_bifactor_grm_multiple_group_and_oakes_se() {
    // 16-item affectivity scale structure:
    // General factor (dim 0)
    // 4 specific factors (dims 1..=4)
    // 1 method factor W (dim 5) on reverse-keyed items (items 1, 3, 5, 7, 9, 11, 13, 15)
    let n_items = 16usize;
    let n_dims = 6usize;
    let n_cat = 4usize;
    let n_persons = 120usize;
    let n_groups = 2usize;

    // Build loading pattern: items x dims
    let mut loading_pattern = vec![0u8; n_items * n_dims];
    for i in 0..n_items {
        // General factor loads on all items
        loading_pattern[i * n_dims + 0] = 1;

        // 4 specific factors (4 items each)
        let s_dim = 1 + (i / 4);
        loading_pattern[i * n_dims + s_dim] = 1;

        // Method factor W on odd items
        if i % 2 == 1 {
            loading_pattern[i * n_dims + 5] = 1;
        }
    }

    // Assign groups: first 60 in group 0 (reference), next 60 in group 1 (focal)
    let mut group_ids = vec![0usize; n_persons];
    for p in 60..n_persons {
        group_ids[p] = 1;
    }

    // Generate synthetic response matrix with valid categories in 0..4
    let mut y = vec![0usize; n_persons * n_items];
    let mut st = 42u64;
    let mut rng = move || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / ((1u64 << 53) as f64)
    };

    for p in 0..n_persons {
        let g = group_ids[p];
        let th_g = if g == 0 { 0.0 } else { 0.5 };
        for i in 0..n_items {
            let score_prop = 0.5 + 0.1 * th_g + 0.2 * (rng() - 0.5);
            let cat = if score_prop < 0.25 {
                0
            } else if score_prop < 0.50 {
                1
            } else if score_prop < 0.75 {
                2
            } else {
                3
            };
            y[p * n_items + i] = cat;
        }
    }

    let cfg = BifactorGrmConfig {
        max_iter: 10,
        tol: 1e-3,
        ridge: 1e-6,
        newton_iter: 5,
        qmc_draws: 300,
        seed: 12345,
        slope_bound: Some(6.0),
        compute_oakes_se: true,
    };

    let res = fit_bifactor_grm(
        &y,
        None,
        Some(&group_ids),
        n_groups,
        &loading_pattern,
        n_persons,
        n_items,
        n_dims,
        n_cat,
        &cfg,
    ).expect("fit_bifactor_grm must succeed");

    assert_eq!(res.n_dims, 6);
    assert_eq!(res.n_items, 16);
    assert_eq!(res.n_cat, 4);
    assert_eq!(res.n_groups, 2);

    // Group 0 mean and variance must be reference standard normal
    for d in 0..n_dims {
        assert_eq!(res.group_means[d], 0.0);
        assert_eq!(res.group_variances[d], 1.0);
    }

    // Focal group 1 must have valid estimated moments
    assert!(res.group_means[n_dims].is_finite());
    assert!(res.group_variances[n_dims] > 0.0);

    // Slopes and thresholds must be finite and ordered
    assert_eq!(res.slope.len(), n_items * n_dims);
    assert_eq!(res.threshold.len(), n_items * 3);
    for i in 0..n_items {
        let b0 = res.threshold[i * 3 + 0];
        let b1 = res.threshold[i * 3 + 1];
        let b2 = res.threshold[i * 3 + 2];
        assert!(b0 >= b1, "Item {i} thresholds must be ordered: b0={b0} >= b1={b1}");
        assert!(b1 >= b2, "Item {i} thresholds must be ordered: b1={b1} >= b2={b2}");
    }

    // Oakes Standard Errors must be produced
    assert!(res.oakes_se_slope.is_some(), "Oakes SE for slopes must be computed");
    assert!(res.oakes_se_threshold.is_some(), "Oakes SE for thresholds must be computed");
    assert!(res.min_eigenvalue.is_some());
    assert!(res.min_eigenvalue.unwrap() > 0.0, "Observed information matrix must be positive definite");
    assert!(res.condition_number.is_some());
}

#[test]
fn test_bifactor_grm_slope_bound_sensitivity_monotonicity() {
    let n_items = 13usize; // 13-item avoidance coping scale
    let n_dims = 4usize;   // G + 3 specific factors
    let n_cat = 4usize;
    let n_persons = 80usize;

    let mut loading_pattern = vec![0u8; n_items * n_dims];
    for i in 0..n_items {
        loading_pattern[i * n_dims + 0] = 1; // General factor
        let s_dim = 1 + (i % 3);
        loading_pattern[i * n_dims + s_dim] = 1; // Specific factor
    }

    let mut y = vec![0usize; n_persons * n_items];
    let mut st = 9999u64;
    let mut rng = move || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for v in y.iter_mut() {
        *v = (rng() * 4.0).floor().min(3.0) as usize;
    }

    let base_cfg = BifactorGrmConfig {
        max_iter: 8,
        tol: 1e-3,
        ridge: 1e-6,
        newton_iter: 5,
        qmc_draws: 200,
        seed: 777,
        slope_bound: None,
        compute_oakes_se: false,
    };

    let candidate_bounds = [Some(4.0), Some(6.0), Some(8.0), Some(10.0)];
    let entries = fit_bifactor_slope_sensitivity(
        &y,
        None,
        None,
        1,
        &loading_pattern,
        n_persons,
        n_items,
        n_dims,
        n_cat,
        &candidate_bounds,
        &base_cfg,
    ).expect("fit_bifactor_slope_sensitivity must succeed");

    assert_eq!(entries.len(), 4);

    // Verify monotonically non-decreasing log-likelihood as upper bound increases
    for i in 0..entries.len() - 1 {
        let ll_curr = entries[i].loglik;
        let ll_next = entries[i + 1].loglik;
        // As bound is relaxed (4 -> 6 -> 8 -> 10), likelihood space is expanded so loglik is non-decreasing
        assert!(
            ll_next >= ll_curr - 1e-2,
            "Log-likelihood should be monotonically non-decreasing: bound {:?} (ll={}) vs bound {:?} (ll={})",
            entries[i].bound, ll_curr, entries[i+1].bound, ll_next
        );
    }
}

use super::*;

// Small LCG + Box-Muller for deterministic test data.
fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut st = seed.max(1);
    move || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / ((1u64 << 53) as f64)
    }
}
fn normal(u: &mut impl FnMut() -> f64) -> f64 {
    let u1 = u().max(1e-12);
    let u2 = u();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// R1: equipercentile self-equating is the exact identity at every integer
// score with positive frequency (the tightest correctness anchor).
#[test]
fn equate_self_is_identity() {
    let mut u = lcg(11);
    let k = 40usize;
    // a spread of scores covering the interior, all cells populated
    let scores: Vec<f64> = (0..4000)
        .map(|_| (8.0 + 24.0 * normal(&mut u)).round().clamp(0.0, k as f64))
        .collect();
    let g = rel_freq(&scores, k).unwrap();
    let res = equate_eg(&scores, &scores, k, k, EquateMethod::Equipercentile).unwrap();
    let mut maxdev = 0.0_f64;
    for x in 0..=k {
        if g[x] > 0.0 {
            maxdev = maxdev.max((res.y_equivalents[x] - x as f64).abs());
        }
    }
    assert!(
        maxdev < 1e-9,
        "self-equate must be identity, maxdev={maxdev}"
    );
    // includes x=0 whenever it has mass (the low-boundary interpolation)
    assert!(g[0] == 0.0 || (res.y_equivalents[0]).abs() < 1e-9);
}

// R2(a): closed-form moment methods recover the exact generating transform.
#[test]
fn equate_mean_linear_recover_transform() {
    let mut u = lcg(7);
    let k_x = 30usize;
    let x_scores: Vec<f64> = (0..5000)
        .map(|_| (15.0 + 6.0 * normal(&mut u)).round().clamp(0.0, k_x as f64))
        .collect();
    // mean: Y = X + 5 exactly
    let c = 5.0;
    let y_mean: Vec<f64> = x_scores.iter().map(|&x| x + c).collect();
    let rm = equate_eg(&x_scores, &y_mean, k_x, k_x + 5, EquateMethod::Mean).unwrap();
    assert!((rm.intercept - c).abs() < 1e-9 && (rm.slope - 1.0).abs() < 1e-12);
    assert!(rm
        .y_equivalents
        .iter()
        .enumerate()
        .all(|(x, &y)| (y - (x as f64 + c)).abs() < 1e-9));
    // linear: Y = 2*X + 3 exactly (integer affine, positive slope)
    let (a, b) = (2.0_f64, 3.0_f64);
    let k_y = (a * k_x as f64 + b) as usize;
    let y_lin: Vec<f64> = x_scores.iter().map(|&x| a * x + b).collect();
    let rl = equate_eg(&x_scores, &y_lin, k_x, k_y, EquateMethod::Linear).unwrap();
    assert!((rl.slope - a).abs() < 1e-9, "slope {} != {a}", rl.slope);
    assert!(
        (rl.intercept - b).abs() < 1e-9,
        "intercept {} != {b}",
        rl.intercept
    );
    assert!(rl
        .y_equivalents
        .iter()
        .enumerate()
        .all(|(x, &y)| (y - (a * x as f64 + b)).abs() < 1e-9));
}

// R3: with EQUAL anchor distributions (h_V1 = h_V2) and genuinely different X
// vs Y forms, both NEAT methods collapse to EG equipercentile of X onto Y.
// (Equal anchor marginals make the anchor cancel in chaining, and make the FE
// synthetic density equal each group's own marginal.)
#[test]
fn neat_collapses_to_eg_under_equal_anchors() {
    let mut u = lcg(3);
    let n = 6000usize;
    let (k_x, k_y, k_v) = (30usize, 40usize, 15usize);
    // identical anchor score vector for both populations => h_V1 == h_V2 exactly
    let anchor: Vec<f64> = (0..n)
        .map(|_| (7.0 + 3.0 * normal(&mut u)).round().clamp(0.0, k_v as f64))
        .collect();
    // different X and Y forms, correlated with the anchor but not equal to it
    let x_total: Vec<f64> = (0..n)
        .map(|i| {
            (anchor[i] * 1.4 + 4.0 + 4.0 * normal(&mut u))
                .round()
                .clamp(0.0, k_x as f64)
        })
        .collect();
    let y_total: Vec<f64> = (0..n)
        .map(|i| {
            (anchor[i] * 2.0 + 6.0 + 5.0 * normal(&mut u))
                .round()
                .clamp(0.0, k_y as f64)
        })
        .collect();

    let eg = equate_eg(&x_total, &y_total, k_x, k_y, EquateMethod::Equipercentile).unwrap();
    let ch = equate_neat(
        &x_total,
        &anchor,
        &y_total,
        &anchor,
        k_x,
        k_y,
        k_v,
        0.5,
        NeatMethod::ChainedEquipercentile,
    )
    .unwrap();
    let fe = equate_neat(
        &x_total,
        &anchor,
        &y_total,
        &anchor,
        k_x,
        k_y,
        k_v,
        0.5,
        NeatMethod::FrequencyEstimation,
    )
    .unwrap();
    let mut dmax_ch = 0.0_f64;
    let mut dmax_fe = 0.0_f64;
    for x in 0..=k_x {
        dmax_ch = dmax_ch.max((ch.y_equivalents[x] - eg.y_equivalents[x]).abs());
        dmax_fe = dmax_fe.max((fe.y_equivalents[x] - eg.y_equivalents[x]).abs());
    }
    assert!(
        dmax_ch < 1e-9,
        "chained must equal EG under equal anchors: {dmax_ch}"
    );
    assert!(
        dmax_fe < 1e-9,
        "FE must equal EG under equal anchors: {dmax_fe}"
    );
    // FE weight is inert here (h1==h2), so w1 in {0,1} agrees too
    for w1 in [0.0_f64, 1.0] {
        let fw = equate_neat(
            &x_total,
            &anchor,
            &y_total,
            &anchor,
            k_x,
            k_y,
            k_v,
            w1,
            NeatMethod::FrequencyEstimation,
        )
        .unwrap();
        let d = (0..=k_x)
            .map(|x| (fw.y_equivalents[x] - eg.y_equivalents[x]).abs())
            .fold(0.0, f64::max);
        assert!(
            d < 1e-9,
            "FE(w1={w1}) must match EG under equal anchors: {d}"
        );
    }
}

#[test]
fn method_and_error_paths() {
    assert_eq!(
        EquateMethod::parse("EquiPercentile"),
        Some(EquateMethod::Equipercentile)
    );
    assert_eq!(EquateMethod::parse("mean-mean"), None);
    assert_eq!(
        NeatMethod::parse("FE"),
        Some(NeatMethod::FrequencyEstimation)
    );
    assert!(equate_eg(&[], &[1.0], 5, 5, EquateMethod::Mean).is_err());
    assert!(equate_eg(&[6.0], &[1.0], 5, 5, EquateMethod::Mean).is_err()); // out of range
    assert!(equate_neat(
        &[1.0, 2.0],
        &[1.0],
        &[1.0],
        &[1.0],
        5,
        5,
        5,
        0.5,
        NeatMethod::FrequencyEstimation
    )
    .is_err());
    // out-of-range score (>= k+0.5) is now rejected (the old ±0.4 tolerance
    // on the already-rounded index silently binned it to a boundary cell)
    assert!(rel_freq(&[30.6], 30).is_err());
    assert!(rel_freq(&[-0.6], 30).is_err());
    // in-range fractional scores bin to the containing category interval:
    // 30.4 -> cat 30 ([29.5,30.5)), and -0.5 -> cat 0 ([-0.5,0.5))
    assert_eq!(rel_freq(&[30.4], 30).unwrap()[30], 1.0);
    assert_eq!(rel_freq(&[-0.5, 0.0, 1.0], 3).unwrap()[0], 2.0 / 3.0);
}

#[test]
fn equating_boundary_contracts_and_kernel_helpers() {
    assert_eq!(
        bandwidth_or_optimal(Some(0.75), &[0.5, 0.5], 0.5, 0.25, 1),
        0.75
    );
    assert!(bandwidth_or_optimal(None, &[0.5, 0.5], 0.5, 0.25, 1).is_finite());
    assert_eq!(NeatMethod::parse("not-a-method"), None);
    assert!(rel_freq(&[f64::NAN], 1).is_err());

    let g = [0.5, 0.5];
    let f = cdf(&g);
    assert!(cdf(&[]).is_empty());
    assert_eq!(perc_rank(&g, &f, 1, -1.0), 0.0);
    assert_eq!(perc_rank(&g, &f, 1, 2.0), 100.0);
    assert_eq!(perc_rank_inv(&f, 1, 0.0), -0.5);
    assert_eq!(perc_rank_inv(&f, 1, 100.0), 1.5);

    assert!(bivariate(&[0.0], &[0.0, 1.0], 1, 1).is_err());
    assert!(bivariate(&[], &[], 1, 1).is_err());
    assert!(bivariate(&[f64::NAN], &[0.0], 1, 1).is_err());
    assert!(bivariate(&[2.0], &[0.0], 1, 1).is_err());

    assert!(equate_eg(&[0.0], &[0.0], 0, 1, EquateMethod::Mean).is_err());
    assert!(equate_eg(&[0.0, 0.0], &[0.0, 1.0], 1, 1, EquateMethod::Linear).is_err());
    assert!(equate_neat(
        &[0.0],
        &[0.0],
        &[0.0],
        &[0.0],
        0,
        1,
        1,
        0.5,
        NeatMethod::ChainedEquipercentile,
    )
    .is_err());
    assert!(equate_neat(
        &[0.0, 1.0],
        &[0.0, 1.0],
        &[0.0, 1.0],
        &[0.0, 1.0],
        1,
        1,
        1,
        f64::NAN,
        NeatMethod::FrequencyEstimation,
    )
    .is_err());
    assert_eq!(NeatLinearMethod::parse("not-a-method"), None);
    assert_eq!(AnchorKind::parse("not-an-anchor"), None);
    let total = [0.0, 1.0];
    let anchor = [0.0, 1.0];
    let oversized_fe = std::panic::catch_unwind(|| {
        equate_neat(
            &total,
            &anchor,
            &total,
            &anchor,
            MAX_EQUATING_SCORE_POINTS,
            1,
            MAX_EQUATING_SCORE_POINTS,
            0.5,
            NeatMethod::FrequencyEstimation,
        )
    });
    assert!(
        oversized_fe.is_ok(),
        "oversized FE bivariate table must return an error instead of panicking"
    );
    // Reads the core-returned error path and kills mutations that keep only
    // per-axis caps while allowing multi-GB bivariate table allocation.
    assert!(oversized_fe.unwrap().is_err());
    assert!(equate_neat_linear(
        &total,
        &anchor,
        &total,
        &anchor,
        0,
        1,
        0.5,
        NeatLinearMethod::Tucker,
        AnchorKind::Internal,
    )
    .is_err());
    assert!(equate_neat_linear(
        &total[..1],
        &anchor,
        &total,
        &anchor,
        1,
        1,
        0.5,
        NeatLinearMethod::Tucker,
        AnchorKind::Internal,
    )
    .is_err());
    assert!(equate_neat_linear(
        &[],
        &[],
        &total,
        &anchor,
        1,
        1,
        0.5,
        NeatLinearMethod::Tucker,
        AnchorKind::Internal,
    )
    .is_err());
    assert!(equate_neat_linear(
        &[f64::INFINITY, 1.0],
        &anchor,
        &total,
        &anchor,
        1,
        1,
        0.5,
        NeatLinearMethod::Tucker,
        AnchorKind::Internal,
    )
    .is_err());
    assert!(equate_neat_linear(
        &[0.0, 1000.0],
        &anchor,
        &total,
        &anchor,
        1,
        1,
        0.5,
        NeatLinearMethod::Tucker,
        AnchorKind::Internal,
    )
    .is_err());
    assert!(equate_neat_linear(
        &[-0.49, 1.49],
        &anchor,
        &total,
        &anchor,
        1,
        1,
        0.5,
        NeatLinearMethod::Tucker,
        AnchorKind::Internal,
    )
    .is_err());
    assert!(equate_neat_linear(
        &total,
        &anchor,
        &[0.0, 1000.0],
        &anchor,
        1,
        1,
        0.5,
        NeatLinearMethod::Tucker,
        AnchorKind::Internal,
    )
    .is_err());
    assert!(equate_neat_linear(
        &total,
        &anchor,
        &[-0.49, 1.49],
        &anchor,
        1,
        1,
        0.5,
        NeatLinearMethod::Tucker,
        AnchorKind::Internal,
    )
    .is_err());
    assert!(equate_neat_linear(
        &total,
        &[1.0, 0.0],
        &total,
        &anchor,
        1,
        1,
        0.5,
        NeatLinearMethod::LevineObserved,
        AnchorKind::Internal,
    )
    .is_err());
    let huge_anchor = equate_neat_linear(
        &total,
        &[1.0e308, -1.0e308],
        &total,
        &[1.0e308, -1.0e308],
        1,
        1,
        0.5,
        NeatLinearMethod::Tucker,
        AnchorKind::Internal,
    );
    // Reads the core-returned result path and kills mutations that allow finite
    // but overflowing anchor moments to produce NaN conversion tables.
    assert!(huge_anchor.is_err());
    let oversized = std::panic::catch_unwind(|| {
        equate_neat_linear(
            &total,
            &anchor,
            &total,
            &anchor,
            usize::MAX,
            1,
            0.5,
            NeatLinearMethod::Tucker,
            AnchorKind::Internal,
        )
    });
    assert!(
        oversized.is_ok(),
        "oversized k_x must return an error instead of panicking"
    );
    // Reads the core-returned error path and kills mutations that let
    // NEAT-linear allocate conversion tables from unchecked k_x.
    assert!(oversized.unwrap().is_err());
    assert!(equate_neat_linear(
        &[1.0, 1.0],
        &anchor,
        &total,
        &anchor,
        1,
        1,
        0.5,
        NeatLinearMethod::Tucker,
        AnchorKind::Internal,
    )
    .is_err());

    assert_eq!(quantile_type7(&[2.0], 0.3), 2.0);
    assert_eq!(quantile_type7(&[1.0, 2.0], 1.0), 2.0);
    assert!(analytic_see(&total, &total, 1, 1, EquateMethod::Mean, 1.0).is_err());
    assert!(analytic_see(&[0.0, 0.0], &total, 1, 1, EquateMethod::Mean, 0.95).is_err());

    assert!(loglinear_smooth(&[1.0], 1).is_err());
    assert!(loglinear_smooth(&[1.0, 1.0], 0).is_err());
    assert!(loglinear_smooth(&[1.0, f64::NAN], 1).is_err());
    assert!(loglinear_smooth(&[0.0, 0.0], 1).is_err());
    let exact = loglinear_smooth(&[1.0, 1.0], 1).unwrap();
    assert!(exact.converged);
    assert_eq!(exact.termination_reason, "gradient_tolerance");
    let rank_deficient = ortho_poly_design(0, 1);
    assert_eq!(rank_deficient.len(), 1);
    let mut zero_mass = [0.0, 0.0];
    renormalize(&mut zero_mass);
    assert_eq!(zero_mass, [0.0, 0.0]);

    assert_eq!(
        Continuization::parse("uniform"),
        Some(Continuization::Uniform)
    );
    assert_eq!(
        Continuization::parse("kernel"),
        Some(Continuization::Gaussian)
    );
    assert_eq!(Continuization::parse("not-a-kernel"), None);
    assert_eq!(expanded_upper_bandwidth(3.0, 3.0), 6.0);
    assert_eq!(expanded_upper_bandwidth(2.0, 3.0), 3.0);
    assert!(validate_optional_bandwidth(None, "bandwidth").is_ok());
    assert!(validate_optional_bandwidth(Some(0.5), "bandwidth").is_ok());
    assert!(validate_optional_bandwidth(Some(f64::NAN), "bandwidth").is_err());

    // The Gaussian has full support, so exact endpoint probabilities force the
    // inverse-CDF bracketing guards to expand beyond the nominal score range.
    let left = kernel_inv(&g, 0.5, 0.25, 0.5, 0.0, 1);
    let right = kernel_inv(&g, 0.5, 0.25, 0.5, 1.0, 1);
    assert!(left < -0.5 && right > 1.5);
    let derivative_free = kernel_inv(&[0.0, 0.0], 0.5, 0.25, 0.5, 0.5, 1);
    assert!(derivative_free.is_finite());

    // Exercise bandwidth refinement at both sharp and strongly multimodal
    // densities, plus a smooth density where the golden-section candidate wins.
    for mut density in [
        vec![1.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.45, 0.0, 0.05, 0.0, 0.5],
        vec![0.05, 0.2, 0.5, 0.2, 0.05],
    ] {
        renormalize(&mut density);
        let (mu, sd) = moments(&density);
        let h = optimal_bandwidth(&density, mu, sd * sd, density.len() - 1);
        assert!(h.is_finite() && h > 0.0);
    }

    assert!(equate_eg_ext(
        &total,
        &total,
        0,
        1,
        ext(Continuization::Uniform, None, None, None, None),
    )
    .is_err());
    assert!(equate_eg_ext(
        &total,
        &total,
        1,
        1,
        ext(Continuization::Gaussian, None, None, Some(0.0), Some(0.5)),
    )
    .is_err());
    assert!(equate_eg_ext(
        &total,
        &total,
        1,
        1,
        ext(
            Continuization::Gaussian,
            None,
            None,
            Some(0.5),
            Some(f64::NAN)
        ),
    )
    .is_err());
    assert!(equate_eg_ext(
        &[0.0, 0.0],
        &[0.0, 0.0],
        1,
        1,
        ext(Continuization::Gaussian, None, None, Some(0.5), Some(0.5)),
    )
    .is_err());
}

// FE requires the two groups to share anchor support; fully disjoint anchors
// would otherwise silently collapse the synthetic density (finding: garbage
// conversion table returned as Ok). Chained composition has no such
// requirement and still returns a result.
#[test]
fn fe_rejects_disjoint_anchor_support() {
    let x_total = vec![1.0, 2.0, 3.0, 2.0, 1.0, 3.0];
    let x_anchor = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]; // support {0,1}
    let y_total = vec![2.0, 3.0, 1.0, 2.0, 3.0, 1.0];
    let y_anchor = vec![4.0, 5.0, 4.0, 5.0, 4.0, 5.0]; // support {4,5}
    assert!(equate_neat(
        &x_total,
        &x_anchor,
        &y_total,
        &y_anchor,
        5,
        5,
        5,
        0.5,
        NeatMethod::FrequencyEstimation,
    )
    .is_err());
    // also at the boundary weight w1=0 (the all-zero-density degenerate case)
    assert!(equate_neat(
        &x_total,
        &x_anchor,
        &y_total,
        &y_anchor,
        5,
        5,
        5,
        0.0,
        NeatMethod::FrequencyEstimation,
    )
    .is_err());
    assert!(equate_neat(
        &x_total,
        &x_anchor,
        &y_total,
        &y_anchor,
        5,
        5,
        5,
        0.5,
        NeatMethod::ChainedEquipercentile,
    )
    .is_ok());
}

// 2PL population number-correct density on a GH grid, via Lord-Wingersky.
fn pop_density(a: &[f64], b: &[f64], nodes: &[f64], weights: &[f64]) -> Vec<f64> {
    let n_items = a.len();
    let n_nodes = nodes.len();
    let mut probs = vec![0.0_f64; n_items * n_nodes];
    for i in 0..n_items {
        for (t, &th) in nodes.iter().enumerate() {
            probs[i * n_nodes + t] = 1.0 / (1.0 + (-(a[i] * th + b[i])).exp());
        }
    }
    let f = crate::scoring::lord_wingersky(&probs, n_items, n_nodes);
    (0..=n_items)
        .map(|s| (0..n_nodes).map(|t| weights[t] * f[s * n_nodes + t]).sum())
        .collect()
}

fn interior_bias_rmse(
    a_x: &[f64],
    b_x: &[f64],
    a_y: &[f64],
    b_y: &[f64],
    n: usize,
    reps: usize,
    seed: u64,
) -> (f64, f64) {
    let (k_x, k_y) = (a_x.len(), a_y.len());
    let (nodes, weights) = crate::quadrature::gh_rule(41).unwrap();
    // deterministic population reference e_Y*(x)
    let gx_pop = pop_density(a_x, b_x, nodes, weights);
    let gy_pop = pop_density(a_y, b_y, nodes, weights);
    let e_ref = equipercentile(&gx_pop, &gy_pop, k_x, k_y);
    let mut u = lcg(seed);
    let mut sum = vec![0.0_f64; k_x + 1];
    let mut sum2 = vec![0.0_f64; k_x + 1];
    let sim = |u: &mut dyn FnMut() -> f64, a: &[f64], b: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|_| {
                let th = {
                    let u1 = u().max(1e-12);
                    let u2 = u();
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                };
                a.iter()
                    .zip(b)
                    .filter(|(&ai, &bi)| u() < 1.0 / (1.0 + (-(ai * th + bi)).exp()))
                    .count() as f64
            })
            .collect()
    };
    for _ in 0..reps {
        let xs = sim(&mut u, a_x, b_x);
        let ys = sim(&mut u, a_y, b_y);
        let est = equate_eg(&xs, &ys, k_x, k_y, EquateMethod::Equipercentile).unwrap();
        for x in 0..=k_x {
            let d = est.y_equivalents[x] - e_ref[x];
            sum[x] += d;
            sum2[x] += d * d;
        }
    }
    // trim the outer ~5% of the score range where zero-cell sampling dominates
    let lo = (k_x as f64 * 0.05).ceil() as usize;
    let hi = k_x - lo;
    let mut max_bias = 0.0_f64;
    let mut rmse_acc = 0.0_f64;
    let mut cnt = 0usize;
    for x in lo..=hi {
        max_bias = max_bias.max((sum[x] / reps as f64).abs());
        rmse_acc += sum2[x] / reps as f64;
        cnt += 1;
    }
    (max_bias, (rmse_acc / cnt as f64).sqrt())
}
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn equate_monte_carlo_500() {
    // distinct 2PL forms X (30 items) and Y (40 items)
    let k_x = 30usize;
    let k_y = 40usize;
    let a_x: Vec<f64> = (0..k_x)
        .map(|i| 0.8 + 0.5 * ((i % 5) as f64 / 4.0))
        .collect();
    let b_x: Vec<f64> = (0..k_x)
        .map(|i| 1.5 - 3.0 * i as f64 / (k_x - 1) as f64)
        .collect();
    let a_y: Vec<f64> = (0..k_y)
        .map(|i| 0.9 + 0.4 * ((i % 4) as f64 / 3.0))
        .collect();
    let b_y: Vec<f64> = (0..k_y)
        .map(|i| 1.8 - 3.6 * i as f64 / (k_y - 1) as f64)
        .collect();

    let reps = 500usize;
    let (bias1, rmse1) = interior_bias_rmse(&a_x, &b_x, &a_y, &b_y, 1000, reps, 4001);
    let (bias4, rmse4) = interior_bias_rmse(&a_x, &b_x, &a_y, &b_y, 4000, reps, 7001);
    let ratio = rmse1 / rmse4;
    println!(
        "[equate 500] N=1000: max|bias|={bias1:.4} RMSE={rmse1:.4}  \
         N=4000: max|bias|={bias4:.4} RMSE={rmse4:.4}  RMSE ratio={ratio:.3} (expect ~2)"
    );
    // the empirical equipercentile converges to the population equipercentile
    // of the same Lord-Wingersky densities (that population transform IS the
    // estimand; R1/R2/R3 supply the independent identification):
    assert!(
        bias1 < 0.15 && bias4 < 0.08,
        "bias should be small and shrink: {bias1}, {bias4}"
    );
    assert!(
        (1.6..=2.4).contains(&ratio),
        "RMSE should shrink ~1/sqrt(N): ratio={ratio}"
    );
}

fn ext(
    cont: Continuization,
    sx: Option<usize>,
    sy: Option<usize>,
    hx: Option<f64>,
    hy: Option<f64>,
) -> EgSmoothOptions {
    EgSmoothOptions {
        continuization: cont,
        smooth_degree_x: sx,
        smooth_degree_y: sy,
        bandwidth_x: hx,
        bandwidth_y: hy,
    }
}

// Anchor 1: uniform-kernel ext == existing equipercentile, bit-exact.
#[test]
fn ext_uniform_matches_equipercentile() {
    let mut u = lcg(21);
    let (n, kx, ky) = (3000usize, 30usize, 30usize);
    let xs: Vec<f64> = (0..n)
        .map(|_| (15.0 + 6.0 * normal(&mut u)).round().clamp(0.0, kx as f64))
        .collect();
    let ys: Vec<f64> = (0..n)
        .map(|_| (14.0 + 7.0 * normal(&mut u)).round().clamp(0.0, ky as f64))
        .collect();
    let base = equate_eg(&xs, &ys, kx, ky, EquateMethod::Equipercentile).unwrap();
    let e = equate_eg_ext(
        &xs,
        &ys,
        kx,
        ky,
        ext(Continuization::Uniform, None, None, None, None),
    )
    .unwrap();
    let d = (0..=kx)
        .map(|x| (base.y_equivalents[x] - e.y_equivalents[x]).abs())
        .fold(0.0, f64::max);
    assert!(
        d < 1e-12,
        "uniform-kernel ext must equal equipercentile: {d}"
    );
}

// Anchors 2 & 3: log-linear presmoothing preserves the first T sample moments
// exactly (on the u=x/k scale) and, saturated at T=k, reproduces rel_freq.
#[test]
fn loglinear_preserves_moments_and_saturates() {
    let mut u = lcg(5);
    let k = 40usize;
    let scores: Vec<f64> = (0..5000)
        .map(|_| (20.0 + 7.0 * normal(&mut u)).round().clamp(0.0, k as f64))
        .collect();
    let g = rel_freq(&scores, k).unwrap();
    let n = scores.len() as f64;
    let counts: Vec<f64> = g.iter().map(|&p| p * n).collect();
    let fit = loglinear_smooth(&counts, 4).unwrap();
    assert!(fit.converged);
    assert!((fit.probs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(fit.probs.iter().all(|&p| p >= 0.0));
    for (j, &fm) in fit.moments.iter().enumerate() {
        let order = (j + 1) as i32;
        let sm: f64 = (0..=k)
            .map(|x| (x as f64 / k as f64).powi(order) * g[x])
            .sum();
        assert!(
            (fm - sm).abs() < 1e-8,
            "moment {order} not preserved: {fm} vs {sm}"
        );
    }
    let sat = loglinear_smooth(&counts, k).unwrap();
    let d = (0..=k)
        .map(|x| (sat.probs[x] - g[x]).abs())
        .fold(0.0, f64::max);
    assert!(d < 1e-9, "saturated loglinear must reproduce rel_freq: {d}");
}

#[test]
fn equating_rejects_nonconverged_presmoothing() {
    let counts = [0usize, 1564, 426, 0, 1008, 0, 0];
    let scores: Vec<f64> = counts
        .iter()
        .enumerate()
        .flat_map(|(score, &count)| std::iter::repeat_n(score as f64, count))
        .collect();
    let fit = loglinear_smooth(
        &counts.iter().map(|&count| count as f64).collect::<Vec<_>>(),
        5,
    )
    .unwrap();
    assert!(
        !fit.converged,
        "fixture must exercise the non-converged path"
    );
    assert_eq!(fit.termination_reason, "line_search_stalled");
    assert!(fit.final_gradient_max > fit.gradient_tolerance);

    let err = equate_eg_ext(
        &scores,
        &scores,
        6,
        6,
        ext(Continuization::Uniform, Some(5), Some(5), None, None),
    )
    .unwrap_err();
    assert!(err.contains("did not converge"), "unexpected error: {err}");
}

// Anchors 4 & 6: Gaussian-kernel self-equate is the identity (F_h == G_h), and
// the continuized density preserves the discrete mean and variance.
#[test]
fn kernel_self_equate_and_mean_var() {
    let mut u = lcg(9);
    let k = 30usize;
    let xs: Vec<f64> = (0..4000)
        .map(|_| (15.0 + 6.0 * normal(&mut u)).round().clamp(0.0, k as f64))
        .collect();
    let res = equate_eg_ext(
        &xs,
        &xs,
        k,
        k,
        ext(Continuization::Gaussian, None, None, Some(0.6), Some(0.6)),
    )
    .unwrap();
    let g = rel_freq(&xs, k).unwrap();
    let mut dmax = 0.0_f64;
    for x in 0..=k {
        if g[x] > 0.0 {
            dmax = dmax.max((res.y_equivalents[x] - x as f64).abs());
        }
    }
    // exact in exact arithmetic (F_h == G_h); the ~1e-8 residual is the
    // erfc approximation (|err| < 1.2e-7) through the numeric inverse
    assert!(dmax < 1e-6, "kernel self-equate must be identity: {dmax}");
    assert_eq!(res.h_x, 0.6);
    let (mu, sd) = moments(&g);
    let sig2 = sd * sd;
    let h = 0.8;
    let (lo, hi, steps) = (-6.0_f64, k as f64 + 6.0, 20000usize);
    let dx = (hi - lo) / steps as f64;
    let (mut m0, mut m1, mut m2) = (0.0_f64, 0.0, 0.0);
    for i in 0..steps {
        let x = lo + (i as f64 + 0.5) * dx;
        let fh = kernel_pdf(&g, mu, sig2, h, x);
        m0 += fh * dx;
        m1 += x * fh * dx;
        m2 += x * x * fh * dx;
    }
    let mean = m1 / m0;
    let var = m2 / m0 - mean * mean;
    assert!((mean - mu).abs() < 1e-3, "kernel mean {mean} != {mu}");
    assert!(
        (var - sig2).abs() < 1e-2 * sig2.max(1.0),
        "kernel var {var} != {sig2}"
    );
}

// Anchor 5: a very large bandwidth drives Gaussian-kernel equating to LINEAR.
#[test]
fn kernel_large_bandwidth_is_linear() {
    let mut u = lcg(13);
    let (kx, ky) = (30usize, 40usize);
    let xs: Vec<f64> = (0..4000)
        .map(|_| (15.0 + 6.0 * normal(&mut u)).round().clamp(0.0, kx as f64))
        .collect();
    let ys: Vec<f64> = (0..4000)
        .map(|_| (22.0 + 8.0 * normal(&mut u)).round().clamp(0.0, ky as f64))
        .collect();
    let lin = equate_eg(&xs, &ys, kx, ky, EquateMethod::Linear).unwrap();
    let ker = equate_eg_ext(
        &xs,
        &ys,
        kx,
        ky,
        ext(Continuization::Gaussian, None, None, Some(1e6), Some(1e6)),
    )
    .unwrap();
    let d = (0..=kx)
        .map(|x| (lin.y_equivalents[x] - ker.y_equivalents[x]).abs())
        .fold(0.0, f64::max);
    assert!(d < 1e-4, "large-h kernel must match linear: {d}");
}

// Anchor 8: presmoothed self-equate is still the identity.
#[test]
fn presmoothed_self_equate_is_identity() {
    let mut u = lcg(17);
    let k = 40usize;
    let xs: Vec<f64> = (0..3000)
        .map(|_| (20.0 + 7.0 * normal(&mut u)).round().clamp(0.0, k as f64))
        .collect();
    let res = equate_eg_ext(
        &xs,
        &xs,
        k,
        k,
        ext(Continuization::Uniform, Some(5), Some(5), None, None),
    )
    .unwrap();
    let g = density(&xs, k, Some(5)).unwrap();
    let mut dmax = 0.0_f64;
    for x in 0..=k {
        if g[x] > 1e-12 {
            dmax = dmax.max((res.y_equivalents[x] - x as f64).abs());
        }
    }
    assert!(
        dmax < 1e-8,
        "presmoothed self-equate must be identity: {dmax}"
    );
}

// Fix guard: on a non-unimodal penalty (bimodal density) the golden-section
// refinement can land in a worse cell, so optimal_bandwidth must fall back to
// the grid best rather than ship it.
#[test]
fn optimal_bandwidth_never_worse_than_grid() {
    let k = 40usize;
    let mut r = vec![0.0_f64; k + 1];
    for j in 0..=k {
        let d1 = (j as f64 - 8.0) / 2.0;
        let d2 = (j as f64 - 32.0) / 2.0;
        r[j] = (-0.5 * d1 * d1).exp() + (-0.5 * d2 * d2).exp();
    }
    let s: f64 = r.iter().sum();
    for v in r.iter_mut() {
        *v /= s;
    }
    let (mu, sd) = moments(&r);
    let sig2 = sd * sd;
    let h = optimal_bandwidth(&r, mu, sig2, k);
    assert!(h.is_finite() && h > 0.0);
    let pen_h = kernel_penalty(&r, mu, sig2, h, k);
    let grid_best = (0..=40)
        .map(|i| kernel_penalty(&r, mu, sig2, 0.1 + (3.0 - 0.1) * i as f64 / 40.0, k))
        .fold(f64::INFINITY, f64::min);
    assert!(
        pen_h <= grid_best + 1e-12,
        "optimal_bandwidth worse than grid: {pen_h} vs {grid_best}"
    );
}

// Gaussian-kernel MC with a FIXED bandwidth shared by the population reference
// and the per-rep estimator, so the assertion measures density-sampling error
// alone (penalty-selected h would inject selection noise).
fn kernel_bias_rmse(
    a_x: &[f64],
    b_x: &[f64],
    a_y: &[f64],
    b_y: &[f64],
    n: usize,
    reps: usize,
    seed: u64,
    h: f64,
) -> (f64, f64) {
    let (k_x, k_y) = (a_x.len(), a_y.len());
    let (nodes, weights) = crate::quadrature::gh_rule(41).unwrap();
    let gx_pop = pop_density(a_x, b_x, nodes, weights);
    let gy_pop = pop_density(a_y, b_y, nodes, weights);
    let (mux, sdx) = moments(&gx_pop);
    let (muy, sdy) = moments(&gy_pop);
    let e_ref = kernel_equate(
        &gx_pop,
        &gy_pop,
        mux,
        sdx * sdx,
        muy,
        sdy * sdy,
        k_x,
        k_y,
        h,
        h,
    );
    let mut u = lcg(seed);
    let sim = |u: &mut dyn FnMut() -> f64, a: &[f64], b: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|_| {
                let th = {
                    let u1 = u().max(1e-12);
                    let u2 = u();
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                };
                a.iter()
                    .zip(b)
                    .filter(|(&ai, &bi)| u() < 1.0 / (1.0 + (-(ai * th + bi)).exp()))
                    .count() as f64
            })
            .collect()
    };
    let mut sum = vec![0.0_f64; k_x + 1];
    let mut sum2 = vec![0.0_f64; k_x + 1];
    for _ in 0..reps {
        let xs = sim(&mut u, a_x, b_x);
        let ys = sim(&mut u, a_y, b_y);
        let est = equate_eg_ext(
            &xs,
            &ys,
            k_x,
            k_y,
            ext(Continuization::Gaussian, None, None, Some(h), Some(h)),
        )
        .unwrap();
        for x in 0..=k_x {
            let d = est.y_equivalents[x] - e_ref[x];
            sum[x] += d;
            sum2[x] += d * d;
        }
    }
    let lo = (k_x as f64 * 0.05).ceil() as usize;
    let hi = k_x - lo;
    let mut max_bias = 0.0_f64;
    let mut rmse_acc = 0.0_f64;
    let mut cnt = 0usize;
    for x in lo..=hi {
        max_bias = max_bias.max((sum[x] / reps as f64).abs());
        rmse_acc += sum2[x] / reps as f64;
        cnt += 1;
    }
    (max_bias, (rmse_acc / cnt as f64).sqrt())
}
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn kernel_equate_monte_carlo_500() {
    let k_x = 30usize;
    let k_y = 40usize;
    let a_x: Vec<f64> = (0..k_x)
        .map(|i| 0.8 + 0.5 * ((i % 5) as f64 / 4.0))
        .collect();
    let b_x: Vec<f64> = (0..k_x)
        .map(|i| 1.5 - 3.0 * i as f64 / (k_x - 1) as f64)
        .collect();
    let a_y: Vec<f64> = (0..k_y)
        .map(|i| 0.9 + 0.4 * ((i % 4) as f64 / 3.0))
        .collect();
    let b_y: Vec<f64> = (0..k_y)
        .map(|i| 1.8 - 3.6 * i as f64 / (k_y - 1) as f64)
        .collect();
    let reps = 500usize;
    let h = 0.6_f64;
    let (bias1, rmse1) = kernel_bias_rmse(&a_x, &b_x, &a_y, &b_y, 1000, reps, 5001, h);
    let (bias4, rmse4) = kernel_bias_rmse(&a_x, &b_x, &a_y, &b_y, 4000, reps, 8001, h);
    let ratio = rmse1 / rmse4;
    println!(
        "[kernel equate 500] h={h} N=1000: max|bias|={bias1:.4} RMSE={rmse1:.4}  \
         N=4000: max|bias|={bias4:.4} RMSE={rmse4:.4}  RMSE ratio={ratio:.3} (expect ~2)"
    );
    assert!(
        bias1 < 0.15 && bias4 < 0.08,
        "bias should be small and shrink: {bias1}, {bias4}"
    );
    assert!(
        (1.6..=2.4).contains(&ratio),
        "RMSE should shrink ~1/sqrt(N): {ratio}"
    );
}

// Primary anchor: with equal anchor moments (a shared anchor vector) every
// Tucker/Levine variant collapses to EG linear equating of X onto Y, for any
// w1 and anchor kind.
#[test]
fn neat_linear_collapses_to_eg_linear() {
    let (kx, ky) = (30usize, 40usize);
    let mut u = lcg(41);
    let n = 4000usize;
    // a shared anchor vector (equal anchor moments by construction) that is
    // genuinely correlated with both totals (so Levine's covariance is positive)
    let anchor: Vec<f64> = (0..n)
        .map(|_| (7.0 + 3.0 * normal(&mut u)).round().clamp(0.0, 15.0))
        .collect();
    let x_total: Vec<f64> = anchor
        .iter()
        .map(|&v| {
            (1.5 * v + 4.0 + 3.0 * normal(&mut u))
                .round()
                .clamp(0.0, kx as f64)
        })
        .collect();
    let y_total: Vec<f64> = anchor
        .iter()
        .map(|&v| {
            (1.8 * v + 6.0 + 4.0 * normal(&mut u))
                .round()
                .clamp(0.0, ky as f64)
        })
        .collect();
    let eg = equate_eg(&x_total, &y_total, kx, ky, EquateMethod::Linear).unwrap();
    for m in [NeatLinearMethod::Tucker, NeatLinearMethod::LevineObserved] {
        for ak in [AnchorKind::Internal, AnchorKind::External] {
            for w1 in [0.0_f64, 0.5, 1.0] {
                let r = equate_neat_linear(&x_total, &anchor, &y_total, &anchor, kx, ky, w1, m, ak)
                    .unwrap();
                assert!(
                    (r.slope - eg.slope).abs() < 1e-9 && (r.intercept - eg.intercept).abs() < 1e-9,
                    "collapse {m:?}/{ak:?}/w1={w1}: slope {} vs {}, int {} vs {}",
                    r.slope,
                    eg.slope,
                    r.intercept,
                    eg.intercept
                );
                let d = (0..=kx)
                    .map(|x| (r.y_equivalents[x] - eg.y_equivalents[x]).abs())
                    .fold(0.0, f64::max);
                assert!(d < 1e-9, "table mismatch: {d}");
            }
        }
    }
}

// Pins the internal-vs-external Levine gamma (the crux) against a NumPy oracle
// (N-denominator moments): the three gamma branches give three distinct
// slope/intercept pairs.
#[test]
fn neat_linear_gamma_hand_computed() {
    let x1 = [3.0, 5., 7., 9., 4., 6., 8., 2.];
    let v1 = [1.0, 2., 2., 3., 1., 2., 3., 1.];
    let y2 = [2.0, 5., 8., 11., 4., 7., 10., 1.];
    let v2 = [2.0, 4., 4., 6., 3., 5., 6., 2.];
    let (kx, ky, w1) = (11usize, 11usize, 0.5_f64);
    let tk = equate_neat_linear(
        &x1,
        &v1,
        &y2,
        &v2,
        kx,
        ky,
        w1,
        NeatLinearMethod::Tucker,
        AnchorKind::Internal,
    )
    .unwrap();
    assert!(
        (tk.slope - 0.8006819908).abs() < 1e-8 && (tk.intercept + 3.0616870634).abs() < 1e-8,
        "tucker {} {}",
        tk.slope,
        tk.intercept
    );
    let li = equate_neat_linear(
        &x1,
        &v1,
        &y2,
        &v2,
        kx,
        ky,
        w1,
        NeatLinearMethod::LevineObserved,
        AnchorKind::Internal,
    )
    .unwrap();
    assert!(
        (li.slope - 0.7403094687).abs() < 1e-8 && (li.intercept + 3.0252464118).abs() < 1e-8,
        "levine-int {} {}",
        li.slope,
        li.intercept
    );
    let le = equate_neat_linear(
        &x1,
        &v1,
        &y2,
        &v2,
        kx,
        ky,
        w1,
        NeatLinearMethod::LevineObserved,
        AnchorKind::External,
    )
    .unwrap();
    assert!(
        (le.slope - 0.7550256824).abs() < 1e-8 && (le.intercept + 3.017543311).abs() < 1e-8,
        "levine-ext {} {}",
        le.slope,
        le.intercept
    );
    // Tucker ignores the anchor kind
    let tk2 = equate_neat_linear(
        &x1,
        &v1,
        &y2,
        &v2,
        kx,
        ky,
        w1,
        NeatLinearMethod::Tucker,
        AnchorKind::External,
    )
    .unwrap();
    assert_eq!(tk.slope, tk2.slope);
    assert_eq!(
        NeatLinearMethod::parse("levine"),
        Some(NeatLinearMethod::LevineObserved)
    );
    assert_eq!(AnchorKind::parse("ext"), Some(AnchorKind::External));
    // error paths: bad w1, constant anchor (zero variance), Levine on a zero-cov anchor
    assert!(equate_neat_linear(
        &x1,
        &v1,
        &y2,
        &v2,
        kx,
        ky,
        1.5,
        NeatLinearMethod::Tucker,
        AnchorKind::Internal
    )
    .is_err());
    let const_v = [2.0_f64; 8];
    assert!(equate_neat_linear(
        &x1,
        &const_v,
        &y2,
        &v2,
        kx,
        ky,
        w1,
        NeatLinearMethod::Tucker,
        AnchorKind::Internal
    )
    .is_err());
}

// Common-regression generative model (satisfies the Tucker assumption); the
// estimator's equated table converges to the large-N reference at ~1/sqrt(N).
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn neat_linear_monte_carlo_500() {
    let (kt_x, kt_y, kv) = (40usize, 45usize, 15usize);
    let (sdv, beta, tau) = (2.5_f64, 1.2_f64, 3.0_f64);
    let gen = |u: &mut dyn FnMut() -> f64,
               n: usize,
               muv: f64,
               alpha: f64,
               kt: usize|
     -> (Vec<f64>, Vec<f64>) {
        let nd = |u: &mut dyn FnMut() -> f64| {
            let u1 = u().max(1e-12);
            let u2 = u();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let mut tot = vec![0.0_f64; n];
        let mut anc = vec![0.0_f64; n];
        for i in 0..n {
            let v = muv + sdv * nd(u);
            let t = alpha + beta * v + tau * nd(u);
            anc[i] = v.round().clamp(0.0, kv as f64);
            tot[i] = t.round().clamp(0.0, kt as f64);
        }
        (tot, anc)
    };
    // reference from a large calibration draw through the same sampler+rounding
    let mut ur = lcg(9100);
    let n_reference = 2_000_000usize;
    let (rx, rxa) = gen(&mut ur, n_reference, 6.0, 5.0, kt_x);
    let (ry, rya) = gen(&mut ur, n_reference, 9.0, 8.0, kt_y);
    let e_ref = equate_neat_linear(
        &rx,
        &rxa,
        &ry,
        &rya,
        kt_x,
        kt_y,
        0.5,
        NeatLinearMethod::Tucker,
        AnchorKind::Internal,
    )
    .unwrap();
    let bias_rmse = |n: usize, seed: u64| -> (f64, f64) {
        let mut u = lcg(seed);
        let reps = 500usize;
        let mut sum = vec![0.0_f64; kt_x + 1];
        let mut sum2 = vec![0.0_f64; kt_x + 1];
        for _ in 0..reps {
            let (xt, xa) = gen(&mut u, n, 6.0, 5.0, kt_x);
            let (yt, ya) = gen(&mut u, n, 9.0, 8.0, kt_y);
            let est = equate_neat_linear(
                &xt,
                &xa,
                &yt,
                &ya,
                kt_x,
                kt_y,
                0.5,
                NeatLinearMethod::Tucker,
                AnchorKind::Internal,
            )
            .unwrap();
            for x in 0..=kt_x {
                let d = est.y_equivalents[x] - e_ref.y_equivalents[x];
                sum[x] += d;
                sum2[x] += d * d;
            }
        }
        let lo = (kt_x as f64 * 0.05).ceil() as usize;
        let hi = kt_x - lo;
        let (mut mb, mut ra, mut c) = (0.0_f64, 0.0_f64, 0usize);
        for x in lo..=hi {
            mb = mb.max((sum[x] / reps as f64).abs());
            ra += sum2[x] / reps as f64;
            c += 1;
        }
        (mb, (ra / c as f64).sqrt())
    };
    let (b1, r1) = bias_rmse(1000, 111);
    let (b4, r4) = bias_rmse(4000, 222);
    let ratio = r1 / r4;
    println!("[neat-linear 500] N=1000: max|bias|={b1:.4} RMSE={r1:.4}  N=4000: max|bias|={b4:.4} RMSE={r4:.4}  ratio={ratio:.3}");
    assert!(
        b1 < 0.20 && b4 < 0.10,
        "bias should be small and shrink: {b1}, {b4}"
    );
    assert!(
        (1.6..=2.4).contains(&ratio),
        "RMSE should shrink ~1/sqrt(N): {ratio}"
    );
}

// helper: two near-normal EG samples of size n
fn see_gen(u: &mut impl FnMut() -> f64, n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let xs = (0..n)
        .map(|_| (15.0 + 5.0 * normal(u)).round().clamp(0.0, k as f64))
        .collect();
    let ys = (0..n)
        .map(|_| (16.0 + 5.0 * normal(u)).round().clamp(0.0, k as f64))
        .collect();
    (xs, ys)
}

// A1: delta-method Linear SEE agrees with the bootstrap Linear SEE.
#[test]
fn see_analytic_linear_matches_bootstrap() {
    let mut u = lcg(71);
    let (k, n) = (30usize, 3000usize);
    let (xs, ys) = see_gen(&mut u, n, k);
    let a = analytic_see(&xs, &ys, k, k, EquateMethod::Linear, 0.95).unwrap();
    let b = bootstrap_see(&xs, &ys, k, k, EquateMethod::Linear, 2000, 0.95, 12345).unwrap();
    let (lo, hi) = (
        (k as f64 * 0.1).ceil() as usize,
        k - (k as f64 * 0.1).ceil() as usize,
    );
    let mut maxrel = 0.0_f64;
    for x in lo..=hi {
        if a.se[x] > 1e-6 {
            maxrel = maxrel.max((b.se[x] - a.se[x]).abs() / a.se[x]);
        }
    }
    assert!(
        maxrel < 0.15,
        "analytic vs bootstrap Linear SEE relative gap too large: {maxrel}"
    );
}

// A2: Mean SEE is constant in x and equals the closed form.
#[test]
fn see_mean_is_constant() {
    let mut u = lcg(72);
    let (k, n) = (30usize, 2000usize);
    let (xs, ys) = see_gen(&mut u, n, k);
    let a = analytic_see(&xs, &ys, k, k, EquateMethod::Mean, 0.95).unwrap();
    let (_, sx) = moments(&rel_freq(&xs, k).unwrap());
    let (_, sy) = moments(&rel_freq(&ys, k).unwrap());
    let expected = (sx * sx / n as f64 + sy * sy / n as f64).sqrt();
    for x in 0..=k {
        assert!(
            (a.se[x] - expected).abs() < 1e-9 && (a.se[x] - a.se[0]).abs() < 1e-12,
            "Mean SEE not constant"
        );
    }
}

// A3/A4: bootstrap sanity (positive SE, CI brackets the estimate, ~1/sqrt(N)
// shrink), determinism, and the input guards.
#[test]
fn see_bootstrap_sanity_and_guards() {
    let mut u = lcg(73);
    let k = 20usize;
    let (x1, y1) = see_gen(&mut u, 1000, k);
    let (x4, y4) = see_gen(&mut u, 4000, k);
    let b1 = bootstrap_see(&x1, &y1, k, k, EquateMethod::Equipercentile, 500, 0.95, 7).unwrap();
    let b4 = bootstrap_see(&x4, &y4, k, k, EquateMethod::Equipercentile, 500, 0.95, 7).unwrap();
    let (lo, hi) = (
        (k as f64 * 0.1).ceil() as usize,
        k - (k as f64 * 0.1).ceil() as usize,
    );
    for x in lo..=hi {
        assert!(b1.se[x] > 0.0);
        assert!(
            b1.ci_lo[x] <= b1.y_equivalents[x] + 1e-9 && b1.y_equivalents[x] <= b1.ci_hi[x] + 1e-9
        );
    }
    let ratio: f64 = (lo..=hi)
        .map(|x| b1.se[x] / b4.se[x].max(1e-9))
        .sum::<f64>()
        / (hi - lo + 1) as f64;
    assert!(
        (1.5..=2.6).contains(&ratio),
        "SE should ~halve when N x4: {ratio}"
    );
    // determinism
    let d1 = bootstrap_see(&x1, &y1, k, k, EquateMethod::Linear, 300, 0.95, 99).unwrap();
    let d2 = bootstrap_see(&x1, &y1, k, k, EquateMethod::Linear, 300, 0.95, 99).unwrap();
    assert_eq!(d1.se, d2.se);
    let linear_with_degenerate_resamples = bootstrap_see(
        &[0.0, 1.0],
        &[0.0, 1.0],
        1,
        1,
        EquateMethod::Linear,
        10,
        0.95,
        1,
    );
    // Reads the crate-returned SEE result and kills mutations that propagate
    // degenerate linear bootstrap replicate errors to valid original samples.
    assert!(linear_with_degenerate_resamples.is_ok());
    // guards
    assert!(bootstrap_see(&x1, &y1, k, k, EquateMethod::Mean, 1, 0.95, 1).is_err());
    assert!(bootstrap_see(&x1, &y1, k, k, EquateMethod::Mean, 100, 1.5, 1).is_err());
    let oversized = std::panic::catch_unwind(|| {
        bootstrap_see(&x1, &y1, k, k, EquateMethod::Mean, usize::MAX, 0.95, 1)
    });
    assert!(
        oversized.is_ok(),
        "oversized n_boot must return an error instead of panicking"
    );
    assert!(oversized.unwrap().is_err());
    let oversized_k = std::panic::catch_unwind(|| {
        bootstrap_see(&x1, &y1, usize::MAX, k, EquateMethod::Mean, 100, 0.95, 1)
    });
    assert!(
        oversized_k.is_ok(),
        "oversized k_x must return an error instead of panicking"
    );
    assert!(oversized_k.unwrap().is_err());
    let oversized_ky = std::panic::catch_unwind(|| {
        bootstrap_see(&x1, &y1, 1, usize::MAX, EquateMethod::Mean, 100, 0.95, 1)
    });
    assert!(
        oversized_ky.is_ok(),
        "oversized k_y must return an error instead of panicking"
    );
    assert!(oversized_ky.unwrap().is_err());
    // Reads the crate-returned error path and kills mutations that remove the
    // explicit bootstrap work cap while leaving overflow-only checks in place.
    assert!(bootstrap_see(&x1, &y1, k, k, EquateMethod::Mean, 10_001, 0.95, 1).is_err());
    assert!(analytic_see(&x1, &y1, k, k, EquateMethod::Equipercentile, 0.95).is_err());
}

#[test]
fn neat_frequency_rejects_oversized_anchor_ceiling_without_panic() {
    let x = vec![0.0, 1.0, 1.0, 2.0];
    let xv = vec![0.0, 1.0, 1.0, 2.0];
    let y = vec![0.0, 1.0, 2.0, 2.0];
    let yv = vec![0.0, 1.0, 1.0, 2.0];

    let oversized = std::panic::catch_unwind(|| {
        equate_neat(
            &x,
            &xv,
            &y,
            &yv,
            2,
            2,
            usize::MAX,
            0.5,
            NeatMethod::FrequencyEstimation,
        )
    });

    assert!(
        oversized.is_ok(),
        "oversized k_v must return an error instead of panicking"
    );
    // Reads the core-returned error path and kills mutations that leave
    // bivariate's `(k_s + 1) * (k_v + 1)` allocation unchecked.
    assert!(oversized.unwrap().is_err());
}

// The bootstrap SE approximates the TRUE sampling SD of e_Y(x) (from an outer
// Monte-Carlo that redraws fresh 2PL samples) within Monte-Carlo tolerance.
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn see_bootstrap_monte_carlo_500() {
    let (k_x, k_y, n) = (30usize, 40usize, 2000usize);
    let a_x: Vec<f64> = (0..k_x)
        .map(|i| 0.8 + 0.5 * ((i % 5) as f64 / 4.0))
        .collect();
    let b_x: Vec<f64> = (0..k_x)
        .map(|i| 1.5 - 3.0 * i as f64 / (k_x - 1) as f64)
        .collect();
    let a_y: Vec<f64> = (0..k_y)
        .map(|i| 0.9 + 0.4 * ((i % 4) as f64 / 3.0))
        .collect();
    let b_y: Vec<f64> = (0..k_y)
        .map(|i| 1.8 - 3.6 * i as f64 / (k_y - 1) as f64)
        .collect();
    let sim = |u: &mut dyn FnMut() -> f64, a: &[f64], b: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|_| {
                let th = {
                    let u1 = u().max(1e-12);
                    let u2 = u();
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                };
                a.iter()
                    .zip(b)
                    .filter(|(&ai, &bi)| u() < 1.0 / (1.0 + (-(ai * th + bi)).exp()))
                    .count() as f64
            })
            .collect()
    };
    let run = |method: EquateMethod, label: &str| {
        // outer MC: true SD of e_Y(x) over R fresh samples
        let r_out = 500usize;
        let mut uo = lcg(3300);
        let mut vals = vec![0.0_f64; r_out * (k_x + 1)];
        for r in 0..r_out {
            let xs = sim(&mut uo, &a_x, &b_x);
            let ys = sim(&mut uo, &a_y, &b_y);
            let e = equate_eg(&xs, &ys, k_x, k_y, method).unwrap();
            vals[r * (k_x + 1)..(r + 1) * (k_x + 1)].copy_from_slice(&e.y_equivalents);
        }
        let true_sd: Vec<f64> = (0..=k_x)
            .map(|x| {
                let col: Vec<f64> = (0..r_out).map(|r| vals[r * (k_x + 1) + x]).collect();
                let m = col.iter().sum::<f64>() / r_out as f64;
                (col.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / (r_out as f64 - 1.0)).sqrt()
            })
            .collect();
        // mean bootstrap SE over n_samp fresh samples
        let n_samp = 40usize;
        let mut ub = lcg(9900);
        let mut sum_se = vec![0.0_f64; k_x + 1];
        for s_i in 0..n_samp {
            let xs = sim(&mut ub, &a_x, &b_x);
            let ys = sim(&mut ub, &a_y, &b_y);
            let n_boot = 300usize;
            let s = bootstrap_see(
                &xs,
                &ys,
                k_x,
                k_y,
                method,
                n_boot,
                0.95,
                41_000 + s_i as u64,
            )
            .unwrap();
            for x in 0..=k_x {
                sum_se[x] += s.se[x];
            }
        }
        let (lo, hi) = (
            (k_x as f64 * 0.05).ceil() as usize,
            k_x - (k_x as f64 * 0.05).ceil() as usize,
        );
        let (mut rmin, mut rmax) = (f64::INFINITY, f64::NEG_INFINITY);
        for x in lo..=hi {
            let ratio = (sum_se[x] / n_samp as f64) / true_sd[x].max(1e-9);
            rmin = rmin.min(ratio);
            rmax = rmax.max(ratio);
        }
        println!("[see 500] {label}: interior boot/true SD ratio in [{rmin:.3}, {rmax:.3}]");
        assert!(
            rmin > 0.80 && rmax < 1.20,
            "{label} bootstrap SEE off true SD: [{rmin}, {rmax}]"
        );
    };
    run(EquateMethod::Linear, "linear");
    run(EquateMethod::Equipercentile, "equipercentile");
}

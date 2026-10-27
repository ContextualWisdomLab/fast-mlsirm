//! Declared ordinal-score SRMSR RED/GREEN controls.
use mlsirm_core::ordinal_srmsr::evaluate_multigroup_bifactor_grm;
use mlsirm_core::samefit_grm::quadrature;

fn bank(n_groups: usize, n_items: usize, n_cat: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let m = n_cat - 1;
    let mut a = vec![vec![0.0; n_items]; n_groups];
    let s = vec![vec![0.0; n_items]; n_groups];
    let mut t = vec![vec![0.0; n_items * m]; n_groups];
    for g in 0..n_groups {
        for i in 0..n_items {
            a[g][i] = 1.0 + 0.25 * i as f64;
            for k in 0..m {
                t[g][i * m + k] = 1.5 - 1.5 * k as f64;
            }
        }
    }
    (a, s, t)
}

fn brute_bifactor_moments(
    a_g: &[f64],
    a_s: &[f64],
    thresholds: &[f64],
    specific_map: &[i32],
    n_specific: usize,
    mean_g: f64,
    sd_g: f64,
    mean_s: &[f64],
    sd_s: &[f64],
    n_items: usize,
    n_cat: usize,
    q_g: usize,
    q_s: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (xg, wg) = quadrature(q_g).unwrap();
    let (xs, ws) = if n_specific > 0 { quadrature(q_s).unwrap() } else { (vec![0.0], vec![1.0]) };
    let m = n_cat - 1;
    let mut mean = vec![0.0; n_items];
    let mut second = vec![0.0; n_items];
    let pairs = n_items * (n_items - 1) / 2;
    let mut cross = vec![0.0; pairs];
    let qs_len = xs.len();
    let total_s = qs_len.pow(n_specific as u32);
    for (ig, (&ng, &w_g)) in xg.iter().zip(&wg).enumerate() {
        let _ = ig;
        let tg = mean_g + sd_g * ng;
        for s_idx in 0..total_s {
            let mut rem = s_idx;
            let mut w = w_g;
            let mut ts = vec![0.0; n_specific];
            for s in 0..n_specific {
                let d = rem % qs_len;
                rem /= qs_len;
                w *= ws[d];
                ts[s] = mean_s[s] + sd_s[s] * xs[d];
            }
            let mut e1 = vec![0.0; n_items];
            let mut e2 = vec![0.0; n_items];
            for i in 0..n_items {
                let sp = specific_map[i];
                let eta = a_g[i] * tg + if sp >= 0 { a_s[i] * ts[sp as usize] } else { 0.0 };
                let raw = mlsirm_core::poly::grm_logprobs(eta, &thresholds[i * m..(i + 1) * m]);
                for (k, &lp) in raw.iter().enumerate() {
                    let p = lp.exp();
                    let kf = k as f64;
                    e1[i] += kf * p;
                    e2[i] += kf * kf * p;
                }
                mean[i] += w * e1[i];
                second[i] += w * e2[i];
            }
            let mut pair = 0;
            for i in 0..n_items {
                for j in i + 1..n_items {
                    cross[pair] += w * e1[i] * e1[j];
                    pair += 1;
                }
            }
        }
    }
    (mean, second, cross)
}

fn synthetic_responses(n_persons_per_group: usize, n_groups: usize, n_items: usize, n_cat: usize) -> (Vec<usize>, Vec<usize>) {
    let mut y = Vec::with_capacity(n_persons_per_group * n_groups * n_items);
    let mut groups = Vec::with_capacity(n_persons_per_group * n_groups);
    let mut state = 20261027u64;
    for g in 0..n_groups {
        for p in 0..n_persons_per_group {
            groups.push(g);
            let latent = ((p * 7 + g * 3) % n_cat) as usize;
            for i in 0..n_items {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let noise = ((state >> 33) as usize) % n_cat;
                y.push((latent + noise + i) % n_cat);
            }
        }
    }
    (y, groups)
}

#[test]
fn bifactor_expected_correlations_match_brute_product_grid_and_differ_from_first_cut() {
    let (a, mut s, t) = bank(2, 4, 4);
    let map = [0, 0, 1, -1];
    s[0] = vec![0.6, 0.8, 0.7, 0.0];
    s[1] = vec![0.6, 0.5, 0.7, 0.0];
    let g_mean = [0.0, 0.25];
    let g_sd = [1.0, 1.15];
    let s_mean = [vec![0.0, 0.0], vec![0.1, -0.15]];
    let s_sd = [vec![1.0, 1.0], vec![0.9, 1.2]];
    let (y, groups) = synthetic_responses(24, 2, 4, 4);
    let r = evaluate_multigroup_bifactor_grm(
        &y,
        None,
        &groups,
        2,
        4,
        4,
        &map,
        2,
        &a,
        &s,
        &t,
        &g_mean,
        &g_sd,
        &s_mean,
        &s_sd,
        7,
        7,
    )
    .expect("declared bifactor ordinal SRMSR");
    assert_eq!(r.definition, "ordinal_score_pearson_srmsr");
    assert_eq!(r.category_scores, "0..=n_cat-1");
    assert_eq!(r.n_pairs_per_group, 6);
    assert_eq!(r.n_total_pairs, 12);
    assert_eq!(r.groups.len(), 2);
    assert!(r.unweighted_srmsr.is_finite() && r.unweighted_srmsr > 0.0);
    assert!(r.group_size_weighted_srmsr.is_finite() && r.group_size_weighted_srmsr > 0.0);
    for g in 0..2 {
        let (mean, second, cross) = brute_bifactor_moments(
            &a[g], &s[g], &t[g], &map, 2, g_mean[g], g_sd[g], &s_mean[g], &s_sd[g], 4, 4, 7, 7,
        );
        let mut pair = 0;
        let mut sum_sq = 0.0;
        for i in 0..4 {
            for j in i + 1..4 {
                let row = &r.residuals[g * 6 + pair];
                assert_eq!((row.group, row.item_i, row.item_j, row.n_pair), (g, i, j, 24));
                let vi = second[i] - mean[i] * mean[i];
                let vj = second[j] - mean[j] * mean[j];
                let expected = (cross[pair] - mean[i] * mean[j]) / (vi * vj).sqrt();
                assert!((row.expected_corr - expected).abs() < 1e-12, "{row:?} vs {expected}");
                assert!((row.residual - (row.observed_corr - expected)).abs() < 1e-12);
                sum_sq += row.residual * row.residual;
                pair += 1;
            }
        }
        assert!((r.groups[g].srmsr - (sum_sq / 6.0).sqrt()).abs() < 1e-12);
    }
}

#[test]
fn declared_srmsr_rejects_empty_group_and_degenerate_variance() {
    let (a, s, t) = bank(2, 3, 4);
    let err = evaluate_multigroup_bifactor_grm(
        &[0, 1, 2, 1, 2, 3],
        None,
        &[0, 0],
        2,
        3,
        4,
        &[-1; 3],
        0,
        &a,
        &s,
        &t,
        &[0.0, 0.0],
        &[1.0, 1.0],
        &[vec![], vec![]],
        &[vec![], vec![]],
        7,
        1,
    );
    assert!(err.is_err(), "empty group must fail closed");

    let constant = vec![0usize; 12];
    let groups = vec![0, 0, 1, 1];
    let deg = evaluate_multigroup_bifactor_grm(
        &constant,
        None,
        &groups,
        2,
        3,
        4,
        &[-1; 3],
        0,
        &a,
        &s,
        &t,
        &[0.0, 0.0],
        &[1.0, 1.0],
        &[vec![], vec![]],
        &[vec![], vec![]],
        7,
        1,
    );
    assert!(deg.is_err(), "zero observed item variance must fail closed");
}

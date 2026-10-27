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
        assert_eq!(r.groups[g].n_group, 24);
    }
    let first_cut = first_threshold_indicator_srmsr(
        &y, &groups, 2, 4, 4, &a, &s, &t, &map, 2, &g_mean, &g_sd, &s_mean, &s_sd, 7, 7,
    );
    assert!(
        (r.unweighted_srmsr - first_cut).abs() > 1e-12,
        "ordinal SRMSR {} must differ from first-cut indicator SRMSR {}",
        r.unweighted_srmsr,
        first_cut
    );
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
    assert_eq!(err.unwrap_err(), "empty group or unobserved item pair");

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
    assert_eq!(
        deg.unwrap_err(),
        "zero item variance prevents declared ordinal correlation"
    );
}

/// First-threshold indicator SRMSR (`samefit_grm_joint` lines 163–191):
/// dichotomize each category score at `> 0` and take the root mean squared
/// residual between sample and model indicator Pearson correlations.
#[allow(clippy::too_many_arguments)]
fn first_threshold_indicator_srmsr(
    y: &[usize],
    groups: &[usize],
    n_groups: usize,
    n_items: usize,
    n_cat: usize,
    a_g: &[Vec<f64>],
    a_s: &[Vec<f64>],
    thresholds: &[Vec<f64>],
    specific_map: &[i32],
    n_specific: usize,
    mean_g: &[f64],
    sd_g: &[f64],
    mean_s: &[Vec<f64>],
    sd_s: &[Vec<f64>],
    q_g: usize,
    q_s: usize,
) -> f64 {
    let (xg, wg) = quadrature(q_g).unwrap();
    let (xs, ws) = if n_specific > 0 {
        quadrature(q_s).unwrap()
    } else {
        (vec![0.0], vec![1.0])
    };
    let m = n_cat - 1;
    let qs_len = xs.len();
    let total_s = qs_len.pow(n_specific as u32);
    let n = groups.len();
    let mut sum_sq = 0.0;
    let mut pair_count = 0usize;
    for g in 0..n_groups {
        let rows: Vec<usize> = (0..n).filter(|&p| groups[p] == g).collect();
        let n_g = rows.len() as f64;
        let mut mi = vec![0.0; n_items];
        let mut mij = vec![0.0; n_items * (n_items - 1) / 2];
        for (&ng, &w_g) in xg.iter().zip(&wg) {
            let tg = mean_g[g] + sd_g[g] * ng;
            for s_idx in 0..total_s {
                let mut rem = s_idx;
                let mut w = w_g;
                let mut ts = vec![0.0; n_specific];
                for s in 0..n_specific {
                    let d = rem % qs_len;
                    rem /= qs_len;
                    w *= ws[d];
                    ts[s] = mean_s[g][s] + sd_s[g][s] * xs[d];
                }
                let mut p_gt0 = vec![0.0; n_items];
                for i in 0..n_items {
                    let sp = specific_map[i];
                    let eta = a_g[g][i] * tg + if sp >= 0 { a_s[g][i] * ts[sp as usize] } else { 0.0 };
                    let raw = mlsirm_core::poly::grm_logprobs(eta, &thresholds[g][i * m..(i + 1) * m]);
                    p_gt0[i] = raw[1..].iter().map(|&lp| lp.exp()).sum();
                    mi[i] += w * p_gt0[i];
                }
                let mut idx = 0;
                for i in 0..n_items {
                    for j in i + 1..n_items {
                        mij[idx] += w * p_gt0[i] * p_gt0[j];
                        idx += 1;
                    }
                }
            }
        }
        let mut idx = 0;
        for i in 0..n_items {
            for j in i + 1..n_items {
                let mut oi = 0.0;
                let mut oj = 0.0;
                let mut oij = 0.0;
                for &p in &rows {
                    let xi = if y[p * n_items + i] > 0 { 1.0 } else { 0.0 };
                    let xj = if y[p * n_items + j] > 0 { 1.0 } else { 0.0 };
                    oi += xi;
                    oj += xj;
                    oij += xi * xj;
                }
                oi /= n_g;
                oj /= n_g;
                oij /= n_g;
                let denom_o = (oi * (1.0 - oi) * oj * (1.0 - oj)).sqrt();
                let denom_m = (mi[i] * (1.0 - mi[i]) * mi[j] * (1.0 - mi[j])).sqrt();
                let residual = (oij - oi * oj) / denom_o - (mij[idx] - mi[i] * mi[j]) / denom_m;
                sum_sq += residual * residual;
                pair_count += 1;
                idx += 1;
            }
        }
    }
    (sum_sq / pair_count as f64).sqrt()
}

fn call_general_only(
    y: &[usize],
    observed: Option<&[bool]>,
    groups: &[usize],
    n_groups: usize,
    n_items: usize,
    n_cat: usize,
) -> Result<mlsirm_core::ordinal_srmsr::OrdinalSrmsr, String> {
    let (a, s, t) = bank(n_groups, n_items, n_cat);
    evaluate_multigroup_bifactor_grm(
        y,
        observed,
        groups,
        n_groups,
        n_items,
        n_cat,
        &vec![-1; n_items],
        0,
        &a,
        &s,
        &t,
        &vec![0.0; n_groups],
        &vec![1.0; n_groups],
        &vec![vec![]; n_groups],
        &vec![vec![]; n_groups],
        7,
        1,
    )
}

#[test]
fn reviewer_strengthened_controls_kill_all_mutants() {
    let n_items = 3;
    let n_cat = 4;
    let (a, s, t) = bank(2, n_items, n_cat);
    let (mut y, mut groups) = synthetic_responses(20, 1, n_items, n_cat);
    let (y1, g1) = synthetic_responses(60, 1, n_items, n_cat);
    let y1: Vec<usize> = y1.into_iter().map(|v| (v + 1) % n_cat).collect();
    let g1: Vec<usize> = g1.into_iter().map(|_| 1usize).collect();
    y.extend(y1);
    groups.extend(g1);
    let r = call_general_only(&y, None, &groups, 2, n_items, n_cat).expect("unequal groups");
    assert_eq!(r.groups[0].n_group, 20);
    assert_eq!(r.groups[1].n_group, 60);
    let pairs = n_items * (n_items - 1) / 2;
    let mut sum_sq = [0.0, 0.0];
    for row in &r.residuals {
        sum_sq[row.group] += row.residual * row.residual;
    }
    let expected_unweighted = ((sum_sq[0] + sum_sq[1]) / (2 * pairs) as f64).sqrt();
    let expected_weighted =
        ((20.0 * sum_sq[0] + 60.0 * sum_sq[1]) / ((20.0 + 60.0) * pairs as f64)).sqrt();
    assert!((r.unweighted_srmsr - expected_unweighted).abs() < 1e-12);
    assert!((r.group_size_weighted_srmsr - expected_weighted).abs() < 1e-12);
    assert!((r.unweighted_srmsr - r.group_size_weighted_srmsr).abs() > 1e-12);
    let first_cut = first_threshold_indicator_srmsr(
        &y,
        &groups,
        2,
        n_items,
        n_cat,
        &a,
        &s,
        &t,
        &[-1; 3],
        0,
        &[0.0, 0.0],
        &[1.0, 1.0],
        &[vec![], vec![]],
        &[vec![], vec![]],
        7,
        1,
    );
    assert!((r.unweighted_srmsr - first_cut).abs() > 1e-12);

    let empty = call_general_only(&[0, 1, 2, 1, 2, 3], None, &[0, 0], 2, 3, 4);
    assert_eq!(empty.unwrap_err(), "empty group or unobserved item pair");

    let constant = vec![0usize; 12];
    let deg = call_general_only(&constant, None, &[0, 0, 1, 1], 2, 3, 4);
    assert_eq!(
        deg.unwrap_err(),
        "zero item variance prevents declared ordinal correlation"
    );
}

/// Item-level missingness must not mix a pairwise cross-product with univariate
/// moments. The mask below makes that mixture equal 1.25 (> 1.0); the
/// pairwise-complete Pearson correlation equals 1.0. A fully masked respondent
/// is not counted in `n_group`.
#[test]
fn reviewer_probe_partial_missingness_and_unequal_groups() {
    let n_items = 3;
    let n_cat = 4;
    let mut y = vec![
        0, 0, 0, // person 0: complete (0, 0) on pair (0, 1)
        2, 2, 1, // person 1: complete (2, 2) on pair (0, 1)
        0, 9, 2, // person 2: item 0 = 0, item 1 masked
        2, 9, 1, // person 3: item 0 = 2, item 1 masked
        1, 9, 3, // person 4: item 0 = 1, item 1 masked
        9, 0, 1, // person 5: item 0 masked, item 1 = 0
        9, 2, 2, // person 6: item 0 masked, item 1 = 2
        9, 1, 0, // person 7: item 0 masked, item 1 = 1
        9, 9, 9, // person 8: fully masked
    ];
    let mut mask = vec![
        true, true, true, //
        true, true, true, //
        true, false, true, //
        true, false, true, //
        true, false, true, //
        false, true, true, //
        false, true, true, //
        false, true, true, //
        false, false, false, //
    ];
    let mut groups = vec![0; 9];
    let (y1, g1) = synthetic_responses(8, 1, n_items, n_cat);
    let y1: Vec<usize> = y1.into_iter().map(|v| (v + 2) % n_cat).collect();
    y.extend(y1);
    mask.extend(std::iter::repeat(true).take(8 * n_items));
    groups.extend(g1.into_iter().map(|_| 1usize));

    let broken = mixed_margin_correlation(&y, &mask, &groups, 0, 0, 1);
    assert!(
        (broken - 1.25).abs() < 1e-12,
        "reviewer probe must produce 1.25 before repair, got {broken}"
    );

    let r = call_general_only(&y, Some(&mask), &groups, 2, n_items, n_cat)
        .expect("pairwise-complete ordinal correlation");
    let pair = r
        .residuals
        .iter()
        .find(|row| row.group == 0 && row.item_i == 0 && row.item_j == 1)
        .unwrap();
    assert!(
        pair.observed_corr.abs() <= 1.0,
        "observed correlation exceeded 1.0: {}",
        pair.observed_corr
    );
    assert_eq!(pair.n_pair, 2);
    assert!((pair.observed_corr - 1.0).abs() < 1e-12, "{}", pair.observed_corr);
    assert_eq!(r.groups[0].n_group, 8, "fully masked respondent is not a group weight");
    assert_eq!(r.groups[1].n_group, 8);
}

/// The defective estimator: univariate moments over each item's own observed
/// rows, cross-product over the pairwise-complete rows.
fn mixed_margin_correlation(
    y: &[usize],
    mask: &[bool],
    groups: &[usize],
    group: usize,
    i: usize,
    j: usize,
) -> f64 {
    let n_items = 3;
    let mut ni = 0.0;
    let mut nj = 0.0;
    let mut sum_i = 0.0;
    let mut sum_i2 = 0.0;
    let mut sum_j = 0.0;
    let mut sum_j2 = 0.0;
    let mut cij = 0.0;
    let mut np = 0.0;
    for (person, &g) in groups.iter().enumerate() {
        if g != group {
            continue;
        }
        let row = person * n_items;
        if mask[row + i] {
            let v = y[row + i] as f64;
            ni += 1.0;
            sum_i += v;
            sum_i2 += v * v;
        }
        if mask[row + j] {
            let v = y[row + j] as f64;
            nj += 1.0;
            sum_j += v;
            sum_j2 += v * v;
        }
        if mask[row + i] && mask[row + j] {
            cij += y[row + i] as f64 * y[row + j] as f64;
            np += 1.0;
        }
    }
    let mean_i = sum_i / ni;
    let mean_j = sum_j / nj;
    let var_i = sum_i2 / ni - mean_i * mean_i;
    let var_j = sum_j2 / nj - mean_j * mean_j;
    let cross = cij / np;
    (cross - mean_i * mean_j) / (var_i * var_j).sqrt()
}

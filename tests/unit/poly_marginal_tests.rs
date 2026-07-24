use super::*;

#[test]
fn lsirm_rejects_unbounded_categories_and_iterations() {
    let y = [0usize];
    assert!(fit_poly_lsirm(
        &y,
        None,
        1,
        1,
        POLY_MAX_CAT + 1,
        1,
        PolyModel::Grm,
        7,
        7,
        1,
        1e-6,
    )
    .is_err());
    assert!(fit_poly_lsirm(
        &y,
        None,
        1,
        1,
        2,
        1,
        PolyModel::Grm,
        7,
        7,
        POLY_MAX_ITER + 1,
        1e-6,
    )
    .is_err());
}

#[test]
fn poly_marginal_boundaries_and_grm_paths_are_explicit() {
    assert!(xi_tensor_grid(99, 1).is_err());
    assert_eq!(xi_tensor_grid(41, 100).unwrap_err(), "xi grid too large");
    assert_eq!(
        xi_tensor_grid(41, 4).unwrap_err(),
        "q_xi ** latent_dim exceeds the tensor-grid limit"
    );

    let thresholds = [0.5, -0.5];
    let counts = [1.0, 2.0, 1.0];
    let lp = poly_cell(0.25, PolyModel::Grm, &thresholds, 3);
    assert_eq!(lp.len(), 3);
    assert!((lp.iter().map(|v| v.exp()).sum::<f64>() - 1.0).abs() < 1e-12);
    let (g_thresholds, g_base) = poly_cat_grad(0.25, PolyModel::Grm, &thresholds, &counts);
    assert_eq!(g_thresholds.len(), 2);
    assert!(g_thresholds.iter().all(|v| v.is_finite()));
    assert!(g_base.is_finite());

    let theta = [0.0];
    let xi = [0.0];
    let empty_counts = [0.0, 0.0, 0.0];
    let ctx = ItemCtx {
        model: PolyModel::Grm,
        n_cat: 3,
        latent_dim: 1,
        eps: 1e-8,
        theta: &theta,
        xi_grid: &xi,
        n_xi: 1,
        rbar_i: &empty_counts,
        lambda_alpha: 0.0,
        mu_alpha: 0.0,
        lambda_zeta: 0.0,
    };
    let (objective, gradient) = item_neg_ll_grad(&[0.0, 0.5, -0.5, 0.0], &ctx);
    assert_eq!(objective, 0.0);
    assert_eq!(gradient, vec![0.0; 4]);

    assert!(fit_poly_lsirm(&[], None, 0, 1, 2, 1, PolyModel::Grm, 7, 7, 1, 1e-6).is_err());
    assert!(fit_poly_lsirm(&[], None, 1, 0, 2, 1, PolyModel::Grm, 7, 7, 1, 1e-6).is_err());
    assert!(fit_poly_lsirm(&[], None, 1, 1, 2, 1, PolyModel::Grm, 7, 7, 1, 1e-6).is_err());
    assert!(fit_poly_lsirm(&[0], Some(&[]), 1, 1, 2, 1, PolyModel::Grm, 7, 7, 1, 1e-6,).is_err());
    assert!(fit_poly_lsirm(
        &[3],
        Some(&[true]),
        1,
        1,
        3,
        1,
        PolyModel::Grm,
        7,
        7,
        1,
        1e-6,
    )
    .is_err());

    // Reads crate output (`fit.loglik`, dimensions below). Kills the mutation
    // that validates masked cells or indexes `freq[y]` without the observed guard.
    let fit = fit_poly_lsirm(
        &[0, 99],
        Some(&[true, false]),
        2,
        1,
        3,
        3,
        PolyModel::Grm,
        7,
        7,
        1,
        1e-6,
    )
    .unwrap();
    assert_eq!(fit.n_iter, 1);
    assert!(fit.loglik.is_finite());
    assert_eq!(fit.slope.len(), 1);
    assert_eq!(fit.cat_params[0].len(), 2);
    assert_eq!(fit.zeta.len(), 3);
    assert_eq!(fit.theta_eap.len(), 2);
    assert_eq!(fit.theta_sd.len(), 2);
    assert_eq!(fit.xi_eap.len(), 6);
    assert!(fit
        .theta_eap
        .iter()
        .chain(&fit.theta_sd)
        .chain(&fit.xi_eap)
        .all(|v| v.is_finite()));
}

fn dist_matrix(z: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut out = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            let mut s = 0.0;
            for k in 0..d {
                let dd = z[i * d + k] - z[j * d + k];
                s += dd * dd;
            }
            out.push(s.sqrt());
        }
    }
    out
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
}

#[test]
fn fit_poly_lsirm_recovers_positions_and_slopes() {
    let (n_persons, n_items, k, ld) = (1500usize, 6usize, 3usize, 2usize);
    let mut st = 314159u64;
    let mut u = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / ((1u64 << 53) as f64)
    };
    macro_rules! nrm {
        () => {{
            let u1 = u().max(1e-12);
            let u2 = u();
            (-2.0_f64 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }};
    }
    // true item positions on two separated clusters, slopes, GPCM intercepts
    let mut zeta_true = vec![0.0_f64; n_items * ld];
    for i in 0..n_items {
        let cx = if i < n_items / 2 { -1.2 } else { 1.2 };
        zeta_true[i * ld] = cx + 0.3 * nrm!();
        zeta_true[i * ld + 1] = 0.3 * nrm!();
    }
    let a_true: Vec<f64> = (0..n_items).map(|i| 1.0 + 0.08 * i as f64).collect();
    let c_true: Vec<Vec<f64>> = (0..n_items)
        .map(|i| vec![0.0, 0.2 - 0.05 * i as f64, -0.2 + 0.05 * i as f64])
        .collect();
    let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
    let mut y = vec![0usize; n_persons * n_items];
    let mut theta_true = vec![0.0_f64; n_persons];
    for p in 0..n_persons {
        let theta = nrm!();
        theta_true[p] = theta;
        let xi: Vec<f64> = (0..ld).map(|_| nrm!()).collect();
        for i in 0..n_items {
            let mut dist2 = 1e-8;
            for kk in 0..ld {
                let dd = xi[kk] - zeta_true[i * ld + kk];
                dist2 += dd * dd;
            }
            let base = a_true[i] * theta - dist2.sqrt();
            let mut ic = vec![0.0; k];
            ic[1..].copy_from_slice(&c_true[i][1..]);
            let lp = gpcm_logprobs(base, &scores, &ic);
            let uu = u();
            let mut cum = 0.0;
            let mut cat = k - 1;
            for (c, l) in lp.iter().enumerate() {
                cum += l.exp();
                if uu < cum {
                    cat = c;
                    break;
                }
            }
            y[p * n_items + i] = cat;
        }
    }
    let fit = fit_poly_lsirm(
        &y,
        None,
        n_persons,
        n_items,
        k,
        ld,
        PolyModel::Gpcm,
        7,
        7,
        40,
        1e-5,
    )
    .unwrap();
    assert!(fit.loglik.is_finite());
    // ABSOLUTE-agreement checks (correlation only shows association, not
    // identity): slope RMSE, and RMSE of the item-item distance matrix, which
    // is exactly invariant to the position rotation/reflection/translation
    // ambiguity while gamma = 1 fixes its absolute scale.
    let slope_rmse = rmse(&a_true, &fit.slope);
    assert!(slope_rmse < 0.25, "slope RMSE {slope_rmse}");
    let dm_true = dist_matrix(&zeta_true, n_items, ld);
    let dm_hat = dist_matrix(&fit.zeta, n_items, ld);
    let pos_rmse = rmse(&dm_true, &dm_hat);
    assert!(pos_rmse < 0.6, "position distance-matrix RMSE {pos_rmse}");
    // person trait recovery: EAP is shrunk toward the prior, so correlation
    // (association) is the appropriate metric here, not RMSE
    let corr = {
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let (mt, me) = (mean(&theta_true), mean(&fit.theta_eap));
        let (mut num, mut dt, mut de) = (0.0, 0.0, 0.0);
        for p in 0..n_persons {
            num += (theta_true[p] - mt) * (fit.theta_eap[p] - me);
            dt += (theta_true[p] - mt).powi(2);
            de += (fit.theta_eap[p] - me).powi(2);
        }
        num / (dt.sqrt() * de.sqrt())
    };
    assert!(corr > 0.6, "theta EAP corr {corr}");
    assert!(fit.theta_sd.iter().all(|s| s.is_finite() && *s > 0.0));
}

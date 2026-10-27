//! Conditional M-step for the finite standardized-node quadrature objective.
//! The latent integration indices stay fixed. Group means/SDs enter the item
//! predictor, so posterior counts and both M-steps optimize the SAME discrete
//! objective, not a moment approximation to a different continuous integral.
//! Chalmers (2012, section 3.1, pp. 5–6, Eqs. 6–11; dichotomous EM
//! illustration, with ordinal probabilities from Eq. 2); Meng and Rubin
//! (1993, pp. 268–269, Eqs. 1.1–1.2 and conditional maximization discussion).
//!
//! References:
//! Chalmers, R. P. (2012). mirt: A multidimensional item response theory package
//! for the R environment. Journal of Statistical Software, 48(6), 1–29.
//! https://doi.org/10.18637/jss.v048.i06
//! Meng, X.-L., & Rubin, D. B. (1993). Maximum likelihood estimation via the
//! ECM algorithm: A general framework. Biometrika, 80(2), 267–278.
//! https://doi.org/10.1093/biomet/80.2.267
//! Caller Newton1e-9/ridge/iteration controls are unchanged; no CPU fitting.
use crate::bifactor_grm::{BifactorMultigroupConfig, ItemParams, Validated};
pub(super) fn update(
    v: &Validated,
    pars: &[Vec<ItemParams>],
    counts: &[Vec<Vec<Vec<f64>>>],
    cfg: &BifactorMultigroupConfig,
    x: &[f64],
    z: &[f64],
    mus: &mut [f64],
    sds: &mut [f64],
    sm: &mut [Vec<f64>],
    ss: &mut [Vec<f64>],
) -> Result<bool, String> {
    let mut limited = false;
    for g in 1..pars.len() {
        // (factor index, SD rather than mean); general factor index0.
        let mut roles = vec![(0usize, false), (0, true)];
        let mut init = vec![mus[g], sds[g]];
        for s in 0..v.n_specific {
            if cfg.estimate_specific_means {
                roles.push((s + 1, false));
                init.push(sm[g][s]);
            }
            if cfg.estimate_specific_vars {
                roles.push((s + 1, true));
                init.push(ss[g][s]);
            }
        }
        let p = roles.len();
        let (limit, _) = crate::gpu_mstep::limits()?;
        let rows = counts[g].iter().try_fold(0usize, |n, r| {
            n.checked_add(r.len()).ok_or("group row size overflow")
        })?;
        let row_width = p
            .checked_add(v.n_cat.checked_mul(2).ok_or("group width overflow")?)
            .and_then(|n| n.checked_sub(1))
            .ok_or("group width overflow")?;
        let input = rows
            .checked_mul(row_width)
            .and_then(|n| n.checked_add(15 + 2 * p))
            .and_then(|n| n.checked_mul(2))
            .ok_or("group input overflow")?;
        crate::gpu_mstep::checked_buffer(input, limit)?;
        if input > 33_554_432 {
            return Err("group exact offset range exceeded".into());
        }
        let width = 1 + p + p * p;
        let mut data = vec![p as f64, p as f64, 0.0, v.n_cat as f64, 0.0];
        data.extend_from_slice(&init);
        let mut nn = 0;
        for (i, par) in pars[g].iter().enumerate() {
            for (t, &node_g) in x.iter().enumerate() {
                let nh = if par.a_s.is_some() { z.len() } else { 1 };
                for h in 0..nh {
                    let owner = v.item_block[i];
                    let mut constant = 0.0;
                    if let Some(s) = owner {
                        if !cfg.estimate_specific_means {
                            constant += par.a_s.unwrap() * sm[g][s];
                        }
                        if !cfg.estimate_specific_vars {
                            constant += par.a_s.unwrap() * ss[g][s] * z[h];
                        }
                    }
                    for &(factor, sd) in &roles {
                        let coordinate = if factor == 0 {
                            par.a_g * if sd { node_g } else { 1.0 }
                        } else if owner == Some(factor - 1) {
                            par.a_s.unwrap() * if sd { z[h] } else { 1.0 }
                        } else {
                            0.0
                        };
                        data.push(coordinate);
                    }
                    data.extend(par.d.iter().map(|d| d + constant));
                    data.extend_from_slice(
                        &counts[g][i][if par.a_s.is_some() {
                            t * z.len() + h
                        } else {
                            t
                        }],
                    );
                    nn += 1;
                }
            }
        }
        data[2] = nn as f64;
        data.extend([
            0.0,
            0.0,
            1.0,
            cfg.ridge,
            cfg.newton_iter as f64,
            25.0,
            1e-9,
            1e-9,
            1e-14,
            1e-4,
        ]);
        data[4] = data.len() as f64;
        data.extend(roles.iter().map(|r| if r.1 { 1.0 } else { 0.0 }));
        if data
            .iter()
            .any(|v| !v.is_finite() || !(*v as f32).is_finite())
        {
            return Err("invalid conditional group M-step input".into());
        }
        let packed: Vec<f32> = data
            .into_iter()
            .flat_map(|v| {
                let h = v as f32;
                [h, (v - h as f64) as f32]
            })
            .collect();
        let cells = width + 3 * p + 9;
        let raw = crate::gpu_mstep::dispatch(
            include_str!("gpu_group_em.wgsl"),
            &packed,
            cells * 2,
            1,
            "main",
        )?;
        let r: Vec<f64> = raw
            .chunks_exact(2)
            .map(|w| w[0] as f64 + w[1] as f64)
            .collect();
        let status = r[width + 3 * p];
        if status != 1.0 && status != 5.0 {
            return Err(format!("conditional group M-step refused status={status}"));
        }
        #[cfg(test)]
        if std::env::var_os("FMLS_GROUP_GRADIENT_ORACLE").is_some() {
            let mut gradient = vec![0.0_f64; p];
            let row_width = p + 2 * v.n_cat - 1;
            let represented: Vec<f64> = packed
                .chunks_exact(2)
                .map(|w| w[0] as f64 + w[1] as f64)
                .collect();
            let next = &r[width..width + p];
            let mut before = 0.0_f64;
            let mut after = 0.0_f64;
            for n in 0..nn {
                let row = 5 + p + n * row_width;
                let coords = &represented[row..row + p];
                let thresholds = &represented[row + p..row + p + v.m1];
                let counts = &represented[row + p + v.m1..row + row_width];
                let eta = coords.iter().zip(next).map(|(x, b)| x * b).sum::<f64>();
                let eta0 = coords.iter().zip(&init).map(|(x, b)| x * b).sum::<f64>();
                let lp = super::grm_logprobs(eta, thresholds);
                let lp0 = super::grm_logprobs(eta0, thresholds);
                after -= counts.iter().zip(lp).map(|(w, l)| w * l).sum::<f64>();
                before -= counts.iter().zip(lp0).map(|(w, l)| w * l).sum::<f64>();
                let (gbase, _) = super::grm_node_gradient(eta, thresholds, counts);
                for j in 0..p {
                    gradient[j] -= gbase * coords[j];
                }
            }
            let norm = gradient.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            eprintln!("GROUP_NATIVE_STATIONARITY group={g} status={status} gradient_inf={norm:.16e} expected_objective_change={:.16e}",after-before);
            assert!(after <= before + 32.0 * f64::EPSILON * (1.0 + before.abs()));
            if status == 1.0 {
                assert!(
                    norm <= 1e-9,
                    "device group stationary status must meet unchanged native1e-9: {norm}"
                );
            }
        }
        limited |= status == 5.0;
        for (j, &(factor, sd)) in roles.iter().enumerate() {
            let value = r[width + j];
            if !value.is_finite() || (sd && value <= 0.0) {
                return Err("invalid group conditional result".into());
            }
            if factor == 0 {
                if sd {
                    sds[g] = value
                } else {
                    mus[g] = value
                }
            } else if sd {
                ss[g][factor - 1] = value
            } else {
                sm[g][factor - 1] = value
            }
        }
    }
    Ok(limited)
}

//! Joint GRM moment tangent in the fitted common/free and population coordinates.
//! Maydeu-Olivares (2013), equations 4 and 10–11: derivatives must span actual
//! free parameters. Reference: Maydeu-Olivares, A. (2013). Goodness-of-fit assessment
//! of item response theory models. Measurement, 11, 71–101.
//! https://doi.org/10.1080/15366367.2013.831680
use crate::samefit_grm::{decode, Coordinate, PackedArchive};

/// Hybrid joint M2: GPU f32 moments/Xi, CPU f64 tangent/projection/distributions.
/// The empirical-margin independence baseline is new, not a recovered historical null.
/// Maydeu-Olivares (2013, p. 78, Equations 10–11) supplies the joint quadratic
/// form. Independent groups use block covariance N/N_g * Xi_g. Returned
/// measured-rank df does not establish calibration at nonidentified points.
///
/// # References
/// Maydeu-Olivares, A. (2013). Goodness-of-fit assessment of item response theory
/// models. *Measurement: Interdisciplinary Research and Perspectives, 11*(3),
/// 71–101. https://doi.org/10.1080/15366367.2013.831680
#[cfg(feature = "gpu")]
pub fn joint_diagnostics(
    y: &[usize],
    groups: &[usize],
    a: &PackedArchive,
    q: usize,
) -> Result<serde_json::Value, String> {
    joint_diagnostics_with_backend(y, groups, a, q, MomentBackend::Gpu)
}
/// Explicit moment arithmetic selection; GPU failures never select CPU.
#[derive(Clone, Copy, Debug)]
pub enum MomentBackend {
    Cpu,
    Gpu,
}
/// Joint residual projection with explicit moment backend selection.
/// Maydeu-Olivares (2013, p. 78, Equations 10–11) supplies the quadratic form;
/// p. 82, Equations 26–28 supplies RMSEA. The implementation's measured rank
/// is algebraic; rank-deficient points require separate calibration evidence.
///
/// # References
/// Maydeu-Olivares, A. (2013). Goodness-of-fit assessment of item response theory
/// models. *Measurement: Interdisciplinary Research and Perspectives, 11*(3),
/// 71–101. https://doi.org/10.1080/15366367.2013.831680
#[cfg(feature = "gpu")]
pub fn joint_diagnostics_with_backend(
    y: &[usize],
    groups: &[usize],
    a: &PackedArchive,
    q: usize,
    backend: MomentBackend,
) -> Result<serde_json::Value, String> {
    let j = a.n_items();
    let z = a.n_categories() - 1;
    let ng = a.n_groups();
    let n = groups.len();
    if n == 0 || y.len() != n * j || y.iter().any(|&v| v > z) || groups.iter().any(|&g| g >= ng) {
        return Err("invalid complete joint input".into());
    }
    let cons = constraints(j, z);
    let s = cons.len();
    let total = s * ng;
    let p = a.packed_bits().len();
    // Packed columns can be redundant. Positive df is checked by the measured
    // tangent rank in strict::project, not by the raw coordinate count.
    let (nodes, weights) = crate::samefit_grm::quadrature(q)?;
    let gpu = if matches!(backend, MomentBackend::Cpu) {
        crate::samefit_strict::cpu_moments(a, &nodes, &weights)?
    } else {
        crate::samefit_grm_gpu::moments_gpu(a, &nodes, &weights)?
    };
    let (cpu_m, delta) = analytic_moments_delta(a, &nodes, &weights)?;
    let cpu_oracle = crate::samefit_strict::cpu_moments(a, &nodes, &weights)?;
    let max_xi_error = cpu_oracle
        .xi
        .iter()
        .zip(&gpu.xi)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    let max_m_error = cpu_m
        .iter()
        .zip(&gpu.moments)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    if max_m_error > 5e-6 {
        return Err(format!("GPU moment parity failed {max_m_error}"));
    }
    let mut counts = vec![0usize; ng];
    let mut obs = vec![0.0; total];
    for (row, &g) in groups.iter().enumerate() {
        counts[g] += 1;
        for (r, c) in cons.iter().enumerate() {
            if c.iter().all(|&(i, k)| y[row * j + i] > k) {
                obs[g * s + r] += 1.0;
            }
        }
    }
    if counts.iter().any(|&v| v == 0) {
        return Err("empty group".into());
    }
    for g in 0..ng {
        for r in 0..s {
            obs[g * s + r] /= counts[g] as f64;
        }
    }
    let residual: Vec<_> = obs.iter().zip(&gpu.moments).map(|(o, m)| o - m).collect();
    let mut xi = vec![0.0; total * total];
    for g in 0..ng {
        for r in 0..s {
            for c in 0..s {
                xi[(g * s + r) * total + g * s + c] =
                    gpu.xi[g * s * s + r * s + c] * n as f64 / counts[g] as f64;
            }
        }
    }
    let strict = crate::samefit_strict::project(&residual, &delta, xi, total, p, n as f64)?;
    let m2 = strict["m2"].as_f64().ok_or("missing M2")?;
    let mut null_m2 = 0.0;
    let mut null_df = 0.0;
    let mut baseline_projections = vec![];
    let null_p = j * z;
    for g in 0..ng {
        let margin = &obs[g * s..g * s + null_p];
        if margin.iter().any(|&v| v <= 0.0 || v >= 1.0) {
            return Err("boundary independence margin".into());
        }
        let product =
            |cs: &[(usize, usize)]| cs.iter().map(|&(i, k)| margin[i * z + k]).product::<f64>();
        let nm: Vec<_> = cons.iter().map(|c| product(c)).collect();
        let mut nd = vec![0.0; s * null_p];
        let mut nx = vec![0.0; s * s];
        for (r, c) in cons.iter().enumerate() {
            for (idx, &(i, k)) in c.iter().enumerate() {
                nd[r * null_p + i * z + k] = c
                    .iter()
                    .enumerate()
                    .filter(|(h, _)| *h != idx)
                    .map(|(_, &(i, k))| margin[i * z + k])
                    .product();
            }
            for (h, other) in cons.iter().enumerate() {
                let mut merged = c.clone();
                for &(i, k) in other {
                    if let Some(v) = merged.iter_mut().find(|v| v.0 == i) {
                        v.1 = v.1.max(k);
                    } else {
                        merged.push((i, k));
                    }
                }
                nx[r * s + h] = product(&merged) - nm[r] * nm[h];
            }
        }
        let ne: Vec<_> = obs[g * s..(g + 1) * s]
            .iter()
            .zip(&nm)
            .map(|(o, m)| o - m)
            .collect();
        let bp = crate::samefit_strict::project(&ne, &nd, nx, s, null_p, counts[g] as f64)?;
        null_m2 += bp["m2"].as_f64().ok_or("missing baseline M2")?;
        null_df += bp["df"].as_f64().ok_or("missing baseline df")?;
        baseline_projections.push(bp);
    }
    let mut output = fit_indices(m2, strict["df"].as_f64().unwrap(), n, null_m2, null_df)?;
    output["strict_projection"] = strict;
    output["baseline_projections"] = serde_json::json!(baseline_projections);
    let mut residual_rows = vec![];
    let mut squared = 0.0;
    let mut pair_count = 0;
    for g in 0..ng {
        let mut row = 0;
        for i in 0..j {
            for h in i + 1..j {
                let pair = j * z + row * z * z;
                let oi = obs[g * s + i * z];
                let oh = obs[g * s + h * z];
                let mi = gpu.moments[g * s + i * z];
                let mh = gpu.moments[g * s + h * z];
                let denom_o = (oi * (1.0 - oi) * oh * (1.0 - oh)).sqrt();
                let denom_m = (mi * (1.0 - mi) * mh * (1.0 - mh)).sqrt();
                if denom_o == 0.0 || denom_m == 0.0 {
                    return Err("degenerate first threshold residual".into());
                }
                let residual = (obs[g * s + pair] - oi * oh) / denom_o
                    - (gpu.moments[g * s + pair] - mi * mh) / denom_m;
                squared += residual * residual;
                pair_count += 1;
                residual_rows
                    .push(serde_json::json!({"group":g,"item_i":i,"item_j":h,"residual":residual}));
                row += 1;
            }
        }
    }
    output["first_threshold_indicator_srmsr"] =
        serde_json::json!((squared / pair_count as f64).sqrt());
    output["first_threshold_indicator_residuals"] = serde_json::json!(residual_rows);
    output["gpu_dispatch"] = serde_json::json!({"backend":gpu.backend,"device":gpu.device_name,"dispatches":gpu.dispatches,"moments_and_xi":if gpu.dispatches==0{"cpu_f64"}else{"gpu_f32"},"residual":"cpu_f64","delta":"cpu_f64_analytic","projection":"cpu_f64_whitened_pivoted_reorthogonalization_no_ridge","distribution":"cpu_f64","max_moment_abs_error":max_m_error,"max_xi_abs_error":max_xi_error});
    output["baseline"] = serde_json::json!(
        "new groupwise empirical cumulative margins; external independence comparator"
    );
    output["n_moments"] = serde_json::json!(total);
    output["n_parameters"] = serde_json::json!(p);
    output["group_counts"] = serde_json::json!(counts);
    output["quadpts"] = serde_json::json!(q);
    output["adoption_gate"]=serde_json::json!("pending rank/conditioning, f32 Xi sensitivity, quadrature sensitivity and independent review");
    Ok(output)
}

/// Enumerate cumulative univariate and cross-item bivariate indicators.
/// Maydeu-Olivares (2013, pp. 77–78, Equations 4 and 10–11) defines the
/// first/second-order moment space; cumulative indicators are its invertible
/// nonredundant category-indicator basis change.
///
/// # References
/// Maydeu-Olivares, A. (2013). Goodness-of-fit assessment of item response theory
/// models. *Measurement: Interdisciplinary Research and Perspectives, 11*(3),
/// 71–101. https://doi.org/10.1080/15366367.2013.831680
pub fn constraints(j: usize, z: usize) -> Vec<Vec<(usize, usize)>> {
    let mut out = vec![];
    for i in 0..j {
        for k in 0..z {
            out.push(vec![(i, k)]);
        }
    }
    for i in 0..j {
        for h in i + 1..j {
            for k in 0..z {
                for l in 0..z {
                    out.push(vec![(i, k), (h, l)]);
                }
            }
        }
    }
    out
}
/// CPU f64 analytic tangent, including focal mean and log SD via theta=mu+sd*z.
/// This is not GPU execution. Used as a numerical oracle and integration stage.
/// Maydeu-Olivares (2013, p. 77, Equation 4) defines Delta as the derivative of
/// univariate/bivariate probabilities in the fitted coordinates. Product and
/// chain rules here evaluate that derivative after a cumulative-indicator basis
/// change. The paper does not specify GPU arithmetic or this coordinate encoding.
///
/// # References
/// Maydeu-Olivares, A. (2013). Goodness-of-fit assessment of item response theory
/// models. *Measurement: Interdisciplinary Research and Perspectives, 11*(3),
/// 71–101. https://doi.org/10.1080/15366367.2013.831680
pub fn analytic_moments_delta(
    a: &PackedArchive,
    nodes: &[f64],
    weights: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let d = decode(a)?;
    let coords = a.coordinates()?;
    let p = coords.len();
    let j = a.n_items();
    let z = a.n_categories() - 1;
    let groups = a.n_groups();
    if nodes.is_empty()
        || nodes.len() != weights.len()
        || nodes.iter().any(|v| !v.is_finite())
        || weights.iter().any(|v| !v.is_finite() || *v < 0.0)
        || (weights.iter().sum::<f64>() - 1.0).abs() > 1e-10
    {
        return Err("invalid quadrature".into());
    }
    let cons = constraints(j, z);
    let s = cons.len();
    let mut m = vec![0.0; groups * s];
    let mut delta = vec![0.0; groups * s * p];
    // A common item has only the group-zero slope coordinate; free items have all groups.
    let common: Vec<bool> = (0..j)
        .map(|i| !coords.contains(&Coordinate::Slope(1, i)))
        .collect();
    for g in 0..groups {
        for (&t, &w) in nodes.iter().zip(weights) {
            let theta = d.means[g] + d.sds[g] * t;
            let mut prob = vec![0.0; j * z];
            let mut deriv = vec![0.0; j * z * p];
            for i in 0..j {
                for k in 0..z {
                    let row = i * z + k;
                    let eta = d.slope[g * j + i] * theta + d.intercepts[(g * j + i) * z + k];
                    let s = 1.0 / (1.0 + (-eta).exp());
                    prob[row] = s;
                    for (c, coord) in coords.iter().enumerate() {
                        let v = match *coord {
                            Coordinate::Slope(h, item) if item == i && (h == g || common[i]) => {
                                theta
                            }
                            Coordinate::FirstIntercept(h, item)
                                if item == i && (h == g || common[i]) =>
                            {
                                1.0
                            }
                            Coordinate::LogGap(h, item, gap)
                                if item == i && (h == g || common[i]) && gap <= k =>
                            {
                                -f64::from_bits(a.packed_bits()[c]).exp()
                            }
                            Coordinate::Mean(h) if h == g => d.slope[g * j + i],
                            Coordinate::LogSd(h) if h == g => d.slope[g * j + i] * d.sds[g] * t,
                            _ => 0.0,
                        };
                        deriv[row * p + c] = s * (1.0 - s) * v;
                    }
                }
            }
            for (r, cs) in cons.iter().enumerate() {
                let r = g * s + r;
                let val = cs.iter().map(|&(i, k)| prob[i * z + k]).product::<f64>();
                m[r] += w * val;
                for c in 0..p {
                    let mut dv = 0.0;
                    for (idx, &(i, k)) in cs.iter().enumerate() {
                        let others = cs
                            .iter()
                            .enumerate()
                            .filter(|(h, _)| *h != idx)
                            .map(|(_, &(i, k))| prob[i * z + k])
                            .product::<f64>();
                        dv += deriv[(i * z + k) * p + c] * others;
                    }
                    delta[r * p + c] += w * dv;
                }
            }
        }
    }
    Ok((m, delta))
}
/// Reuse native distribution functions and comparative metrics; CPU f64 stage.
/// RMSEA and CI follow Maydeu-Olivares (2013), equations 26–28.
/// Maydeu-Olivares (2013, p. 82, Equations 26–28) gives the N*df denominator
/// and the 0.95/0.05 noncentral-CDF endpoints. Derived numerical range checks
/// are fail-closed implementation guards, not scientific acceptance cutoffs.
///
/// # References
/// Maydeu-Olivares, A. (2013). Goodness-of-fit assessment of item response theory
/// models. *Measurement: Interdisciplinary Research and Perspectives, 11*(3),
/// 71–101. https://doi.org/10.1080/15366367.2013.831680
pub fn fit_indices(
    m2: f64,
    df: f64,
    n: usize,
    null_m2: f64,
    null_df: f64,
) -> Result<serde_json::Value, String> {
    if !m2.is_finite()
        || m2 < 0.0
        || !df.is_finite()
        || df <= 0.0
        || n == 0
        || !null_m2.is_finite()
        || null_m2 < 0.0
        || !null_df.is_finite()
        || null_df <= 0.0
    {
        return Err("invalid fit index inputs".into());
    }
    let denom = n as f64 * df;
    if !denom.is_finite() || denom <= 0.0 {
        return Err("nonfinite or nonpositive RMSEA denominator".into());
    }
    let p_value = crate::fitstats::chi2_sf(m2, df);
    let lower_lambda = crate::fitstats::nc_lambda_for(m2, df, 0.95);
    let upper_lambda = crate::fitstats::nc_lambda_for(m2, df, 0.05);
    let rmsea = ((m2 - df).max(0.0) / denom).sqrt();
    let lower = (lower_lambda / denom).sqrt();
    let upper = (upper_lambda / denom).sqrt();
    if !p_value.is_finite()
        || !(0.0..=1.0).contains(&p_value)
        || [lower_lambda, upper_lambda, rmsea, lower, upper]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
        || lower_lambda > upper_lambda
        || lower > upper
    {
        return Err("invalid fit index distribution output".into());
    }
    let (cfi, tli) = crate::fitstats::comparative_fit_metrics(m2, df, null_m2, null_df);
    Ok(
        serde_json::json!({"m2":m2,"df":df,"p_value":p_value,"rmsea":rmsea,"rmsea_ci90_lower":lower,"rmsea_ci90_upper":upper,"cfi":cfi,"tli":tli,"null_m2":null_m2,"null_df":null_df,"distribution_device":"cpu_f64","rmsea_denominator":"N * df","rmsea_denominator_value":denom}),
    )
}

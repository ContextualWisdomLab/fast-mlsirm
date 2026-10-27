//! One-factor grouped GRM coefficient transform and likelihood ACOV delta method.
//! Logistic latent-response residual variance pi²/3; reference mean=0, SD=1.
//!
//! Method scope: Muthén and Asparouhov (2002, pp. 2–5, Eqs. 1–9, 16–19)
//! describe CP/LRV parameterizations; the exact standard-logistic representation
//! used here is derived in `docs/samefit_fifa.md`. No normal-ogive/logit identity
//! is assumed. See the public method comments for verified locators/references.
use crate::samefit_grm::{decode, Coordinate, PackedArchive};
use serde::Serialize;
#[cfg(feature = "gpu")]
#[path = "samefit_fifa_gpu.rs"]
pub mod gpu;
#[derive(Debug, Serialize)]
pub struct Row {
    pub name: String,
    pub estimate: f64,
}
#[derive(Debug, Serialize)]
pub struct Transformation {
    pub rows: Vec<Row>,
    pub jacobian: Vec<f64>,
    pub n_parameters: usize,
}
impl Transformation {
    fn add(&mut self, name: String, estimate: f64, derivative: Vec<f64>) -> Result<(), String> {
        if !estimate.is_finite() || derivative.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite coefficient transform".into());
        }
        self.rows.push(Row { name, estimate });
        self.jacobian.extend(derivative);
        Ok(())
    }
}
/// Standardized group latent-response loading and thresholds are descriptive
/// representations of the SAME fitted logit GRM, not a second factor fit.
/// `threshold_standardized` is the latent-response category boundary at theta=mu,
/// divided by total response SD; its sign is opposite the GRM intercept.
///
/// Muthén and Asparouhov (2002, pp. 2–5, Eqs. 1–9, 16–19) give the
/// conditional-probability/LRV formulation. This implementation declares
/// `L=a*theta+epsilon`, independent standard-logistic epsilon, boundary `-d`;
/// its variance is `(a*sigma)^2+pi²/3`. The logistic variance is given by
/// R Core Team (n.d., Details). Standardization and analytic differentiation
/// are algebra under this convention, not a probit approximation or a new fit.
///
/// # References
/// Muthén, B., & Asparouhov, T. (2002). *Latent variable analysis with categorical
/// outcomes: Multiple-group and growth modeling in Mplus* (Mplus Web Notes
/// No. 4, Version 5). https://statmodel.com/download/webnotes/CatMGLong.pdf
///
/// R Core Team. (n.d.). *The logistic distribution* (R stats documentation,
/// version 4.6.0). https://stat.ethz.ch/R-manual/R-devel/library/stats/html/Logistic.html
pub fn transform(a: &PackedArchive) -> Result<Transformation, String> {
    let d = decode(a)?;
    let coords = a.coordinates()?;
    let p = coords.len();
    let j = a.n_items();
    let z = a.n_categories() - 1;
    let mut out = Transformation {
        rows: vec![],
        jacobian: vec![],
        n_parameters: p,
    };
    let v = std::f64::consts::PI.powi(2) / 3.;
    for g in 0..a.n_groups() {
        let mut m = vec![0.; p];
        let mut s = vec![0.; p];
        for (c, coord) in coords.iter().enumerate() {
            match coord {
                Coordinate::Mean(h) if *h == g => m[c] = 1.,
                Coordinate::LogSd(h) if *h == g => s[c] = d.sds[g],
                _ => (),
            }
        }
        out.add(format!("g{g}.mean"), d.means[g], m.clone())?;
        out.add(format!("g{g}.sd"), d.sds[g], s.clone())?;
        for i in 0..j {
            let ix = g * j + i;
            let a0 = d.slope[ix];
            let sd = d.sds[g];
            let effective = a0 * sd;
            let denom = effective.hypot(v.sqrt());
            let mut da = vec![0.; p];
            let mut dd = vec![vec![0.; p]; z];
            for (c, coord) in coords.iter().enumerate() {
                let owns = |h: usize, item: usize| {
                    item == i && (h == g || (h == 0 && !coords.contains(&Coordinate::Slope(g, i))))
                };
                match *coord {
                    Coordinate::Slope(h, item) if owns(h, item) => da[c] = 1.,
                    Coordinate::FirstIntercept(h, item) if owns(h, item) => {
                        for row in &mut dd {
                            row[c] = 1.;
                        }
                    }
                    Coordinate::LogGap(h, item, k) if owns(h, item) => {
                        for row in dd.iter_mut().skip(k) {
                            row[c] = -f64::from_bits(a.packed_bits()[c]).exp();
                        }
                    }
                    _ => (),
                }
            }
            let de: Vec<_> = (0..p).map(|c| sd * da[c] + a0 * s[c]).collect();
            out.add(format!("g{g}.i{i}.slope_logit"), a0, da.clone())?;
            let dl: Vec<_> = de.iter().map(|x| x * v / denom.powi(3)).collect();
            out.add(
                format!("g{g}.i{i}.lambda_logit_latent_response"),
                effective / denom,
                dl,
            )?;
            for k in 0..z {
                let intercept = d.intercepts[ix * z + k];
                out.add(
                    format!("g{g}.i{i}.intercept[{k}]"),
                    intercept,
                    dd[k].clone(),
                )?;
                if a0.abs() < 1e-12 {
                    return Err("difficulty undefined at zero slope".into());
                }
                out.add(
                    format!("g{g}.i{i}.difficulty[{k}]"),
                    -intercept / a0,
                    (0..p)
                        .map(|c| -dd[k][c] / a0 + intercept * da[c] / a0.powi(2))
                        .collect(),
                )?;
                let centered = intercept + a0 * d.means[g];
                let dc: Vec<_> = (0..p)
                    .map(|c| dd[k][c] + d.means[g] * da[c] + a0 * m[c])
                    .collect();
                let dt: Vec<_> = (0..p)
                    .map(|c| -dc[c] / denom + centered * effective * de[c] / denom.powi(3))
                    .collect();
                out.add(
                    format!("g{g}.i{i}.threshold_standardized[{k}]"),
                    -centered / denom,
                    dt,
                )?;
            }
        }
    }
    Ok(out)
}

/// Multigroup bifactor GRM latent-response FIFA coefficient transform over
/// `bifactor_coordinates`.
///
/// For group `g`, item `i`, general factor `G ~ N(mu_{G,g}, sigma_{G,g}^2)`, and
/// orthogonal specific factor `S_{s(i)} ~ N(mu_{S,g,s}, sigma_{S,g,s}^2)`, the
/// underlying logistic latent response has variance
/// `V_{g,i} = (a_{G,g,i} sigma_{G,g})^2 + (a_{S,g,i} sigma_{S,g,s})^2 + pi^2 / 3`
/// and total SD `D_{g,i} = sqrt(V_{g,i})`. Standardized general/specific loadings
/// and thresholds at the group mean `(mu_{G,g}, mu_{S,g,s})` are differentiated
/// analytically with respect to every coordinate in `a.coordinates()` for delta-method
/// inference via [`infer`] or [`infer_bifactor`].
pub fn transform_bifactor(
    a: &crate::samefit_grm::BifactorPackedArchive,
) -> Result<Transformation, String> {
    use crate::mask_grm::Coordinate as BifactorCoordinate;
    let d = crate::samefit_grm::decode_bifactor(a)?;
    let coords = a.coordinates()?;
    let p = coords.len();
    let j = a.n_items();
    let z = a.n_categories() - 1;
    let n_spec = a.n_specific();
    let spec_map = a.specific_map();
    let common = a.anchor();
    let mut out = Transformation {
        rows: vec![],
        jacobian: vec![],
        n_parameters: p,
    };
    let v = std::f64::consts::PI.powi(2) / 3.;
    for g in 0..a.n_groups() {
        let mut dm_g = vec![0.; p];
        let mut ds_g = vec![0.; p];
        for (c, coord) in coords.iter().enumerate() {
            match *coord {
                BifactorCoordinate::Mean(h) if h == g => dm_g[c] = 1.,
                BifactorCoordinate::LogSd(h) if h == g => ds_g[c] = d.general_sds[g],
                _ => (),
            }
        }
        out.add(format!("g{g}.mean"), d.general_means[g], dm_g.clone())?;
        out.add(format!("g{g}.sd"), d.general_sds[g], ds_g.clone())?;
        let mut dm_s = vec![vec![0.; p]; n_spec];
        let mut ds_s = vec![vec![0.; p]; n_spec];
        for s in 0..n_spec {
            for (c, coord) in coords.iter().enumerate() {
                match *coord {
                    BifactorCoordinate::SpecificMean(h, sp) if h == g && sp == s => {
                        dm_s[s][c] = 1.;
                    }
                    BifactorCoordinate::SpecificLogSd(h, sp) if h == g && sp == s => {
                        ds_s[s][c] = d.specific_sds[g * n_spec + s];
                    }
                    _ => (),
                }
            }
            out.add(
                format!("g{g}.specific[{s}].mean"),
                d.specific_means[g * n_spec + s],
                dm_s[s].clone(),
            )?;
            out.add(
                format!("g{g}.specific[{s}].sd"),
                d.specific_sds[g * n_spec + s],
                ds_s[s].clone(),
            )?;
        }
        for i in 0..j {
            let ix = g * j + i;
            let ag = d.general_slope[ix];
            let aspec = d.specific_slope[ix];
            let sd_g = d.general_sds[g];
            let sp = spec_map[i];
            let (mu_s, sd_s) = if sp >= 0 {
                let s = sp as usize;
                (
                    d.specific_means[g * n_spec + s],
                    d.specific_sds[g * n_spec + s],
                )
            } else {
                (0.0, 1.0)
            };
            let eg = ag * sd_g;
            let es = if sp >= 0 { aspec * sd_s } else { 0.0 };
            let denom = (eg * eg + es * es + v).sqrt();
            let mut dag = vec![0.; p];
            let mut das = vec![0.; p];
            let mut dd = vec![vec![0.; p]; z];
            for (c, coord) in coords.iter().enumerate() {
                let owns = |h: usize, item: usize| item == i && (h == g || common[i]);
                match *coord {
                    BifactorCoordinate::Slope(h, item) if owns(h, item) => dag[c] = 1.,
                    BifactorCoordinate::SpecificSlope(h, item) if owns(h, item) => das[c] = 1.,
                    BifactorCoordinate::FirstIntercept(h, item) if owns(h, item) => {
                        for row in &mut dd {
                            row[c] = 1.;
                        }
                    }
                    BifactorCoordinate::LogGap(h, item, k) if owns(h, item) => {
                        for row in dd.iter_mut().skip(k) {
                            row[c] = -f64::from_bits(a.packed_bits()[c]).exp();
                        }
                    }
                    _ => (),
                }
            }
            let deg: Vec<_> = (0..p).map(|c| sd_g * dag[c] + ag * ds_g[c]).collect();
            let des: Vec<_> = if sp >= 0 {
                let s = sp as usize;
                (0..p).map(|c| sd_s * das[c] + aspec * ds_s[s][c]).collect()
            } else {
                vec![0.; p]
            };
            out.add(format!("g{g}.i{i}.slope_logit"), ag, dag.clone())?;
            let dlg: Vec<_> = (0..p)
                .map(|c| ((es * es + v) * deg[c] - eg * es * des[c]) / denom.powi(3))
                .collect();
            out.add(
                format!("g{g}.i{i}.lambda_logit_latent_response"),
                eg / denom,
                dlg,
            )?;
            if sp >= 0 {
                out.add(
                    format!("g{g}.i{i}.specific_slope_logit"),
                    aspec,
                    das.clone(),
                )?;
                let dls: Vec<_> = (0..p)
                    .map(|c| ((eg * eg + v) * des[c] - eg * es * deg[c]) / denom.powi(3))
                    .collect();
                out.add(
                    format!("g{g}.i{i}.specific_lambda_logit_latent_response"),
                    es / denom,
                    dls,
                )?;
            }
            let dpsi: Vec<_> = (0..p)
                .map(|c| -2.0 * v * (eg * deg[c] + es * des[c]) / denom.powi(4))
                .collect();
            out.add(
                format!("g{g}.i{i}.uniqueness_latent_response"),
                v / (denom * denom),
                dpsi,
            )?;
            for k in 0..z {
                let intercept = d.intercepts[ix * z + k];
                out.add(
                    format!("g{g}.i{i}.intercept[{k}]"),
                    intercept,
                    dd[k].clone(),
                )?;
                if ag.abs() < 1e-12 {
                    return Err("difficulty undefined at zero slope".into());
                }
                out.add(
                    format!("g{g}.i{i}.difficulty[{k}]"),
                    -intercept / ag,
                    (0..p)
                        .map(|c| -dd[k][c] / ag + intercept * dag[c] / ag.powi(2))
                        .collect(),
                )?;
                let centered =
                    intercept + ag * d.general_means[g] + if sp >= 0 { aspec * mu_s } else { 0.0 };
                let dc: Vec<_> = (0..p)
                    .map(|c| {
                        let spec_term = if sp >= 0 {
                            let s = sp as usize;
                            mu_s * das[c] + aspec * dm_s[s][c]
                        } else {
                            0.0
                        };
                        dd[k][c] + d.general_means[g] * dag[c] + ag * dm_g[c] + spec_term
                    })
                    .collect();
                let dt: Vec<_> = (0..p)
                    .map(|c| {
                        -dc[c] / denom + centered * (eg * deg[c] + es * des[c]) / denom.powi(3)
                    })
                    .collect();
                out.add(
                    format!("g{g}.i{i}.threshold_standardized[{k}]"),
                    -centered / denom,
                    dt,
                )?;
            }
        }
    }
    Ok(out)
}

/// Transform multigroup bifactor coordinates and apply observed-information delta SEs.
pub fn infer_bifactor(
    a: &crate::samefit_grm::BifactorPackedArchive,
    information: &[f64],
) -> Result<Inference, String> {
    let t = transform_bifactor(a)?;
    infer(&t, information)
}
#[derive(Debug, Serialize)]
pub struct Inference {
    pub rows: Vec<InferenceRow>,
    pub parameter_covariance: Vec<f64>,
    pub transformed_covariance: Vec<f64>,
    pub inverse_residual_max: f64,
    pub covariance_source: &'static str,
    pub confidence: f64,
}
#[derive(Debug, Serialize)]
pub struct InferenceRow {
    pub name: String,
    pub estimate: f64,
    pub se: f64,
    pub ci95_lower: f64,
    pub ci95_upper: f64,
}
/// Refuse indefiniteness, asymmetry or rank loss rather than pseudoinverse SE.
///
/// First-order `J C J^T` uses the sensitivity-matrix rule of Joint Committee
/// for Guides in Metrology (2011, p. 15, §6.2.1.3, Eq. 3; first-order scope
/// in §5.6.1). `C` is the inverse supplied observed likelihood information;
/// the caller must establish ML regularity and the fixed model/anchor mask.
/// Pointwise normal-Wald intervals use the standard-normal 97.5th percentile
/// (Xu & Long, 2005, pp. 554–556, §7.2, Eq. 2). They may cross coefficient
/// bounds and do not adjust for anchor selection. The inverse residual bound
/// is an implementation check, not a literature-derived scientific threshold.
///
/// # References
/// Joint Committee for Guides in Metrology. (2011). *Evaluation of measurement
/// data—Supplement 2 to the “Guide to the expression of uncertainty in
/// measurement”—Extension to any number of output quantities* (JCGM 102:2011).
/// https://doi.org/10.59161/JCGM102-2011
///
/// Xu, J., & Long, J. S. (2005). Confidence intervals for predicted outcomes in
/// regression models for categorical outcomes. *The Stata Journal, 5*(4), 537–559.
/// https://doi.org/10.1177/1536867X0500500405
pub fn infer(t: &Transformation, information: &[f64]) -> Result<Inference, String> {
    let p = t.n_parameters;
    if p == 0
        || p.checked_mul(p) != Some(information.len())
        || information.iter().any(|x| !x.is_finite())
        || t.rows.iter().any(|r| !r.estimate.is_finite())
    {
        return Err("invalid likelihood information".into());
    }
    for i in 0..p {
        for k in 0..p {
            if (information[i * p + k] - information[k * p + i]).abs()
                > 1e-7 * (1. + information[i * p + k].abs())
            {
                return Err("asymmetric information".into());
            }
        }
    }
    let (passed, min, _) = crate::inference::second_order_test(information, p, 0.)?;
    if !passed || min <= 0. {
        return Err("nonpositive likelihood information".into());
    }
    let cov = crate::inference::vcov_from_hessian(information, p, 0.)?;
    if cov.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite parameter covariance".into());
    }
    if t.jacobian.len() != t.rows.len() * p || t.jacobian.iter().any(|x| !x.is_finite()) {
        return Err("invalid transformation Jacobian".into());
    }
    let mut inverse_residual_max = 0.0_f64;
    for i in 0..p {
        for j in 0..p {
            let product = (0..p)
                .map(|k| information[i * p + k] * cov[k * p + j])
                .sum::<f64>();
            inverse_residual_max =
                inverse_residual_max.max((product - if i == j { 1. } else { 0. }).abs());
        }
    }
    if inverse_residual_max > 1e-8 {
        return Err("information inverse residual failed".into());
    }
    let n = t.rows.len();
    let mut intermediate = vec![0.; n * p];
    for i in 0..n {
        for j in 0..p {
            intermediate[i * p + j] = (0..p).map(|k| t.jacobian[i * p + k] * cov[k * p + j]).sum();
        }
    }
    let mut transformed_covariance = vec![0.; n * n];
    for i in 0..n {
        for j in 0..n {
            transformed_covariance[i * n + j] = (0..p)
                .map(|k| intermediate[i * p + k] * t.jacobian[j * p + k])
                .sum();
        }
    }
    let mut rows = Vec::with_capacity(t.rows.len());
    for (r, row) in t.rows.iter().enumerate() {
        let grad = &t.jacobian[r * p..(r + 1) * p];
        let mut variance = 0.;
        for i in 0..p {
            for k in 0..p {
                variance += grad[i] * cov[i * p + k] * grad[k];
            }
        }
        if !variance.is_finite() || variance < 0. {
            return Err("invalid transformed variance".into());
        }
        let se = variance.sqrt();
        rows.push(InferenceRow {
            name: row.name.clone(),
            estimate: row.estimate,
            se,
            ci95_lower: row.estimate - 1.959963984540054 * se,
            ci95_upper: row.estimate + 1.959963984540054 * se,
        });
    }
    Ok(Inference {
        rows,
        parameter_covariance: cov,
        transformed_covariance,
        inverse_residual_max,
        covariance_source: "inverse_observed_likelihood_information_cpu_f64",
        confidence: 0.95,
    })
}

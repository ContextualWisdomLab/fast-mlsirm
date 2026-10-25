//! One-factor grouped GRM coefficient transform and likelihood ACOV delta method.
//! Logistic latent-response residual variance pi²/3; reference mean=0, SD=1.
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

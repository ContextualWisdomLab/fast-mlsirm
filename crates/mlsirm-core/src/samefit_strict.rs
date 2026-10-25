//! Unregularized covariance whitening and rank-revealing tangent projection.
//! Rank cutoff is machine precision times dimensions; never an adaptive ridge.
//! Quadratic-form basis: Maydeu-Olivares (2013, p. 78, Equations 10–11).
//! Maydeu-Olivares, A. (2013). Goodness-of-fit assessment of item response theory
//! models. Measurement, 11(3), 71–101. https://doi.org/10.1080/15366367.2013.831680
use serde_json::{json, Value};
pub fn project(
    e: &[f64],
    d: &[f64],
    mut x: Vec<f64>,
    s: usize,
    p: usize,
    n: f64,
) -> Result<Value, String> {
    if s == 0
        || e.len() != s
        || s.checked_mul(p) != Some(d.len())
        || s.checked_mul(s) != Some(x.len())
        || !n.is_finite()
        || n <= 0.0
        || e.iter().chain(d).chain(&x).any(|v| !v.is_finite())
    {
        return Err("invalid strict projection input".into());
    }
    let scale = (0..s).map(|i| x[i * s + i].abs()).fold(0., f64::max);
    let mut min_pivot = f64::INFINITY;
    for j in 0..s {
        let pivot = x[j * s + j] - (0..j).map(|k| x[j * s + k] * x[j * s + k]).sum::<f64>();
        if !pivot.is_finite() || pivot <= 0.0 {
            return Err(format!(
                "Xi not positive definite without ridge: pivot={j}, value={pivot:e}"
            ));
        }
        min_pivot = min_pivot.min(pivot);
        x[j * s + j] = pivot.sqrt();
        for i in j + 1..s {
            if (x[i * s + j] - x[j * s + i]).abs() > 1e-12 * scale.max(1.0) {
                return Err("asymmetric Xi".into());
            }
            let v = x[i * s + j] - (0..j).map(|k| x[i * s + k] * x[j * s + k]).sum::<f64>();
            x[i * s + j] = v / x[j * s + j];
        }
    }
    let whiten = |b: &[f64]| {
        let mut v = vec![0.; s];
        for i in 0..s {
            v[i] = (b[i] - (0..i).map(|k| x[i * s + k] * v[k]).sum::<f64>()) / x[i * s + i];
        }
        v
    };
    let mut r = whiten(e);
    let mut cols: Vec<Vec<f64>> = (0..p)
        .map(|c| whiten(&(0..s).map(|i| d[i * p + c]).collect::<Vec<_>>()))
        .collect();
    if r.iter()
        .chain(cols.iter().flatten())
        .any(|v| !v.is_finite())
    {
        return Err("nonfinite whitened projection input".into());
    }
    // Normalize columns so rank does not depend on parameter units.
    for c in &mut cols {
        let norm = c.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !norm.is_finite() {
            return Err("nonfinite tangent norm".into());
        }
        if norm > 0.0 {
            for v in c {
                *v /= norm;
            }
        }
    }
    let tol = f64::EPSILON * (s.max(p) as f64);
    let mut rank = 0;
    let mut pivots = vec![];
    for k in 0..p {
        let best = (k..p)
            .max_by(|&a, &b| {
                cols[a]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f64>()
                    .total_cmp(&cols[b].iter().map(|v| v * v).sum::<f64>())
            })
            .unwrap();
        cols.swap(k, best);
        let norm = cols[k].iter().map(|v| v * v).sum::<f64>().sqrt();
        pivots.push(norm);
        if norm <= tol {
            break;
        }
        for v in &mut cols[k] {
            *v /= norm;
        }
        let q = cols[k].clone();
        for _ in 0..2 {
            let a = r.iter().zip(&q).map(|(a, b)| a * b).sum::<f64>();
            for i in 0..s {
                r[i] -= a * q[i];
            }
            for c in cols.iter_mut().skip(k + 1) {
                let a = c.iter().zip(&q).map(|(a, b)| a * b).sum::<f64>();
                for i in 0..s {
                    c[i] -= a * q[i];
                }
            }
        }
        rank += 1;
    }
    let m2 = n * r.iter().map(|v| v * v).sum::<f64>();
    if rank >= s || !m2.is_finite() {
        return Err("nonpositive df or nonfinite M2".into());
    }
    Ok(
        json!({"m2":m2,"rank":rank,"df":s-rank,"ridge":0.0,"rank_tolerance":tol,"normalized_tangent_pivots":pivots,"xi_min_cholesky_pivot":min_pivot,"xi_max_diagonal":scale}),
    )
}
#[cfg(feature = "gpu")]
pub fn cpu_moments(
    a: &crate::samefit_grm::PackedArchive,
    nodes: &[f64],
    weights: &[f64],
) -> Result<crate::samefit_grm_gpu::GpuMoments, String> {
    let d = crate::samefit_grm::decode(a)?;
    let j = a.n_items();
    let z = a.n_categories() - 1;
    let ng = a.n_groups();
    let cs = crate::samefit_grm_joint::constraints(j, z);
    let s = cs.len();
    let (m, _) = crate::samefit_grm_joint::analytic_moments_delta(a, nodes, weights)?;
    let mut xi = vec![0.; ng * s * s];
    for g in 0..ng {
        let probs: Vec<Vec<f64>> = nodes
            .iter()
            .map(|t| {
                let th = d.means[g] + d.sds[g] * t;
                (0..j * z)
                    .map(|u| {
                        1.0 / (1.0
                            + (-(d.slope[g * j + u / z] * th + d.intercepts[g * j * z + u])).exp())
                    })
                    .collect()
            })
            .collect();
        for r in 0..s {
            for c in 0..=r {
                let mut merged = cs[r].clone();
                for &(i, k) in &cs[c] {
                    if let Some(v) = merged.iter_mut().find(|v| v.0 == i) {
                        v.1 = v.1.max(k);
                    } else {
                        merged.push((i, k));
                    }
                }
                let cross = probs
                    .iter()
                    .zip(weights)
                    .map(|(pr, w)| w * merged.iter().map(|&(i, k)| pr[i * z + k]).product::<f64>())
                    .sum::<f64>();
                let v = cross - m[g * s + r] * m[g * s + c];
                xi[g * s * s + r * s + c] = v;
                xi[g * s * s + c * s + r] = v;
            }
        }
    }
    Ok(crate::samefit_grm_gpu::GpuMoments {
        moments: m,
        xi,
        backend: "CPU_f64_explicit".into(),
        device_name: "CPU".into(),
        dispatches: 0,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn duplicate_tangent() {
        let v = project(
            &[1., 2., 3.],
            &[1., 1., 0., 0., 0., 0.],
            vec![1., 0., 0., 0., 1., 0., 0., 0., 1.],
            3,
            2,
            10.,
        )
        .unwrap();
        assert_eq!(v["rank"], 1);
        assert_eq!(v["df"], 2);
        assert!((v["m2"].as_f64().unwrap() - 130.).abs() < 1e-12);
        println!("duplicate={v}");
    }
    #[test]
    fn singular_covariance_rejected() {
        assert!(project(&[1., 2.], &[], vec![1., 1., 1., 1.], 2, 0, 10.).is_err());
        let old =
            crate::fitstats::projected_m2(&[1., 2.], &[], vec![1., 1., 1., 1.], 2, 0, 10.).unwrap();
        println!("legacy_singular_accepted={old}");
    }
}

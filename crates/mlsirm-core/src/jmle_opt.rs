//! JMLE packed-vector optimizers (Adam and L-BFGS).
//!
//! Public Python `fit` validates, initializes, and marshals; result-affecting
//! Adam moments / L-BFGS history, two-loop recursion, curvature acceptance, and
//! Armijo line search are owned here so production JMLE does not re-implement
//! optimizer arithmetic on the Python side when `backend="rust"`.
//!
//! # References (APA 7th ed.)
//!
//! Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization.
//! In *3rd International Conference on Learning Representations (ICLR)*.
//! https://arxiv.org/abs/1412.6980
//!
//! Nocedal, J., & Wright, S. J. (2006). *Numerical optimization* (2nd ed.).
//! Springer. https://doi.org/10.1007/978-0-387-40065-5
//!
//! Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large
//! scale optimization. *Mathematical Programming, 45*(1–3), 503–528.
//! https://doi.org/10.1007/BF01589116

/// Minimize a packed objective with bias-corrected Adam (Kingma & Ba, 2015).
///
/// `objective` maps a packed parameter vector to `(objective, gradient, loglik)`.
pub fn adam<F>(
    x0: &[f64],
    objective: &mut F,
    learning_rate: f64,
    max_iter: usize,
    tolerance: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, String), String>
where
    F: FnMut(&[f64]) -> Result<(f64, Vec<f64>, f64), String>,
{
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err("learning_rate must be > 0 and finite".into());
    }
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err("tolerance must be > 0 and finite".into());
    }
    if max_iter == 0 {
        return Err("max_iter must be >= 1".into());
    }
    let n = x0.len();
    if n == 0 {
        return Err("parameter vector must be non-empty".into());
    }
    if !x0.iter().all(|v| v.is_finite()) {
        return Err("initial parameter vector must be finite".into());
    }

    let mut x = x0.to_vec();
    let mut m = vec![0.0; n];
    let mut v = vec![0.0; n];
    let beta1: f64 = 0.9;
    let beta2: f64 = 0.999;
    let mut trace: Vec<f64> = Vec::with_capacity(max_iter);
    let mut loglik_trace: Vec<f64> = Vec::with_capacity(max_iter);
    let mut status = "max_iter_reached".to_string();
    let mut prev = f64::INFINITY;

    for t in 1..=max_iter {
        let (obj, grad, loglik) = objective(&x)?;
        if !obj.is_finite() || grad.len() != n || !grad.iter().all(|g| g.is_finite()) {
            return Ok((x, trace, loglik_trace, "nan_or_inf".into()));
        }
        trace.push(obj);
        loglik_trace.push(loglik);
        let denom = prev.abs().max(1.0);
        if (prev - obj).abs() / denom < tolerance {
            status = "converged".into();
            break;
        }
        prev = obj;
        let t_f = t as f64;
        let bc1 = 1.0 - beta1.powf(t_f);
        let bc2 = 1.0 - beta2.powf(t_f);
        for i in 0..n {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * (grad[i] * grad[i]);
            let m_hat = m[i] / bc1;
            let v_hat = v[i] / bc2;
            x[i] -= learning_rate * m_hat / (v_hat.sqrt() + 1e-8);
        }
    }
    Ok((x, trace, loglik_trace, status))
}

/// Minimize a packed objective with limited-memory BFGS and Armijo backtracking.
pub fn lbfgs<F>(
    x0: &[f64],
    objective: &mut F,
    max_iter: usize,
    tolerance: f64,
    history: usize,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, String), String>
where
    F: FnMut(&[f64]) -> Result<(f64, Vec<f64>, f64), String>,
{
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err("tolerance must be > 0 and finite".into());
    }
    if max_iter == 0 {
        return Err("max_iter must be >= 1".into());
    }
    if history == 0 {
        return Err("lbfgs history must be >= 1".into());
    }
    let n = x0.len();
    if n == 0 {
        return Err("parameter vector must be non-empty".into());
    }
    if !x0.iter().all(|v| v.is_finite()) {
        return Err("initial parameter vector must be finite".into());
    }

    let mut x = x0.to_vec();
    let (mut obj, mut grad, mut loglik) = objective(&x)?;
    if !obj.is_finite() || grad.len() != n || !grad.iter().all(|g| g.is_finite()) {
        return Ok((x, vec![], vec![], "nan_or_inf".into()));
    }
    let mut trace = vec![obj];
    let mut loglik_trace = vec![loglik];
    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut rho_hist: Vec<f64> = Vec::new();
    let mut status = "max_iter_reached".to_string();

    for _ in 0..max_iter {
        let grad_norm = l2_norm(&grad);
        if grad_norm < tolerance {
            status = "converged".into();
            break;
        }

        let mut direction = lbfgs_direction(&grad, &s_hist, &y_hist, &rho_hist);
        for d in &mut direction {
            *d = -*d;
        }
        if dot(&grad, &direction) >= 0.0 {
            direction = grad.iter().map(|g| -g).collect();
        }

        let slope = dot(&grad, &direction);
        let mut step = 1.0;
        let mut accepted = false;
        let mut candidate = x.clone();
        let mut next_obj = obj;
        let mut next_grad = grad.clone();
        let mut next_loglik = loglik;
        for _line in 0..20 {
            for i in 0..n {
                candidate[i] = x[i] + step * direction[i];
            }
            let eval = objective(&candidate)?;
            next_obj = eval.0;
            next_grad = eval.1;
            next_loglik = eval.2;
            if next_obj.is_finite()
                && next_grad.len() == n
                && next_grad.iter().all(|g| g.is_finite())
                && next_obj <= obj + 1e-4 * step * slope
            {
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            status = "line_search_failed".into();
            break;
        }

        let mut s = vec![0.0; n];
        let mut y_delta = vec![0.0; n];
        for i in 0..n {
            s[i] = candidate[i] - x[i];
            y_delta[i] = next_grad[i] - grad[i];
        }
        let ys = dot(&y_delta, &s);
        if ys > 1e-12 {
            s_hist.push(s);
            y_hist.push(y_delta);
            rho_hist.push(1.0 / ys);
            if s_hist.len() > history {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
        }

        x = candidate;
        obj = next_obj;
        grad = next_grad;
        loglik = next_loglik;
        trace.push(obj);
        loglik_trace.push(loglik);
    }
    Ok((x, trace, loglik_trace, status))
}

/// Sequence Adam and/or L-BFGS the same way public JMLE Python did historically.
pub fn run_optimizer<F>(
    x0: &[f64],
    objective: &mut F,
    optimizer: &str,
    max_iter: usize,
    learning_rate: f64,
    tolerance: f64,
    lbfgs_history: usize,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, String, usize), String>
where
    F: FnMut(&[f64]) -> Result<(f64, Vec<f64>, f64), String>,
{
    let opt = optimizer.trim().to_ascii_lowercase();
    if !matches!(opt.as_str(), "adam" | "lbfgs" | "adam_lbfgs") {
        return Err(format!("unsupported optimizer: {optimizer}"));
    }
    if max_iter == 0 {
        return Err("max_iter must be >= 1".into());
    }

    let mut x = x0.to_vec();
    let mut obj_trace: Vec<f64> = Vec::new();
    let mut loglik_trace: Vec<f64> = Vec::new();
    let mut status = "max_iter_reached".to_string();
    let mut n_iter = 0usize;

    if opt == "adam" || opt == "adam_lbfgs" {
        let adam_iter = if opt == "adam" {
            max_iter
        } else {
            (max_iter / 2).max(1)
        };
        let (nx, a_obj, a_ll, a_status) =
            adam(&x, objective, learning_rate, adam_iter, tolerance)?;
        x = nx;
        n_iter += a_obj.len();
        obj_trace.extend(a_obj);
        loglik_trace.extend(a_ll);
        status = a_status;
    }

    if opt == "lbfgs" || opt == "adam_lbfgs" {
        let lbfgs_iter = if opt == "lbfgs" {
            max_iter
        } else {
            max_iter.saturating_sub(n_iter).max(1)
        };
        let (nx, l_obj, l_ll, l_status) =
            lbfgs(&x, objective, lbfgs_iter, tolerance, lbfgs_history)?;
        x = nx;
        n_iter += l_obj.len();
        obj_trace.extend(l_obj);
        loglik_trace.extend(l_ll);
        status = l_status;
    }

    Ok((x, obj_trace, loglik_trace, status, n_iter))
}

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn lbfgs_direction(
    grad: &[f64],
    s_hist: &[Vec<f64>],
    y_hist: &[Vec<f64>],
    rho_hist: &[f64],
) -> Vec<f64> {
    let mut q = grad.to_vec();
    let mut alphas: Vec<f64> = Vec::with_capacity(s_hist.len());
    for ((s, y), &rho) in s_hist.iter().rev().zip(y_hist.iter().rev()).zip(rho_hist.iter().rev()) {
        let alpha = rho * dot(s, &q);
        alphas.push(alpha);
        for i in 0..q.len() {
            q[i] -= alpha * y[i];
        }
    }

    if let (Some(s_last), Some(y_last)) = (s_hist.last(), y_hist.last()) {
        let sy = dot(s_last, y_last);
        let yy = dot(y_last, y_last);
        let scale = if yy > 1e-12 { sy / yy } else { 1.0 };
        for qi in &mut q {
            *qi *= scale;
        }
    }

    for (((s, y), &rho), &alpha) in s_hist
        .iter()
        .zip(y_hist.iter())
        .zip(rho_hist.iter())
        .zip(alphas.iter().rev())
    {
        let beta = rho * dot(y, &q);
        for i in 0..q.len() {
            q[i] += s[i] * (alpha - beta);
        }
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quadratic() -> impl FnMut(&[f64]) -> Result<(f64, Vec<f64>, f64), String> {
        |x: &[f64]| {
            // f(x) = 0.5 * ||x - target||^2 with target = [1, -1]
            let t = [1.0, -1.0];
            let mut g = vec![0.0; 2];
            let mut obj = 0.0;
            for i in 0..2 {
                let d = x[i] - t[i];
                obj += 0.5 * d * d;
                g[i] = d;
            }
            Ok((obj, g, -obj))
        }
    }

    #[test]
    fn adam_moves_toward_quadratic_minimum() {
        let mut obj = quadratic();
        let (x, trace, _, status) = adam(&[0.0, 0.0], &mut obj, 0.1, 200, 1e-10).expect("adam");
        assert!(!trace.is_empty());
        assert!(status == "converged" || status == "max_iter_reached");
        assert!((x[0] - 1.0).abs() < 0.05);
        assert!((x[1] + 1.0).abs() < 0.05);
    }

    #[test]
    fn lbfgs_recovers_quadratic_minimum() {
        let mut obj = quadratic();
        let (x, trace, _, status) = lbfgs(&[0.0, 0.0], &mut obj, 50, 1e-10, 10).expect("lbfgs");
        assert!(trace.len() >= 1);
        assert!(status == "converged" || status == "max_iter_reached");
        assert!((x[0] - 1.0).abs() < 1e-4);
        assert!((x[1] + 1.0).abs() < 1e-4);
    }

    #[test]
    fn adam_lbfgs_sequence_runs() {
        let mut obj = quadratic();
        let (x, t, _, status, n_iter) =
            run_optimizer(&[0.0, 0.0], &mut obj, "adam_lbfgs", 40, 0.05, 1e-10, 10)
                .expect("seq");
        assert!(n_iter >= 1);
        assert!(!t.is_empty());
        assert!(status == "converged" || status == "max_iter_reached" || status == "line_search_failed");
        assert!((x[0] - 1.0).abs() < 0.1);
    }

    #[test]
    fn rejects_bad_optimizer_name() {
        let mut obj = quadratic();
        let err = run_optimizer(&[0.0, 0.0], &mut obj, "sgd", 2, 0.01, 1e-6, 5).unwrap_err();
        assert!(err.contains("unsupported"));
    }
}

//! Holm step-down adjusted p-values, in original input order.
//! Candidate library module; not a deployed native API.
pub fn holm_adjusted_p(p_values: &[f64]) -> Result<Vec<f64>, String> {
    for (i, &p) in p_values.iter().enumerate() {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(format!("p_values[{i}] must be finite and in [0,1]"));
        }
    }
    let mut order: Vec<usize> = (0..p_values.len()).collect();
    order.sort_by(|&a, &b| p_values[a].total_cmp(&p_values[b]).then(a.cmp(&b)));
    let mut adjusted = vec![0.0; p_values.len()];
    let mut previous: f64 = 0.0;
    for (rank, &i) in order.iter().enumerate() {
        previous = previous.max(((p_values.len() - rank) as f64 * p_values[i]).min(1.0));
        adjusted[i] = previous;
    }
    Ok(adjusted)
}


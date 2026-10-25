//! GRM E sweep on the device; see `gpu_mstep` for the verified paper contract.
use crate::gpu_mstep::{checked_buffer, dispatch, limits};
pub(crate) struct Expectation {
    pub loglik: f64,
    pub counts: Vec<Vec<Vec<f64>>>,
    pub theta: Vec<f64>,
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn expectation(
    y: &[usize],
    observed: Option<&[bool]>,
    n: usize,
    items: usize,
    dims: usize,
    cat: usize,
    nodes: &[f64],
    logw: &[f64],
    params: &[Vec<f64>],
    dims_of: &[Vec<usize>],
) -> Result<Expectation, String> {
    let q = logw.len();
    let bad = || "GPU E-step dimensions overflow before allocation".to_string();
    if [n, items, dims, cat, q]
        .iter()
        .any(|x| *x == 0 || *x > 16_777_216)
    {
        return Err(bad());
    }
    let mul = |a: usize, b: usize| a.checked_mul(b).ok_or_else(bad);
    let add = |a: usize, b: usize| a.checked_add(b).ok_or_else(bad);
    let node_cells = mul(q, dims)?;
    let response_cells = mul(n, items)?;
    let param_cells = mul(items, add(dims, cat - 1)?)?;
    let count_cells = mul(mul(items, q)?, cat)?;
    let theta_cells = mul(n, dims)?;
    let input = add(
        add(add(add(5, node_cells)?, q)?, param_cells)?,
        response_cells,
    )?;
    let output = add(
        add(add(add(1, count_cells)?, theta_cells)?, q)?,
        count_cells,
    )?;
    let (limit, _) = limits()?;
    checked_buffer(input, limit)?;
    checked_buffer(output, limit)?;
    if nodes.len() != node_cells
        || y.len() != response_cells
        || observed.is_some_and(|o| o.len() != response_cells)
        || params.len() != items
        || dims_of.len() != items
        || cat < 2
    {
        return Err("invalid GPU E-step shape".into());
    }
    if !nodes
        .iter()
        .chain(logw)
        .all(|x| x.is_finite() && (*x as f32).is_finite())
    {
        return Err("nonfinite GPU quadrature".into());
    }
    let mut data = Vec::with_capacity(input);
    data.extend([n as f32, items as f32, dims as f32, cat as f32, q as f32]);
    data.extend(nodes.iter().map(|x| *x as f32));
    data.extend(logw.iter().map(|x| *x as f32));
    for i in 0..items {
        let ds = &dims_of[i];
        let p = &params[i];
        if ds.is_empty()
            || p.len() != ds.len() + cat - 1
            || ds.iter().any(|d| *d >= dims)
            || p.iter().any(|x| !x.is_finite() || !(*x as f32).is_finite())
            || p[ds.len()..]
                .windows(2)
                .any(|x| (x[0] as f32) <= (x[1] as f32))
        {
            return Err("invalid GPU E-step parameters".into());
        }
        for d in 0..dims {
            data.push(ds.iter().position(|v| *v == d).map_or(0.0, |j| p[j] as f32));
        }
        data.extend(p[ds.len()..].iter().map(|x| *x as f32));
    }
    for (cell, &v) in y.iter().enumerate() {
        if observed.is_some_and(|o| !o[cell]) {
            data.push(-1.0);
        } else {
            if v >= cat {
                return Err("invalid observed category".into());
            }
            data.push(v as f32);
        }
    }
    let v = dispatch(include_str!("gpu_grm_estep.wgsl"), &data, output, 1, "main")?;
    let end = 1 + count_cells + theta_cells;
    if v[..end].iter().any(|x| !x.is_finite()) || v[1..1 + count_cells].iter().any(|x| *x < 0.0) {
        return Err("nonfinite GPU E-step result; CPU numerical fallback forbidden".into());
    }
    Ok(Expectation {
        loglik: v[0] as f64,
        counts: v[1..1 + count_cells]
            .chunks(q * cat)
            .map(|item| {
                item.chunks(cat)
                    .map(|row| row.iter().map(|x| *x as f64).collect())
                    .collect()
            })
            .collect(),
        theta: v[1 + count_cells..end].iter().map(|x| *x as f64).collect(),
    })
}

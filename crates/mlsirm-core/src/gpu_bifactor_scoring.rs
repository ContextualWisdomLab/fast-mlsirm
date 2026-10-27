//! Device transport for the existing GRM log-probability and posterior moments.
//! Reuses the canonical two-word arithmetic from gpu_mstep_ds.wgsl.
//! Adjacent boundary gaps are packed separately before boundary words round;
//! narrowing two O(1) boundaries first can lose relative precision in a tiny gap.
//! This is not a full-device fitter: nodes, packing and orchestration stay host-owned.
//!
//! Transport of existing formulas only: Chalmers (2012, p.3, Eq.2) supplies
//! the category contract; Gibbons et al. (2007, pp.9–10, Eqs.17–18) supplies
//! posterior first/second moments, as documented by the parent native fitter.
//! Strict device range refusals propagate, including every category cell even
//! when the caller's integration weight is zero. The f32 reduced posterior is
//! unchanged; two-word final moment arithmetic does not upgrade its precision.
//!
//! Chalmers, R. P. (2012). mirt: A multidimensional item response theory package
//! for the R environment. Journal of Statistical Software, 48(6), 1–29.
//! https://doi.org/10.18637/jss.v048.i06
//! Gibbons, R. D., et al. (2007). Full-information item bifactor analysis of
//! graded response data. Applied Psychological Measurement, 31(1), 4–19.
//! https://doi.org/10.1177/0146621606289485
use crate::bifactor_grm::ItemParams;

fn packed(values: &[f64]) -> Result<Vec<f32>, String> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        let hi = v as f32;
        let lo = (v - hi as f64) as f32;
        if !v.is_finite() || !hi.is_finite() || !lo.is_finite() || (v != 0.0 && hi == 0.0) {
            return Err("GPU scoring value outside finite two-word transport".into());
        }
        out.extend([hi, lo]);
    }
    Ok(out)
}
fn execute(values: &[f64], rows: usize, body: &str) -> Result<Vec<[f64; 2]>, String> {
    let (limit, groups) = crate::gpu_mstep::limits()?;
    if rows == 0 || groups == 0 || values.len() > 16_777_216 {
        return Err("GPU scoring dispatch/index bound".into());
    }
    crate::gpu_mstep::checked_buffer(
        values.len().checked_mul(2).ok_or("scoring overflow")?,
        limit,
    )?;
    crate::gpu_mstep::checked_buffer(rows.checked_mul(6).ok_or("scoring overflow")?, limit)?;
    let (prefix, _) = include_str!("gpu_mstep_ds.wgsl")
        .split_once("fn np()->u32")
        .ok_or("missing canonical arithmetic")?;
    let source=format!("{prefix}\n{body}\n@compute @workgroup_size(1)\nfn scoring(@builtin(workgroup_id) id:vec3<u32>, @builtin(num_workgroups) n:vec3<u32>) {{for(var row=id.x;row<{rows}u;row+=n.x){{score_row(row);}}}}");
    let raw = crate::gpu_mstep::dispatch(
        &source,
        &packed(values)?,
        rows * 6,
        rows.min(groups) as u32,
        "scoring",
    )?;
    raw.chunks_exact(6)
        .map(|r| {
            if r.iter().any(|x| !x.is_finite()) || r[4] != 1.0 || r[5] != 0.0 {
                Err("GPU scoring arithmetic range or invalid moment".into())
            } else {
                Ok([r[0] as f64 + r[1] as f64, r[2] as f64 + r[3] as f64])
            }
        })
        .collect()
}

/// Device category formula, with adjacent gaps packed before boundary rounding.
/// Node order is general-slow/specific-fast, then category-fast.
pub(crate) fn tables(
    params: &[Vec<ItemParams>],
    block: &[Option<usize>],
    tg: &[Vec<f64>],
    ts: &[Vec<Vec<f64>>],
    nc: usize,
) -> Result<Vec<Vec<Vec<f64>>>, String> {
    if params.is_empty()
        || block.is_empty()
        || nc < 2
        || params.len() != tg.len()
        || params.len() != ts.len()
    {
        return Err("invalid GPU table group geometry".into());
    }
    // Validate all group/item geometry before any numerical dispatch.
    let qg = tg[0].len();
    if qg == 0 {
        return Err("empty primary nodes".into());
    }
    for g in 0..params.len() {
        if params[g].len() != block.len()
            || tg[g].len() != qg
            || tg[g].iter().any(|v| !v.is_finite())
        {
            return Err("invalid GPU table row geometry/nodes".into());
        }
        for (i, p) in params[g].iter().enumerate() {
            if p.d.len() != nc - 1
                || p.d.windows(2).any(|w| w[0] <= w[1])
                || p.d.iter().any(|v| !v.is_finite())
                || !p.a_g.is_finite()
                || p.a_s.is_some_and(|v| !v.is_finite())
                || p.a_s.is_some() != block[i].is_some()
            {
                return Err("invalid GPU table parameters".into());
            }
            if let Some(s) = block[i] {
                if s >= ts[g].len()
                    || ts[g][s].is_empty()
                    || ts[g][s].iter().any(|v| !v.is_finite())
                {
                    return Err("invalid GPU specific nodes".into());
                }
            }
        }
    }
    let mut output = Vec::with_capacity(params.len());
    for g in 0..params.len() {
        let mut group = Vec::with_capacity(block.len());
        for (i, p) in params[g].iter().enumerate() {
            let nodes = block[i].map(|s| ts[g][s].as_slice()).unwrap_or(&[0.0]);
            let rows = qg
                .checked_mul(nodes.len())
                .and_then(|n| n.checked_mul(nc))
                .ok_or("table geometry overflow")?;
            let size = 6usize
                .checked_add(p.d.len())
                .and_then(|n| n.checked_add(nc - 2))
                .and_then(|n| n.checked_add(qg))
                .and_then(|n| n.checked_add(nodes.len()))
                .ok_or("table packing overflow")?;
            if size > 16_777_216 || nc > 16_777_216 {
                return Err("table index bound".into());
            }
            let (limit, _) = crate::gpu_mstep::limits()?;
            crate::gpu_mstep::checked_buffer(
                size.checked_mul(2).ok_or("GPU table byte overflow")?,
                limit,
            )?;
            crate::gpu_mstep::checked_buffer(
                rows.checked_mul(6).ok_or("GPU table output overflow")?,
                limit,
            )?;
            let mut values = vec![
                qg as f64,
                nodes.len() as f64,
                nc as f64,
                p.a_g,
                p.a_s.unwrap_or(0.0),
                0.0,
            ];
            values.extend(&p.d);
            // Pack the gap at its own scale before two-word boundary rounding.
            // This is parameter transport, not host predictor/probability work.
            values.extend(p.d.windows(2).map(|d| d[0] - d[1]));
            values.extend(&tg[g]);
            values.extend(nodes);
            let result = execute(
                &values,
                rows,
                r#"
fn score_row(row:u32) {
 arithmetic_range_failure=false;
 let qg=u32(data[0].x);let qs=u32(data[1].x);let nc=u32(data[2].x);
 let i=row;let c=i%nc;let h=(i/nc)%qs;let t=i/(nc*qs);
 let node_start=6u+(nc-1u)+(nc-2u);
 let base=add(mul(data[3],data[node_start+t]),mul(data[4],data[node_start+qg+h]));
 var lp=ds(0.0);
 if(c==0u){lp=ls(-add(base,data[6]));}
 else if(c==nc-1u){lp=ls(add(base,data[6u+nc-2u]));}
 else {let a=data[6u+c-1u];let b=data[6u+c];
  lp=add(add(ls(add(base,a)),ls(-add(base,b))),lg(gap_mass(data[6u+nc-1u+c-1u])));}
 result[3u*i]=lp;result[3u*i+1u]=ds(0.0);
 result[3u*i+2u]=ds(select(0.0,1.0,!arithmetic_range_failure && finite(lp)));
}
"#,
            )?;
            group.push(result.into_iter().map(|r| r[0]).collect());
        }
        output.push(group);
    }
    #[cfg(test)]
    crate::bifactor_grm::scoring_completion_tests::device_table();
    Ok(output)
}

/// Reduce the actual reduced-kernel posterior, not a different score model.
pub(crate) fn eap(
    post: &[f64],
    groups: &[usize],
    nodes: &[Vec<f64>],
) -> Result<Vec<[f64; 2]>, String> {
    if groups.is_empty() || nodes.is_empty() || nodes[0].is_empty() {
        return Err("empty GPU EAP geometry".into());
    }
    let q = nodes[0].len();
    if groups.len().checked_mul(q) != Some(post.len())
        || nodes.iter().any(|r| r.len() != q)
        || groups.iter().any(|g| *g >= nodes.len())
        || post.iter().any(|p| !p.is_finite() || *p < 0.0)
    {
        return Err("invalid GPU EAP posterior geometry/weights".into());
    }
    let cells = 3usize
        .checked_add(groups.len())
        .and_then(|n| n.checked_add(nodes.len().checked_mul(q)?))
        .and_then(|n| n.checked_add(post.len()))
        .ok_or("GPU EAP packing overflow")?;
    if cells > 16_777_216 {
        return Err("GPU EAP packed index bound".into());
    }
    let (limit, _) = crate::gpu_mstep::limits()?;
    crate::gpu_mstep::checked_buffer(cells.checked_mul(2).ok_or("GPU EAP byte overflow")?, limit)?;
    let mut data = vec![q as f64, groups.len() as f64, nodes.len() as f64];
    data.extend(groups.iter().map(|g| *g as f64));
    data.extend(nodes.iter().flatten());
    data.extend(post);
    let result = execute(
        &data,
        groups.len(),
        r#"
fn score_row(row:u32) {
 arithmetic_range_failure=false;let p=row;let q=u32(data[0].x);let np=u32(data[1].x);let ng=u32(data[2].x);
 let g=u32(data[3u+p].x);var mass=ds(0.0);var first=ds(0.0);var second=ds(0.0);
 for(var t=0u;t<q;t++) {let w=data[3u+np+ng*q+p*q+t];let x=data[3u+np+g*q+t];
  mass=add(mass,w);first=add(first,mul(w,x));second=add(second,mul(mul(w,x),x));}
 if(!lt(ds(0.0),mass)){result[3u*p+2u]=ds(0.0);return;}
 let mean=divd(first,mass);let variance=sub(divd(second,mass),mul(mean,mean));
 var sd=ds(0.0);
 if(lt(ds(0.0),variance)){let hi=sqrt(variance.x);sd=add(ds(hi),divd(sub(variance,mul(ds(hi),ds(hi))),ds(2.0*hi)));}
 result[3u*p]=mean;result[3u*p+1u]=sd;
 result[3u*p+2u]=ds(select(0.0,1.0,finite(mean)&&finite(sd)&&!arithmetic_range_failure));
}
"#,
    )?;
    #[cfg(test)]
    crate::bifactor_grm::scoring_completion_tests::device_eap();
    Ok(result)
}

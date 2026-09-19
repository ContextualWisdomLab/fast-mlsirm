//! GPU-parallel E-step for the Bock-Aitkin bifactor GRM with Gibbons-Hedeker
//! dimension reduction ([`crate::bifactor_grm`]).
//!
//! The reduced E-step is person-separable given the log-probability tables:
//! every person contributes a general-node posterior, block joint posteriors,
//! and a marginal log-likelihood term, which are then reduced over persons
//! into expected counts and group moments. This module implements that
//! person sweep in WGSL `f32` (the widest float WebGPU exposes) across six
//! kernels — accumulate, normalize, joint, two count reductions, and group
//! moments — while the `f64` CPU path in [`crate::bifactor_grm`] remains the
//! numerical reference. When no GPU adapter satisfies the binding budget the
//! entry point returns `None` and the caller falls back to CPU.
//!
//! # Precision
//!
//! Kernels accumulate in `f32` (machine epsilon ≈ 1.19e-7). Expected counts
//! are reductions over persons of posterior weights in [0, 1] and the
//! log-likelihood sums per-person terms, so absolute agreement with the CPU
//! path scales with `n_persons`; parity tests assert fit-level agreement
//! derived from that bound (see `tests/test_bifactor_gpu.py`).
//!
//! # Adapter limits
//!
//! Dispatches are factored into `(x, y, z)` against the adapter's runtime
//! `max_compute_workgroups_per_dimension` (65535 on Apple Metal / WebGPU) so
//! large node×item×category grids (e.g. AC late-life bifactor at q=241) do
//! not panic with a validation error. Storage buffers are sized against
//! `max_storage_buffer_binding_size` / `max_buffer_size` and fall back to CPU
//! when they do not fit — no hardcoded workgroup or byte caps.
//!
//! # References
//!
//! - Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E.,
//!   Bhaumik, D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover,
//!   A. (2007). Full-information item bifactor analysis of graded response
//!   data. *Applied Psychological Measurement, 31*(1), 4–19.
//!   https://doi.org/10.1177/0146621606289485
//! - Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor
//!   analysis. *Psychometrika, 57*(3), 423–436.
//!   https://doi.org/10.1007/BF02295430
//! - Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation
//!   of item parameters: Application of an EM algorithm. *Psychometrika,
//!   46*(4), 443–459. https://doi.org/10.1007/BF02293801

#[cfg(all(feature = "gpu", not(coverage)))]
use crate::gpu::GpuContext;
#[cfg(all(feature = "gpu", not(coverage)))]
use wgpu::util::DeviceExt;

/// Inputs for one reduced E-step sweep. `tables_groups[g][i]` holds the
/// log-probability table of group `g`, item `i` (`qg * qs * n_cat` entries
/// for block items with node `(t, h)` at `(t * qs + h) * n_cat`, `qg * n_cat`
/// entries for general-only items with node `t` at `t * n_cat`).
// Only constructed by the wgpu path; kept for CPU-only builds so the E-step
// call sites stay cfg-independent.
#[cfg_attr(any(not(feature = "gpu"), coverage), allow(dead_code))]
pub(crate) struct ReducedEstepInputs<'a> {
    pub y: &'a [usize],
    pub observed: Option<&'a [bool]>,
    /// Person-to-group map; `None` means single-group (all persons in group 0).
    pub group_id: Option<&'a [usize]>,
    pub n_persons: usize,
    pub n_items: usize,
    pub n_specific: usize,
    pub n_cat: usize,
    pub qg: usize,
    pub qs: usize,
    pub n_groups: usize,
    pub tables_groups: &'a [Vec<Vec<f64>>],
    /// Owning specific block per item (`None` = general-only).
    pub item_block: &'a [Option<usize>],
    /// Member item lists per specific block.
    pub blocks: &'a [Vec<usize>],
    /// General-factor nodes per group (`[g][t]`).
    pub tg_groups: &'a [Vec<f64>],
    /// Specific-factor nodes per group and block (`[g][s][h]`).
    pub ts_groups: &'a [Vec<Vec<f64>>],
    pub log_wg: &'a [f64],
    pub log_ws: &'a [f64],
}

/// One reduced E-step sweep: marginal log-likelihood, expected counts with
/// uniform node stride `qg * qs` (`counts[(g * n_items + i) * stride + n)
/// * n_cat + k]`, general-only items using nodes `n = t < qg`), and the
/// group moment accumulators consumed by the multigroup M-step.
#[cfg_attr(any(not(feature = "gpu"), coverage), allow(dead_code))]
pub(crate) struct ReducedEstepOutputs {
    pub loglik: f64,
    pub counts: Vec<f64>,
    pub counts_stride_nodes: usize,
    pub w_acc: Vec<f64>,
    pub s1_g: Vec<f64>,
    pub s2_g: Vec<f64>,
    pub s2_spec: Vec<f64>,
    pub w_spec: Vec<f64>,
}

#[cfg(all(feature = "gpu", not(coverage)))]
const SHADER: &str = "
struct Dims {
    np: u32,
    ni: u32,
    ns: u32,
    nc: u32,
    qg: u32,
    qs: u32,
    ng: u32,
    stride: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> yobs: array<i32>;
@group(0) @binding(2) var<storage, read> gid: array<u32>;
@group(0) @binding(3) var<storage, read> tables: array<f32>;
@group(0) @binding(4) var<storage, read> tab_off: array<u32>;
@group(0) @binding(5) var<storage, read> item_block: array<i32>;
@group(0) @binding(6) var<storage, read> blk_off: array<u32>;
@group(0) @binding(7) var<storage, read> blk_members: array<u32>;
@group(0) @binding(8) var<storage, read> log_wg: array<f32>;
@group(0) @binding(9) var<storage, read> log_ws: array<f32>;
@group(0) @binding(10) var<storage, read> tg: array<f32>;
@group(0) @binding(11) var<storage, read> ts: array<f32>;
@group(0) @binding(12) var<storage, read_write> genlog: array<f32>;
@group(0) @binding(13) var<storage, read_write> logi: array<f32>;
@group(0) @binding(14) var<storage, read_write> blockacc: array<f32>;
@group(0) @binding(15) var<storage, read_write> ll_buf: array<f32>;
@group(0) @binding(16) var<storage, read_write> postg: array<f32>;
@group(0) @binding(17) var<storage, read_write> joint: array<f32>;
@group(0) @binding(18) var<storage, read_write> anyobs: array<u32>;
@group(0) @binding(19) var<storage, read_write> counts: array<f32>;
@group(0) @binding(20) var<storage, read_write> moments: array<f32>;

fn lse_h(base: u32, n: u32) -> f32 {
    var mx = -1e38;
    for (var h = 0u; h < n; h = h + 1u) {
        let v = blockacc[base + h];
        if (v > mx) { mx = v; }
    }
    var s = 0.0;
    for (var h = 0u; h < n; h = h + 1u) {
        s = s + exp(blockacc[base + h] - mx);
    }
    return mx + log(s);
}

// Flatten a 3-D workgroup grid (workgroup_size 64,1,1) into a 1-D invocation
// index so host-side dispatch can split across x/y/z when a single axis would
// exceed the adapter's max_compute_workgroups_per_dimension (Metal/WebGPU).
fn flat_idx(wid: vec3<u32>, lid: u32, nwg: vec3<u32>) -> u32 {
    let wg = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;
    return wg * 64u + lid;
}

// Per (person, general node): general loglik, block integrals, block table.
@compute @workgroup_size(64)
fn accumulate(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = flat_idx(wid, lid, nwg);
    let total = dims.np * dims.qg;
    if (idx >= total) { return; }
    let p = idx / dims.qg;
    let t = idx % dims.qg;
    let g = gid[p];

    var gen = log_wg[t];
    for (var i = 0u; i < dims.ni; i = i + 1u) {
        if (item_block[i] >= 0) { continue; }
        let yc = yobs[p * dims.ni + i];
        if (yc < 0) { continue; }
        let off = tab_off[g * dims.ni + i];
        gen = gen + tables[off + t * dims.nc + u32(yc)];
    }
    genlog[p * dims.qg + t] = gen;

    for (var s = 0u; s < dims.ns; s = s + 1u) {
        var seen = 0u;
        for (var h = 0u; h < dims.qs; h = h + 1u) {
            var acc = log_ws[h];
            for (var m = blk_off[s]; m < blk_off[s + 1u]; m = m + 1u) {
                let i = blk_members[m];
                let yc = yobs[p * dims.ni + i];
                if (yc < 0) { continue; }
                seen = 1u;
                let off = tab_off[g * dims.ni + i];
                acc = acc + tables[off + (t * dims.qs + h) * dims.nc + u32(yc)];
            }
            blockacc[((p * dims.ns + s) * dims.qg + t) * dims.qs + h] = acc;
        }
        anyobs[p * dims.ns + s] = seen;
        logi[(p * dims.ns + s) * dims.qg + t] =
            lse_h(((p * dims.ns + s) * dims.qg + t) * dims.qs, dims.qs);
    }
}

// Per person: marginal loglik and general posterior.
@compute @workgroup_size(64)
fn normalize(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let p = flat_idx(wid, lid, nwg);
    if (p >= dims.np) { return; }
    var mx = -1e38;
    for (var t = 0u; t < dims.qg; t = t + 1u) {
        var acc = genlog[p * dims.qg + t];
        for (var s = 0u; s < dims.ns; s = s + 1u) {
            acc = acc + logi[(p * dims.ns + s) * dims.qg + t];
        }
        if (acc > mx) { mx = acc; }
        postg[p * dims.qg + t] = acc;
    }
    var denom = 0.0;
    for (var t = 0u; t < dims.qg; t = t + 1u) {
        denom = denom + exp(postg[p * dims.qg + t] - mx);
    }
    ll_buf[p] = mx + log(denom);
    for (var t = 0u; t < dims.qg; t = t + 1u) {
        postg[p * dims.qg + t] = exp(postg[p * dims.qg + t] - mx) / denom;
    }
}

// Per (person, block, general node): joint (t, h) posterior.
@compute @workgroup_size(64)
fn joint_post(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = flat_idx(wid, lid, nwg);
    let total = dims.np * dims.ns * dims.qg;
    if (idx >= total) { return; }
    let p = idx / (dims.ns * dims.qg);
    let rem = idx % (dims.ns * dims.qg);
    let s = rem / dims.qg;
    let t = rem % dims.qg;
    let base = ((p * dims.ns + s) * dims.qg + t) * dims.qs;
    let lw = log_wg[t];
    // Exactly-zero Gauss-Hermite mass at large q yields log_w = -inf; then
    // genlog - log_w is NaN and poisons expected counts (#1976). Skip the node.
    // Use a finite f32 threshold (WGSL rejects abstract -1e300 as unrepresentable).
    if (lw != lw || lw < -1e37) {
        for (var h = 0u; h < dims.qs; h = h + 1u) {
            joint[base + h] = 0.0;
        }
        return;
    }
    var others = genlog[p * dims.qg + t] - lw;
    for (var s2 = 0u; s2 < dims.ns; s2 = s2 + 1u) {
        if (s2 != s) {
            others = others + logi[(p * dims.ns + s2) * dims.qg + t];
        }
    }
    for (var h = 0u; h < dims.qs; h = h + 1u) {
        joint[base + h] = exp(lw + blockacc[base + h] + others - ll_buf[p]);
    }
}

// General-only expected counts over (group, item, t, k).
@compute @workgroup_size(64)
fn reduce_counts_gen(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = flat_idx(wid, lid, nwg);
    let total = dims.ng * dims.ni * dims.qg * dims.nc;
    if (idx >= total) { return; }
    let k = idx % dims.nc;
    let t = (idx / dims.nc) % dims.qg;
    let i = (idx / (dims.nc * dims.qg)) % dims.ni;
    let g = idx / (dims.nc * dims.qg * dims.ni);
    let out = ((g * dims.ni + i) * dims.stride + t) * dims.nc + k;
    if (item_block[i] >= 0) {
        counts[out] = 0.0;
        return;
    }
    var sum = 0.0;
    for (var p = 0u; p < dims.np; p = p + 1u) {
        if (gid[p] != g) { continue; }
        if (yobs[p * dims.ni + i] != i32(k)) { continue; }
        sum = sum + postg[p * dims.qg + t];
    }
    counts[out] = sum;
}

// Block-item expected counts over (group, item, t, h, k).
@compute @workgroup_size(64)
fn reduce_counts_blk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = flat_idx(wid, lid, nwg);
    let total = dims.ng * dims.ni * dims.qg * dims.qs * dims.nc;
    if (idx >= total) { return; }
    let k = idx % dims.nc;
    let h = (idx / dims.nc) % dims.qs;
    let t = (idx / (dims.nc * dims.qs)) % dims.qg;
    let i = (idx / (dims.nc * dims.qs * dims.qg)) % dims.ni;
    let g = idx / (dims.nc * dims.qs * dims.qg * dims.ni);
    let out = ((g * dims.ni + i) * dims.stride + t * dims.qs + h) * dims.nc + k;
    let s = item_block[i];
    if (s < 0) {
        counts[out] = 0.0;
        return;
    }
    let su = u32(s);
    var sum = 0.0;
    for (var p = 0u; p < dims.np; p = p + 1u) {
        if (gid[p] != g) { continue; }
        if (anyobs[p * dims.ns + su] == 0u) { continue; }
        if (yobs[p * dims.ni + i] != i32(k)) { continue; }
        sum = sum + joint[((p * dims.ns + su) * dims.qg + t) * dims.qs + h];
    }
    counts[out] = sum;
}

// Per-group general moments (w, s1, s2) packed as 3 + 2*ns floats.
@compute @workgroup_size(64)
fn reduce_moments_g(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let g = flat_idx(wid, lid, nwg);
    if (g >= dims.ng) { return; }
    let row = g * (3u + 2u * dims.ns);
    var w = 0.0;
    var s1 = 0.0;
    var s2 = 0.0;
    for (var p = 0u; p < dims.np; p = p + 1u) {
        if (gid[p] != g) { continue; }
        for (var t = 0u; t < dims.qg; t = t + 1u) {
            let post = postg[p * dims.qg + t];
            let node = tg[g * dims.qg + t];
            w = w + post;
            s1 = s1 + post * node;
            s2 = s2 + post * node * node;
        }
    }
    moments[row] = w;
    moments[row + 1u] = s1;
    moments[row + 2u] = s2;
}

// Per-(group, block) specific moments (w_spec, s2_spec).
@compute @workgroup_size(64)
fn reduce_moments_s(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = flat_idx(wid, lid, nwg);
    let total = dims.ng * dims.ns;
    if (idx >= total) { return; }
    let g = idx / dims.ns;
    let s = idx % dims.ns;
    let row = g * (3u + 2u * dims.ns);
    var w = 0.0;
    var s2 = 0.0;
    for (var p = 0u; p < dims.np; p = p + 1u) {
        if (gid[p] != g) { continue; }
        if (anyobs[p * dims.ns + s] == 0u) { continue; }
        for (var t = 0u; t < dims.qg; t = t + 1u) {
            for (var h = 0u; h < dims.qs; h = h + 1u) {
                let post = joint[((p * dims.ns + s) * dims.qg + t) * dims.qs + h];
                let node = ts[(g * dims.ns + s) * dims.qs + h];
                w = w + post;
                s2 = s2 + post * node * node;
            }
        }
    }
    moments[row + 3u + s] = w;
    moments[row + 3u + dims.ns + s] = s2;
}
";

/// Minimum `max_storage_buffers_per_shader_stage` for the reduced E-step
/// bind group (1 uniform + 11 read-only + 9 read-write buffers).
#[cfg(all(feature = "gpu", not(coverage)))]
const MIN_STORAGE_BUFFERS: u32 = 20;

/// GPU reduced E-step sweep.
///
/// Returns `None` when no compatible GPU adapter can be initialized (or when
/// the `gpu` feature is disabled), signalling the caller to run the CPU
/// implementation. A `None` here is never a silent wrong result: every
/// caller falls back to the `f64` CPU sweep over the same tables.
#[cfg(all(feature = "gpu", not(coverage)))]
pub(crate) fn e_step_reduced_gpu(inputs: &ReducedEstepInputs) -> Option<ReducedEstepOutputs> {
    use crate::gpu::{
        dispatch_count, dispatch_workgroups_nd, output_buffer, staging_buffer, storage_buffer_fits,
        storage_entry, submit_and_readback,
    };

    let ctx = GpuContext::get()?;
    if ctx.adapter_storage_buffers() < MIN_STORAGE_BUFFERS {
        return None;
    }
    let device = &ctx.device;
    let limits = device.limits();
    let max_wg = limits.max_compute_workgroups_per_dimension;

    let np = inputs.n_persons;
    let ni = inputs.n_items;
    let ns = inputs.n_specific;
    let nc = inputs.n_cat;
    let qg = inputs.qg;
    let qs = inputs.qs;
    let ng = inputs.n_groups;
    let stride = qg * qs;

    // Fail closed on storage binding budget before allocating (Metal/WebGPU
    // report these at runtime; never hardcode a byte cap).
    let buffer_lens = [
        np * ni,                 // yobs as i32 — sized separately below
        np * qg,                 // genlog / postg
        np * ns * qg,            // logi
        np * ns * qg * qs,       // blockacc / joint
        np,                      // ll
        np * ns,                 // anyobs
        ng * ni * stride * nc,   // counts
        ng * (3 + 2 * ns),       // moments
    ];
    for &len in &buffer_lens[1..] {
        if !storage_buffer_fits(&limits, len) {
            return None;
        }
    }
    // yobs is i32; reuse the f32-sized check with equal element width.
    if !storage_buffer_fits(&limits, buffer_lens[0]) {
        return None;
    }

    // y/observed packed as i32 (-1 = missing).
    let mut yobs = vec![-1i32; np * ni];
    for p in 0..np {
        for i in 0..ni {
            let obs = inputs.observed.map_or(true, |o| o[p * ni + i]);
            if obs {
                yobs[p * ni + i] = inputs.y[p * ni + i] as i32;
            }
        }
    }
    let mut gid = vec![0u32; np];
    if let Some(g) = inputs.group_id {
        for (p, &v) in g.iter().enumerate() {
            gid[p] = v as u32;
        }
    }

    // Flattened logprob tables with per-(group, item) offsets.
    let mut tab_off = vec![0u32; ng * ni];
    let mut tables: Vec<f32> = Vec::new();
    for g in 0..ng {
        for i in 0..ni {
            tab_off[g * ni + i] = tables.len() as u32;
            tables.extend(inputs.tables_groups[g][i].iter().map(|&v| v as f32));
        }
    }
    if !storage_buffer_fits(&limits, tables.len()) {
        return None;
    }
    let mut block_of = vec![-1i32; ni];
    for (i, b) in inputs.item_block.iter().enumerate() {
        block_of[i] = b.map_or(-1, |s| s as i32);
    }
    let mut blk_off = vec![0u32; ns + 1];
    let mut blk_members: Vec<u32> = Vec::new();
    for (s, members) in inputs.blocks.iter().enumerate() {
        blk_off[s] = blk_members.len() as u32;
        blk_members.extend(members.iter().map(|&i| i as u32));
    }
    blk_off[ns] = blk_members.len() as u32;

    let tg: Vec<f32> = inputs
        .tg_groups
        .iter()
        .flat_map(|v| v.iter().map(|&x| x as f32))
        .collect();
    let ts: Vec<f32> = inputs
        .ts_groups
        .iter()
        .flat_map(|vv| vv.iter().flat_map(|v| v.iter().map(|&x| x as f32)))
        .collect();
    let log_wg: Vec<f32> = inputs.log_wg.iter().map(|&x| x as f32).collect();
    let log_ws: Vec<f32> = inputs.log_ws.iter().map(|&x| x as f32).collect();

    let dims: [u32; 8] = [
        np as u32,
        ni as u32,
        ns as u32,
        nc as u32,
        qg as u32,
        qs as u32,
        ng as u32,
        stride as u32,
    ];

    let mk_init = |label: &str, bytes: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dims_buf"),
        contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let yobs_buf = mk_init("yobs", bytemuck::cast_slice(&yobs));
    let gid_buf = mk_init("gid", bytemuck::cast_slice(&gid));
    let tables_buf = mk_init("tables", bytemuck::cast_slice(&tables));
    let tab_off_buf = mk_init("tab_off", bytemuck::cast_slice(&tab_off));
    let block_buf = mk_init("item_block", bytemuck::cast_slice(&block_of));
    let blk_off_buf = mk_init("blk_off", bytemuck::cast_slice(&blk_off));
    let blk_members_buf = mk_init("blk_members", bytemuck::cast_slice(&blk_members));
    let log_wg_buf = mk_init("log_wg", bytemuck::cast_slice(&log_wg));
    let log_ws_buf = mk_init("log_ws", bytemuck::cast_slice(&log_ws));
    let tg_buf = mk_init("tg", bytemuck::cast_slice(&tg));
    let ts_buf = mk_init("ts", bytemuck::cast_slice(&ts));

    let genlog_buf = output_buffer(device, "genlog", np * qg);
    let logi_buf = output_buffer(device, "logi", np * ns * qg);
    let blockacc_buf = output_buffer(device, "blockacc", np * ns * qg * qs);
    let ll_buf = output_buffer(device, "ll", np);
    let postg_buf = output_buffer(device, "postg", np * qg);
    let joint_buf = output_buffer(device, "joint", np * ns * qg * qs);
    let anyobs_buf = output_buffer(device, "anyobs", np * ns);
    let counts_buf = output_buffer(device, "counts", ng * ni * stride * nc);
    let moments_buf = output_buffer(device, "moments", ng * (3 + 2 * ns));

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bifactor_reduced_estep"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let mut entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }];
    for binding in 1..=11u32 {
        entries.push(storage_entry(binding, true));
    }
    for binding in 12..=20u32 {
        entries.push(storage_entry(binding, false));
    }
    let bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reduced_estep_bgl"),
            entries: &entries,
        });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reduced_estep_bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dims_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: yobs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: gid_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: tables_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: tab_off_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: block_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: blk_off_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: blk_members_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: log_wg_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: log_ws_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: tg_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: ts_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 12,
                resource: genlog_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 13,
                resource: logi_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 14,
                resource: blockacc_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 15,
                resource: ll_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 16,
                resource: postg_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 17,
                resource: joint_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 18,
                resource: anyobs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 19,
                resource: counts_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 20,
                resource: moments_buf.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("reduced_estep_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let make = |entry_point: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    };
    let pl_acc = make("accumulate");
    let pl_norm = make("normalize");
    let pl_joint = make("joint_post");
    let pl_cgen = make("reduce_counts_gen");
    let pl_cblk = make("reduce_counts_blk");
    let pl_mg = make("reduce_moments_g");
    let pl_ms = make("reduce_moments_s");

    // One compute pass per kernel so storage writes are visible downstream.
    // Factor each 1-D workgroup count into x/y/z against the adapter's
    // max_compute_workgroups_per_dimension (65535 on Apple Metal) so study-
    // scale q×item×category grids (e.g. AC late-life q=241) do not panic.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    for (pipeline, groups) in [
        (&pl_acc, dispatch_count(np * qg)),
        (&pl_norm, dispatch_count(np)),
        (&pl_joint, dispatch_count(np * ns * qg)),
        (&pl_cgen, dispatch_count(ng * ni * qg * nc)),
        (&pl_cblk, dispatch_count(ng * ni * qg * qs * nc)),
        (&pl_mg, dispatch_count(ng)),
        (&pl_ms, dispatch_count(ng * ns)),
    ] {
        let (dx, dy, dz) = dispatch_workgroups_nd(groups.max(1), max_wg)?;
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.set_pipeline(pipeline);
        cpass.dispatch_workgroups(dx, dy, dz);
    }

    let ll_staging = staging_buffer(device, "ll_read", np);
    let counts_staging = staging_buffer(device, "counts_read", ng * ni * stride * nc);
    let moments_staging = staging_buffer(device, "moments_read", ng * (3 + 2 * ns));
    let read = submit_and_readback(
        ctx,
        encoder,
        &[
            (&ll_buf, &ll_staging, np),
            (&counts_buf, &counts_staging, ng * ni * stride * nc),
            (&moments_buf, &moments_staging, ng * (3 + 2 * ns)),
        ],
    )?;
    let mut iter = read.into_iter();
    let ll_vec = iter.next()?;
    let counts_vec = iter.next()?;
    let moments_vec = iter.next()?;

    let mut loglik = 0.0;
    for &v in &ll_vec {
        loglik += f64::from(v);
    }
    let row = 3 + 2 * ns;
    let mut w_acc = vec![0.0; ng];
    let mut s1_g = vec![0.0; ng];
    let mut s2_g = vec![0.0; ng];
    let mut s2_spec = vec![0.0; ng * ns];
    let mut w_spec = vec![0.0; ng * ns];
    for g in 0..ng {
        w_acc[g] = f64::from(moments_vec[g * row]);
        s1_g[g] = f64::from(moments_vec[g * row + 1]);
        s2_g[g] = f64::from(moments_vec[g * row + 2]);
        for s in 0..ns {
            w_spec[g * ns + s] = f64::from(moments_vec[g * row + 3 + s]);
            s2_spec[g * ns + s] = f64::from(moments_vec[g * row + 3 + ns + s]);
        }
    }

    Some(ReducedEstepOutputs {
        loglik,
        counts: counts_vec.into_iter().map(f64::from).collect(),
        counts_stride_nodes: stride,
        w_acc,
        s1_g,
        s2_g,
        s2_spec,
        w_spec,
    })
}

/// CPU-fallback stub used when the `gpu` feature is disabled or under
/// coverage: always returns `None` so the caller runs the CPU E-step.
#[cfg(any(not(feature = "gpu"), coverage))]
#[allow(dead_code)]
pub(crate) fn e_step_reduced_gpu(
    _inputs: &ReducedEstepInputs,
) -> Option<ReducedEstepOutputs> {
    None
}

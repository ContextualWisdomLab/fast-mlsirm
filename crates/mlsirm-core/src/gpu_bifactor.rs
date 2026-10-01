//! GPU-parallel E-step for the Bock-Aitkin bifactor GRM with Gibbons-Hedeker
//! dimension reduction ([`crate::bifactor_grm`]).
//!
//! The reduced E-step is person-separable given the log-probability tables:
//! every person contributes a general-node posterior, block joint posteriors,
//! and a marginal log-likelihood term, which are then reduced over persons
//! into expected counts and group moments. This module implements that
//! person sweep in WGSL `f32` (the widest float WebGPU exposes) across six
//! kernels — accumulate, normalize, joint, two count reductions, and specific
//! moments — while the `f64` CPU path in [`crate::bifactor_grm`] remains the
//! numerical reference. When no GPU adapter satisfies the binding budget the
//! entry point returns `None` and the caller falls back to CPU.
//!
//! # Precision
//!
//! Kernels compute posterior weights and expected counts in WGSL `f32`;
//! general-factor group moments sum those weights against the original nodes
//! in host `f64`. Expected-count and posterior rounding still limit agreement
//! with the `f64` CPU path; fit-level parity needs direct tests (see
//! `tests/test_bifactor_gpu.py`).
//!
//! # Adapter limits
//!
//! Dispatches are factored into `(x, y, z)` against the adapter's runtime
//! `max_compute_workgroups_per_dimension` (65535 on Apple Metal / WebGPU) so
//! large node×item×category grids (e.g. AC late-life bifactor at q=241) do
//! not panic with a validation error. Storage buffers are sized against
//! `max_storage_buffer_binding_size` / `max_buffer_size`. When they do not fit,
//! explicit `Gpu` fits fail; `Auto` may use CPU. No hardcoded byte caps.
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
//! - Higham, N. J. (1993). The accuracy of floating point summation.
//!   *SIAM Journal on Scientific Computing, 14*(4), 783–799, p. 785, eq. (2.6).
//!   https://doi.org/10.1137/0914050
//! - W3C GPU for the Web Working Group. WebGPU Shading Language, §6.2.
//!   https://www.w3.org/TR/WGSL/#floating-point-types

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
    /// Optional normalized posterior arrays; see Cai (2010), pp.608-609.
    pub person_posteriors: Option<ReducedPersonPosteriors>,
    pub counts: Vec<f64>,
    pub counts_stride_nodes: usize,
    pub w_acc: Vec<f64>,
    pub s1_g: Vec<f64>,
    pub s2_g: Vec<f64>,
    pub s2_spec: Vec<f64>,
    pub w_spec: Vec<f64>,
}

/// Normalized GPU posterior weights for arbitrary shared product-grid nodes.
/// Cai (2010), pp.608-609, Appendices A/B, DOI 10.1007/s11336-010-9178-0.
/// `primary[(person * qg) + node]`;
/// `joint[((person * ns + block) * qg + node) * qs + specific_node]`.
/// Contract these with every primary
/// coordinate; the shared-node index does not imply a single primary factor.
/// Weights are computed in WGSL f32 and widened, not recomputed in f64:
/// https://www.w3.org/TR/WGSL/#floating-point-types. Numerical parity and
/// integration sensitivity must be checked independently before reporting.
#[cfg_attr(any(not(feature = "gpu"), coverage), allow(dead_code))]
pub(crate) struct ReducedPersonPosteriors {
    pub primary: Vec<f64>,
    pub joint: Vec<f64>,
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
/// Returns `None` when GPU execution is unavailable. Bifactor callers reject
/// `None` for explicit `Gpu` and may use the `f64` CPU sweep for `Auto`.
/// wgpu 30.0.0, `DeviceType`, classifies CPU adapters as software rendering:
/// https://docs.rs/wgpu/30.0.0/wgpu/enum.DeviceType.html
#[cfg(all(feature = "gpu", not(coverage)))]
pub(crate) fn e_step_reduced_gpu(inputs: &ReducedEstepInputs) -> Option<ReducedEstepOutputs> {
    e_step_reduced_gpu_inner(inputs, false)
}

/// Return individual primary/joint posteriors using the existing GPU kernels.
/// Cai (2010), pp.608-609, Appendices A/B; WGSL f32 precision limits are
/// documented on `ReducedPersonPosteriors`. This route omits item-count and
/// group-moment reductions; those quantities are not valid P>1 moments.
/// Uses adapter buffer/workgroup budgets from wgpu 30.0.0 Limits:
/// https://docs.rs/wgpu/30.0.0/wgpu/struct.Limits.html.
/// `None` means GPU execution/readback failed, never a CPU result.
#[cfg(all(feature = "gpu", not(coverage)))]
pub(crate) fn e_step_reduced_gpu_posteriors(
    inputs: &ReducedEstepInputs,
) -> Option<ReducedEstepOutputs> {
    e_step_reduced_gpu_inner(inputs, true)
}

/// Dispatch the reduced probability products from Cai (2010), pp.608-609.
/// For S=0 use the primary-only conditional probabilities (p.587 equation7,
/// p.588 optional-specific passage, p.589 equations11-12). Empty specific
/// arrays receive unread minimum bindings and no empty readback; shader ns
/// remains zero, so this does not introduce an auxiliary latent dimension.
///
/// References: Cai, L. (2010). A two-tier full-information item factor
/// analysis model with applications. Psychometrika, 75(4), 581-612.
/// https://doi.org/10.1007/s11336-010-9178-0 . gfx-rs Developers. (n.d.).
/// wgpu-core (Version 30.0.0), src/binding_model.rs, BindingZeroSize.
/// https://crates.io/crates/wgpu-core/30.0.0 (installed primary source read).
#[cfg(all(feature = "gpu", not(coverage)))]
fn e_step_reduced_gpu_inner(
    inputs: &ReducedEstepInputs,
    collect_posteriors: bool,
) -> Option<ReducedEstepOutputs> {
    use crate::gpu::{
        dispatch_count, dispatch_workgroups_nd, output_buffer, staging_buffer, storage_buffer_fits,
        storage_entry, submit_and_readback,
    };

    let ctx = GpuContext::get()?;
    if ctx.adapter_storage_buffers() < MIN_STORAGE_BUFFERS {
        return None;
    }
    // wgpu 30.0.0 DeviceType manual calls Cpu software rendering and
    // distinguishes integrated, discrete, and virtual GPUs. A software or
    // unknown adapter cannot establish that this E-step used GPU hardware.
    // https://docs.rs/wgpu/30.0.0/wgpu/enum.DeviceType.html
    if !matches!(
        ctx.adapter_info.device_type,
        wgpu::DeviceType::IntegratedGpu
            | wgpu::DeviceType::DiscreteGpu
            | wgpu::DeviceType::VirtualGpu
    ) {
        return None;
    }
    let device = &ctx.device;
    let limits = device.limits();
    let max_wg = limits.max_compute_workgroups_per_dimension;
    // wgpu 30 Device::push_error_scope captures allocation/dispatch errors;
    // its thread-local RAII guards also pop safely on early Option returns.
    // https://docs.rs/wgpu/30.0.0/wgpu/struct.Device.html#method.push_error_scope
    let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);

    let np = inputs.n_persons;
    let ni = inputs.n_items;
    let ns = inputs.n_specific;
    let nc = inputs.n_cat;
    let qg = inputs.qg;
    let qs = inputs.qs;
    let ng = inputs.n_groups;
    let stride = qg * qs;
    // Retain one-element bindings untouched by the posterior kernels
    // (Cai, Appendix B); wgpu 30 device buffers must be nonempty.
    // https://docs.rs/wgpu/30.0.0/wgpu/struct.Device.html#method.create_buffer_from_hal
    let counts_len = if collect_posteriors {
        1
    } else {
        ng * ni * stride * nc
    };
    let moments_len = if collect_posteriors {
        1
    } else {
        ng * (3 + 2 * ns)
    };

    // Fail closed on storage binding budget before allocating (Metal/WebGPU
    // report these at runtime; never hardcode a byte cap).
    let buffer_lens = [
        np * ni,           // yobs as i32 — sized separately below
        np * qg,           // genlog / postg
        np * ns * qg,      // logi
        np * ns * qg * qs, // blockacc / joint
        np,                // ll
        np * ns,           // anyobs
        counts_len,        // counts (unused posterior binding: one element)
        moments_len,       // moments (unused posterior binding: one element)
    ];
    for &len in &buffer_lens[1..] {
        if !storage_buffer_fits(&limits, len.max(1)) {
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
            // Empty specific-factor arrays have no mathematical entries. Keep
            // an unread 4-byte binding; WGSL loops still use dims.ns=0.
            // wgpu-core 30 binding_model.rs rejects zero-size bindings.
            contents: if bytes.is_empty() { &[0u8; 4] } else { bytes },
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
    let logi_buf = output_buffer(device, "logi", (np * ns * qg).max(1));
    let blockacc_buf = output_buffer(device, "blockacc", (np * ns * qg * qs).max(1));
    let ll_buf = output_buffer(device, "ll", np);
    let postg_buf = output_buffer(device, "postg", np * qg);
    let joint_buf = output_buffer(device, "joint", (np * ns * qg * qs).max(1));
    let anyobs_buf = output_buffer(device, "anyobs", (np * ns).max(1));
    let counts_buf = output_buffer(device, "counts", counts_len);
    let moments_buf = output_buffer(device, "moments", moments_len);

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
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
    let pl_ms = make("reduce_moments_s");

    // One compute pass per kernel so storage writes are visible downstream.
    // Factor each 1-D workgroup count into x/y/z against the adapter's
    // max_compute_workgroups_per_dimension (65535 on Apple Metal) so study-
    // scale q×item×category grids (e.g. AC late-life q=241) do not panic.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    for (index, (pipeline, groups)) in [
        (&pl_acc, dispatch_count(np * qg)),
        (&pl_norm, dispatch_count(np)),
        (&pl_joint, dispatch_count(np * ns * qg)),
        (&pl_cgen, dispatch_count(ng * ni * qg * nc)),
        (&pl_cblk, dispatch_count(ng * ni * qg * qs * nc)),
        (&pl_ms, dispatch_count(ng * ns)),
    ]
    .into_iter()
    .enumerate()
    {
        if collect_posteriors && index >= 3 {
            continue;
        }
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
    let counts_staging =
        (!collect_posteriors).then(|| staging_buffer(device, "counts_read", counts_len));
    let moments_staging =
        (!collect_posteriors).then(|| staging_buffer(device, "moments_read", moments_len));
    let primary_staging = staging_buffer(device, "primary_read", np * qg);
    let joint_staging = (collect_posteriors && ns > 0)
        .then(|| staging_buffer(device, "joint_read", np * ns * qg * qs));
    let mut copies = vec![
        (&ll_buf, &ll_staging, np),
        (&postg_buf, &primary_staging, np * qg),
    ];
    if collect_posteriors {
        if ns > 0 {
            copies.push((&joint_buf, joint_staging.as_ref()?, np * ns * qg * qs));
        }
    } else {
        copies.push((&counts_buf, counts_staging.as_ref()?, counts_len));
        copies.push((&moments_buf, moments_staging.as_ref()?, moments_len));
    }
    let read = submit_and_readback(ctx, encoder, &copies)?;
    let mut iter = read.into_iter();
    let ll_vec = iter.next()?;
    let primary_vec = iter.next()?;
    let (counts_vec, moments_vec, person_posteriors) = if collect_posteriors {
        let primary = primary_vec.iter().copied().map(f64::from).collect();
        let joint = if ns == 0 {
            Vec::new()
        } else {
            iter.next()?.into_iter().map(f64::from).collect()
        };
        (
            Vec::new(),
            vec![0.0; ng * (3 + 2 * ns)],
            Some(ReducedPersonPosteriors { primary, joint }),
        )
    } else {
        (iter.next()?, iter.next()?, None)
    };

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
    // Reduce GPU posterior weights against the original f64 nodes. Bock and
    // Aitkin (1981, p. 448, eqs. (13)–(14)) derive a posterior-weighted node mean;
    // summing in f64 lowers the unit-roundoff term in Higham's recursive-sum
    // bound (1993, p. 785, eq. (2.6)); posterior weights remain f32.
    if !collect_posteriors {
        for (p, &group) in gid.iter().enumerate() {
            let g = group as usize;
            for t in 0..qg {
                let post = f64::from(primary_vec[p * qg + t]);
                let node = inputs.tg_groups[g][t];
                w_acc[g] += post;
                s1_g[g] += post * node;
                s2_g[g] += post * node * node;
            }
        }
    }
    for g in 0..ng {
        for s in 0..ns {
            w_spec[g * ns + s] = f64::from(moments_vec[g * row + 3 + s]);
            s2_spec[g * ns + s] = f64::from(moments_vec[g * row + 3 + ns + s]);
        }
    }

    let internal_error = pollster::block_on(internal.pop());
    let validation_error = pollster::block_on(validation.pop());
    let allocation_error = pollster::block_on(oom.pop());
    if internal_error.is_some() || validation_error.is_some() || allocation_error.is_some() {
        return None;
    }
    Some(ReducedEstepOutputs {
        loglik,
        person_posteriors,
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
pub(crate) fn e_step_reduced_gpu(_inputs: &ReducedEstepInputs) -> Option<ReducedEstepOutputs> {
    None
}

/// Unavailable GPU route in builds without GPU support. See GPU counterpart's
/// Cai (2010), Appendices A/B and wgpu Limits contract; no CPU substitution.
#[cfg(any(not(feature = "gpu"), coverage))]
#[allow(dead_code)]
pub(crate) fn e_step_reduced_gpu_posteriors(
    _inputs: &ReducedEstepInputs,
) -> Option<ReducedEstepOutputs> {
    None
}

/// G1의 비양수 log product 덧셈을 binary64 word로 보존한다.
/// Cai (2010), pp.608–609 Appendices A/B의 곱과 적분은 그대로이며,
/// WGSL u32 연산으로 같은 부호 binary64 덧셈의 guard/round/sticky와
/// round-to-nearest-even을 구현한다. GPU native f64라고 주장하지 않는다.
/// 정규화/exp/log/count contraction은 실제 GPU product를 받아 Rust f64로 수행한다.
/// WGSL §15.7.2/15.7.5는 f32 rounding/reassociation/FTZ를 허용하므로
/// f32 보상합만으로 정밀도를 보장하지 않는다: https://www.w3.org/TR/WGSL/.
/// 덧셈 구현은 IEEE binary64 형식에서 도출했으며 GPU bit parity로 검증한다.
/// 참고: Cai, L. (2010). A two-tier full-information item factor analysis
/// model with applications. Psychometrika, 75(4), 581–612.
/// https://doi.org/10.1007/s11336-010-9178-0.
#[cfg(all(feature = "gpu", not(coverage)))]
const LOG_PRODUCT_SHADER: &str = r#"
struct Dims { np:u32, ni:u32, ns:u32, nc:u32, qg:u32, qs:u32, pad0:u32, pad1:u32 }
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> yobs: array<i32>;
@group(0) @binding(2) var<storage, read> tables: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read> block_of: array<i32>;
@group(0) @binding(5) var<storage, read> block_offsets: array<u32>;
@group(0) @binding(6) var<storage, read> members: array<u32>;
@group(0) @binding(7) var<storage, read> prior_g: array<vec2<u32>>;
@group(0) @binding(8) var<storage, read> prior_s: array<vec2<u32>>;
@group(0) @binding(9) var<storage, read_write> general: array<vec2<u32>>;
@group(0) @binding(10) var<storage, read_write> block: array<vec2<u32>>;
fn shift_jam(v:vec2<u32>, n:u32) -> vec2<u32> {
    if (n==0u) { return v; }
    if (n<32u) {
        let lost=(v.x & ((1u<<n)-1u))!=0u;
        return vec2((v.x>>n)|(v.y<<(32u-n))|select(0u,1u,lost),v.y>>n);
    }
    if (n==32u) { return vec2(v.y|select(0u,1u,v.x!=0u),0u); }
    if (n<64u) {
        let k=n-32u;
        let lost=v.x!=0u || (v.y & ((1u<<k)-1u))!=0u;
        return vec2((v.y>>k)|select(0u,1u,lost),0u);
    }
    return vec2(select(0u,1u,(v.x|v.y)!=0u),0u);
}
fn add_negative(a:vec2<u32>, b:vec2<u32>) -> vec2<u32> {
    // 모델의 log(0)은 흡수 원소다. f32 inf 연산에 맡기지 않고 bit로 보존한다.
    if ((a.y==0xfff00000u && a.x==0u) || (b.y==0xfff00000u && b.x==0u)) {
        return vec2(0u,0xfff00000u);
    }
    let ae=(a.y>>20u)&2047u;
    let be=(b.y>>20u)&2047u;
    var ea=max(ae,1u); var eb=max(be,1u);
    let ah=(a.y&0xfffffu)|select(0u,0x100000u,ae!=0u);
    let bh=(b.y&0xfffffu)|select(0u,0x100000u,be!=0u);
    var ma=vec2(a.x<<3u,(ah<<3u)|(a.x>>29u));
    var mb=vec2(b.x<<3u,(bh<<3u)|(b.x>>29u));
    if (eb>ea) {
        let te=ea;ea=eb;eb=te;
        let tm=ma;ma=mb;mb=tm;
    }
    mb=shift_jam(mb,ea-eb);
    let lo=ma.x+mb.x;
    var sum=vec2(lo,ma.y+mb.y+select(0u,1u,lo<ma.x));
    if ((sum.y&0x1000000u)!=0u) { sum=shift_jam(sum,1u);ea+=1u; }
    let rounding=sum.x&7u;
    var sig=vec2((sum.x>>3u)|(sum.y<<29u),sum.y>>3u);
    if (rounding>4u || (rounding==4u && (sig.x&1u)!=0u)) {
        let sl=sig.x+1u;
        sig=vec2(sl,sig.y+select(0u,1u,sl==0u));
    }
    if ((sig.y&0x200000u)!=0u) { sig=shift_jam(sig,1u);ea+=1u; }
    if ((sig.x|sig.y)==0u) { return vec2(0u,a.y&b.y&0x80000000u); }
    if (ea==1u && sig.y<0x100000u) { ea=0u; }
    if (ea>=2047u) { return vec2(0u,0xfff00000u); }
    return vec2(sig.x,0x80000000u|(ea<<20u)|(sig.y&0xfffffu));
}
@compute @workgroup_size(64)
fn accumulate_words(@builtin(workgroup_id) wid:vec3<u32>,
    @builtin(local_invocation_index) lid:u32,@builtin(num_workgroups) nwg:vec3<u32>) {
    let idx=(wid.x+wid.y*nwg.x+wid.z*nwg.x*nwg.y)*64u+lid;
    if (idx>=dims.np*dims.qg) { return; }
    let p=idx/dims.qg;let g=idx%dims.qg;
    var gen=prior_g[g];
    for (var i=0u;i<dims.ni;i+=1u) {
        let cat=yobs[p*dims.ni+i];
        if (block_of[i]<0 && cat>=0) {
            gen=add_negative(gen,tables[offsets[i]+g*dims.nc+u32(cat)]);
        }
    }
    general[idx]=gen;
    for (var s=0u;s<dims.ns;s+=1u) {
        for (var h=0u;h<dims.qs;h+=1u) {
            var acc=prior_s[h];
            for (var m=block_offsets[s];m<block_offsets[s+1u];m+=1u) {
                let i=members[m];let cat=yobs[p*dims.ni+i];
                if (cat>=0) { acc=add_negative(acc,tables[offsets[i]+(g*dims.qs+h)*dims.nc+u32(cat)]); }
            }
            block[((p*dims.ns+s)*dims.qg+g)*dims.qs+h]=acc;
        }
    }
}
"#;

// test-only host clock: 호출·수집 오버헤드가 포함되며 GPU timestamp가 아니다.
#[cfg(all(test, feature = "gpu", not(coverage)))]
fn profile_start() -> Option<std::time::Instant> {
    crate::two_tier_grm::tests::reference_gpu_timing_enabled().then(std::time::Instant::now)
}
#[cfg(all(test, feature = "gpu", not(coverage)))]
fn profile_finish(name: &'static str, started: Option<std::time::Instant>) {
    if let Some(started) = started {
        crate::two_tier_grm::tests::record_reference_gpu_timing(name, started.elapsed().as_secs_f64());
    }
}

#[cfg(all(feature = "gpu", not(coverage)))]
pub(crate) struct GpuLogProductState {
    ctx: &'static GpuContext,
    dims: wgpu::Buffer,
    responses: wgpu::Buffer,
    buffers: [wgpu::Buffer; 9],
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    ni: usize,
    ns: usize,
    nc: usize,
    qg: usize,
    qs: usize,
    max_persons: usize,
}

#[cfg(all(feature = "gpu", not(coverage)))]
impl GpuLogProductState {
    pub(crate) fn new(inputs: &ReducedEstepInputs) -> Option<Self> {
        use crate::gpu::{output_buffer, storage_buffer_fits, storage_entry};
        #[cfg(test)]
        let validation_clock = profile_start();
        let ctx = GpuContext::get()?;
        if !matches!(ctx.adapter_info.device_type,
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::VirtualGpu) {
            return None;
        }
        let (np, ni, ns, nc, qg, qs) = (inputs.n_persons, inputs.n_items, inputs.n_specific,
            inputs.n_cat, inputs.qg, inputs.qs);
        if inputs.n_groups != 1 || np == 0 || ni == 0 || nc == 0 || qg == 0 || qs == 0 { return None; }
        let response_len = np.checked_mul(ni)?;
        let general_len = np.checked_mul(qg)?;
        let block_len = general_len.checked_mul(ns)?.checked_mul(qs)?;
        if general_len > u32::MAX as usize / 2 || block_len > u32::MAX as usize / 2 { return None; }
        if inputs.item_block.len() != ni || inputs.blocks.len() != ns
            || inputs.log_wg.len() != qg || inputs.log_ws.len() != qs { return None; }
        let group = inputs.tables_groups.first()?;
        if group.len() != ni { return None; }
        let words = |values: &[f64]| -> Option<Vec<[u32; 2]>> {
            values.iter().map(|&v| {
                if v.is_nan() || v == f64::INFINITY || v > 0.0 { return None; }
                let bits = v.to_bits(); Some([bits as u32, (bits >> 32) as u32])
            }).collect()
        };
        if !cfg!(target_endian = "little") { return None; }
        let mut offsets = Vec::with_capacity(ni);
        let mut table_len = 0usize;
        for (i, row) in group.iter().enumerate() {
            let expected = qg.checked_mul(if inputs.item_block[i].is_some() { qs } else { 1 })?.checked_mul(nc)?;
            if row.len() != expected || row.iter().any(|v| v.is_nan() || *v == f64::INFINITY || *v > 0.0) { return None; }
            offsets.push(u32::try_from(table_len).ok()?);
            table_len = table_len.checked_add(row.len())?;
        }
        let prior_g = words(inputs.log_wg)?;
        let prior_s = words(inputs.log_ws)?;
        let block_of = inputs.item_block.iter().map(|b| match b {
            None => Some(-1), Some(s) if *s < ns => i32::try_from(*s).ok(), _ => None,
        }).collect::<Option<Vec<i32>>>()?;
        let mut block_offsets = Vec::with_capacity(ns + 1);
        let mut members = Vec::new();
        for block in inputs.blocks {
            block_offsets.push(u32::try_from(members.len()).ok()?);
            for &i in block { if i >= ni { return None; } members.push(u32::try_from(i).ok()?); }
        }
        block_offsets.push(u32::try_from(members.len()).ok()?);
        let limits = ctx.device.limits();
        let response_buffer_bytes = response_len.checked_mul(4)?.max(4);
        let lens = [response_len, table_len.checked_mul(2)?, ni, ni, ns.checked_add(1)?,
            members.len().max(1), qg.checked_mul(2)?, qs.checked_mul(2)?,
            general_len.checked_mul(2)?, block_len.max(1).checked_mul(2)?];
        if lens.iter().any(|&len| !storage_buffer_fits(&limits, len)) { return None; }
        #[cfg(test)]
        profile_finish("prepare_validation_metadata_seconds", validation_clock);
        let device = &ctx.device;
        #[cfg(test)]
        let scope_clock = profile_start();
        let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        #[cfg(test)]
        profile_finish("prepare_error_scope_setup_seconds", scope_clock);
        let result = (|| {
        let init = |label: &str, bytes: &[u8]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label), contents: if bytes.is_empty() { &[0u8; 8] } else { bytes }, usage: wgpu::BufferUsages::STORAGE,
        });
        #[cfg(test)]
        let uniform_clock = profile_start();
        let dims_values = [np, ni, ns, nc, qg, qs, 0, 0].map(|n| u32::try_from(n).ok()).into_iter().collect::<Option<Vec<_>>>()?;
        let dims = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("reference_word_dims"), contents: bytemuck::cast_slice(&dims_values),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        #[cfg(test)]
        profile_finish("prepare_uniform_buffer_seconds", uniform_clock);
        #[cfg(test)]
        let map_clock = profile_start();
        let table_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("f64_table_words"), size: table_len.checked_mul(8)? as u64,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: true,
        });
        {
            let mut view = table_buffer.slice(..).get_mapped_range_mut().ok()?;
            for (offset, row) in offsets.iter().zip(group) {
                let start = *offset as usize * 8;
                view.slice(start..start + row.len() * 8).copy_from_slice(bytemuck::cast_slice(row));
            }
        }
        table_buffer.unmap();
        #[cfg(test)]
        profile_finish("prepare_table_map_copy_seconds", map_clock);
        #[cfg(test)]
        let buffers_clock = profile_start();
        let responses = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("responses"), size: response_buffer_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let buffers = [table_buffer, init("table_offsets", bytemuck::cast_slice(&offsets)),
            init("item_block", bytemuck::cast_slice(&block_of)), init("block_offsets", bytemuck::cast_slice(&block_offsets)),
            init("members", bytemuck::cast_slice(&members)), init("prior_g_words", bytemuck::cast_slice(&prior_g)),
            init("prior_s_words", bytemuck::cast_slice(&prior_s)), output_buffer(device, "general_words", general_len * 2),
            output_buffer(device, "block_words", block_len.max(1) * 2)];
        #[cfg(test)]
        profile_finish("prepare_input_output_buffers_seconds", buffers_clock);
        #[cfg(test)]
        let layout_clock = profile_start();
        let mut layout_entries = vec![wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }];
        layout_entries.extend((1..=10).map(|i| storage_entry(i, i <= 8)));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &layout_entries });
        let bindings = [&responses, &buffers[0], &buffers[1], &buffers[2], &buffers[3], &buffers[4], &buffers[5], &buffers[6], &buffers[7], &buffers[8]];
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: dims.as_entire_binding() }];
        entries.extend(bindings.iter().enumerate().map(|(i,b)| wgpu::BindGroupEntry { binding: i as u32 + 1, resource: b.as_entire_binding() }));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &entries });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        #[cfg(test)]
        profile_finish("prepare_layout_bindgroup_seconds", layout_clock);
        #[cfg(test)]
        let module_clock = profile_start();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("reference_f64_log_products"), source: wgpu::ShaderSource::Wgsl(LOG_PRODUCT_SHADER.into()) });
        #[cfg(test)]
        profile_finish("prepare_shader_module_seconds", module_clock);
        #[cfg(test)]
        let pipeline_clock = profile_start();
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pipeline_layout), module: &shader,
            entry_point: Some("accumulate_words"), compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None });
        #[cfg(test)]
        profile_finish("prepare_compute_pipeline_seconds", pipeline_clock);
        Some(Self { ctx, dims, responses, buffers, bind_group, pipeline, ni, ns, nc, qg, qs, max_persons: np })
        })();
        #[cfg(test)]
        let receipt_clock = profile_start();
        let internal_error = pollster::block_on(internal.pop()).is_some();
        let validation_error = pollster::block_on(validation.pop()).is_some();
        let oom_error = pollster::block_on(oom.pop()).is_some();
        #[cfg(test)]
        profile_finish("prepare_error_receipt_seconds", receipt_clock);
        if internal_error || validation_error || oom_error { None } else { result }
    }

    pub(crate) fn sweep(&self, y: &[usize], observed: Option<&[bool]>, np: usize) -> Option<(Vec<f64>, Vec<f64>)> {
        use crate::gpu::{dispatch_count, dispatch_workgroups_nd, staging_buffer, submit_and_readback};
        let response_len = np.checked_mul(self.ni)?;
        let general_len = np.checked_mul(self.qg)?;
        let block_len = general_len.checked_mul(self.ns)?.checked_mul(self.qs)?;
        if np == 0 || np > self.max_persons || y.len() != response_len || observed.is_some_and(|m| m.len() != response_len) { return None; }
        let yobs: Vec<i32> = y.iter().enumerate().map(|(i,&v)| {
            if observed.is_some_and(|m| !m[i]) { Some(-1) } else if v >= self.nc { None } else { i32::try_from(v).ok() }
        }).collect::<Option<_>>()?;
        let dims = [np,self.ni,self.ns,self.nc,self.qg,self.qs,0,0].map(|n| u32::try_from(n).ok()).into_iter().collect::<Option<Vec<_>>>()?;
        let device = &self.ctx.device;
        let (x, y_workgroups, z) = dispatch_workgroups_nd(
            dispatch_count(general_len),
            device.limits().max_compute_workgroups_per_dimension,
        )?;
        let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        #[cfg(test)]
        let encode_clock = profile_start();
        self.ctx.queue.write_buffer(&self.dims, 0, bytemuck::cast_slice(&dims));
        self.ctx.queue.write_buffer(&self.responses, 0, bytemuck::cast_slice(&yobs));
        let result = (|| {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(x, y_workgroups, z);
            }
        #[cfg(test)]
        profile_finish("sweep_queue_write_encode_seconds", encode_clock);
        #[cfg(test)]
        let readback_clock = profile_start();
            let general_read = staging_buffer(device, "general_word_read", general_len * 2);
            let block_read = staging_buffer(device, "block_word_read", block_len.max(1) * 2);
            let mut copies = vec![(&self.buffers[7], &general_read, general_len * 2)];
            if self.ns > 0 { copies.push((&self.buffers[8], &block_read, block_len * 2)); }
        #[cfg(test)]
        profile_finish("sweep_readback_buffer_preparation_seconds", readback_clock);
        #[cfg(test)]
        let submit_clock = profile_start();
            let read = submit_and_readback(self.ctx, encoder, &copies)?;
            let decode = |row: &[f32]| -> Option<Vec<f64>> {
                row.chunks_exact(2).map(|v| {
                    let value = f64::from_bits(u64::from(v[0].to_bits()) | (u64::from(v[1].to_bits()) << 32));
                    (value.is_finite() || value == f64::NEG_INFINITY).then_some(value)
                }).collect()
            };
            let general = decode(&read[0])?;
            let block = if self.ns == 0 { Vec::new() } else { decode(&read[1])? };
        #[cfg(test)]
        profile_finish("sweep_submit_map_decode_seconds", submit_clock);
            Some((general, block))
        })();
        #[cfg(test)]
        let receipt_clock = profile_start();
        let internal_error = pollster::block_on(internal.pop()).is_some();
        let validation_error = pollster::block_on(validation.pop()).is_some();
        let oom_error = pollster::block_on(oom.pop()).is_some();
        #[cfg(test)]
        profile_finish("sweep_error_receipt_seconds", receipt_clock);
        if internal_error || validation_error || oom_error { None } else { result }
    }
}

#[cfg(all(feature = "gpu", not(coverage)))]
pub(crate) fn e_step_reduced_gpu_log_products(inputs: &ReducedEstepInputs) -> Option<(Vec<f64>, Vec<f64>)> {
    GpuLogProductState::new(inputs)?.sweep(inputs.y, inputs.observed, inputs.n_persons)
}

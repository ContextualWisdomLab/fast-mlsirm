// Private whole-core GPU execution characterization, no fitting or adoption.
// Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information
// item bifactor analysis. Psychological Methods, 16(3), 221–248.
// Selected local pp.231–232/Eqs.15–17 and p.248/Eq.C6 read.
#[path = "ac_gpu_fixture.rs"]
mod fixture;
/// A usable adapter for the reduced bifactor E-step layout. Mirrors the
/// fail-closed checks of `e_step_reduced_gpu`; CPU-only CI skips, while an
/// explicit `WGPU_BACKEND=metal` request must not silently skip.
fn reduced_estep_adapter_available() -> bool {
    let usable = crate::gpu::GpuContext::get().is_some_and(|ctx| {
        ctx.adapter_storage_buffers() >= crate::gpu_bifactor::MIN_STORAGE_BUFFERS
    });
    if std::env::var("WGPU_BACKEND").is_ok_and(|b| b.eq_ignore_ascii_case("metal")) {
        assert!(
            usable,
            "WGPU_BACKEND=metal was explicit, but no usable Metal adapter was selected"
        );
    }
    usable
}
fn run(q: usize) {
    assert!(cfg!(feature = "gpu"));
    if !reduced_estep_adapter_available() {
        eprintln!("no usable GPU adapter present; skipping fixed GPU count-owner check Q{q}");
        return;
    }
    let (cpu, _, _) = fixture::fixture(q, false, 0.0, false, crate::Device::Cpu);
    let (gpu, cpu_calls, _) = fixture::fixture(q, false, 0.0, false, crate::Device::Gpu);
    let cpu_buffers = fixture::count_buffers();
    let (after, _, _) = fixture::fixture(q, false, 0.0, false, crate::Device::Cpu);
    assert_eq!(cpu, after, "GPU call contaminated CPU invocation");
    assert_eq!(cpu.len(), gpu.len());
    let cpu_values: Vec<f64> = cpu.iter().copied().map(f64::from_bits).collect();
    let gpu_values: Vec<f64> = gpu.iter().copied().map(f64::from_bits).collect();
    let loglik_error = (cpu_values[0] - gpu_values[0]).abs();
    // Fixture layout: ll + flattened group/item/node/category counts,
    // followed by 2-group masses/general moments and two specific moments.
    let count_len = 2 * (4 * q * q + q) * 4;
    let max_count_error = cpu_values[1..1 + count_len]
        .iter()
        .zip(&gpu_values[1..1 + count_len])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let moment_errors: Vec<f64> = cpu_values[1 + count_len..]
        .iter()
        .zip(&gpu_values[1 + count_len..])
        .map(|(a, b)| (a - b).abs())
        .collect();
    // Optional evidence export for hardware receipts; assertions below run regardless.
    if let Ok(out) = std::env::var("AC_GPU_FIXED_OUTPUT") {
        let payload = serde_json::json!({"q":q,"n_persons":12,"n_groups":2,"n_items":5,
        "cpu_loglik":cpu_values[0],"gpu_loglik":gpu_values[0],"loglik_error":loglik_error,
        "max_count_error":max_count_error,"moment_abs_errors":moment_errors,
        "cpu_fallback_lse_calls":cpu_calls,"cpu_fallback_count_buffers":cpu_buffers,
        "returned_f64_values":cpu.len(),"count_f64_values":count_len,
        "cpu_before_after_bit_equal":true,"fits":0,
        "cpu_values":cpu_values,"gpu_values":gpu_values,
        "scope":"Full-core actual Device::Gpu entry at fixed tiny model; fallback denied by existing CPU branch counters. Not fitting/MAP/refinement/coverage."});
        let path = std::path::Path::new(&out).join(format!("q{q}-raw.json"));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        use std::io::Write;
        file.write_all(&serde_json::to_vec(&payload).unwrap())
            .unwrap();
    }
    assert_eq!(cpu_calls, 0, "GPU requested but CPU fallback executed");
    assert_eq!(cpu_buffers, 0, "GPU returned through CPU counts allocation");
    assert!(
        loglik_error <= 1e-3,
        "Existing tiny loglik envelope rejected Q{q}: {loglik_error}"
    );
    assert!(
        max_count_error <= 1e-4,
        "Existing tiny per-count envelope rejected Q{q}: {max_count_error}"
    );
}
#[test]
fn fixed_gpu_no_fallback_q121() {
    run(121);
}
#[test]
fn fixed_gpu_no_fallback_q241() {
    run(241);
}

//! Regression contracts for the precomputed marginal distance hot paths.
//!
//! The marginal estimator's latent interaction term depends only on the
//! `(item, latent-node)` pair, so the table build and M-step hot paths reuse a
//! precomputed table instead of recomputing `sqrt(eps + ||x - zeta_i||^2)` per
//! `(context, item, trait-node, latent-node)` cell. These tests pin the
//! optimization to the scalar reference: distance-kind results must be
//! bit-identical, inner-product results agree to floating-point round-off,
//! and the coarse context-sharded fill must be bit-identical to the serial
//! fill. The ignored benchmark emits the same-head runtime/allocation
//! artifact required by issue #403.

use crate::marginal::{
    build_tables_offset_with_workers, distance_value_table, eta_at_kind,
    latent_interaction_table, log_sigmoid, Contexts, Grids, Tables,
};
use crate::{interaction_kind, model_exec_flags, InteractionKind, ModelConfig, ModelType};

/// Deterministic linear congruential generator for reproducible fixtures.
struct FixtureRng(u64);

impl FixtureRng {
    /// Return the next uniform draw in `(-1, 1)`.
    fn next_signed_unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (((self.0 >> 11) as f64) / ((1_u64 << 53) as f64)) * 2.0 - 1.0
    }
}

/// Build a deterministic synthetic problem for the table-fill contracts.
fn table_fixture(
    model_type: ModelType,
    n_ctx: usize,
    n_items: usize,
    q_t: usize,
    side: usize,
) -> (ModelConfig, Contexts, Grids, Vec<usize>, Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let latent_dim = 2_usize;
    let n_dims = 2_usize;
    let n_x = side * side;
    let mut rng = FixtureRng(0x9E37_79B9_7F4A_7C15);

    let config = ModelConfig {
        n_persons: 8,
        n_items,
        n_dims,
        latent_dim,
        model_type,
        eps_distance: 1e-3,
    };
    let contexts = Contexts {
        n_ctx,
        shift: (0..n_ctx * n_dims).map(|_| rng.next_signed_unit()).collect(),
        scale: (0..n_ctx * n_dims)
            .map(|_| 0.5 + 0.5 * rng.next_signed_unit().abs())
            .collect(),
        u_nodes: Vec::new(),
        u_logw: Vec::new(),
    };
    let mut x_grid = Vec::with_capacity(n_x * latent_dim);
    for row in 0..side {
        for col in 0..side {
            x_grid.push(-2.0 + 4.0 * (row as f64) / ((side - 1) as f64));
            x_grid.push(-2.0 + 4.0 * (col as f64) / ((side - 1) as f64));
        }
    }
    let grids = Grids {
        t_nodes: (0..q_t)
            .map(|t| -3.0 + 6.0 * (t as f64) / ((q_t - 1) as f64))
            .collect(),
        t_logw: vec![0.0; q_t],
        x_grid,
        x_logw: vec![0.0; n_x],
        q_t,
        n_x,
    };
    let factor_id: Vec<usize> = (0..n_items).map(|i| i % n_dims).collect();
    let alpha: Vec<f64> = (0..n_items).map(|_| 0.3 * rng.next_signed_unit()).collect();
    let b: Vec<f64> = (0..n_items).map(|_| rng.next_signed_unit()).collect();
    let zeta: Vec<f64> = (0..n_items * latent_dim)
        .map(|_| 1.5 * rng.next_signed_unit())
        .collect();
    let tau = 0.25;
    (config, contexts, grids, factor_id, alpha, b, zeta, tau)
}

/// Scalar-reference table fill: the pre-optimization per-cell algorithm.
#[allow(clippy::too_many_arguments)]
fn reference_tables(
    alpha: &[f64],
    b: &[f64],
    zeta: &[f64],
    tau: f64,
    config: &ModelConfig,
    factor_id: &[usize],
    ctx: &Contexts,
    grids: &Grids,
    offset: Option<&[f64]>,
) -> Tables {
    let (free_alpha, _uses_space) = model_exec_flags(config.model_type);
    let kind = interaction_kind(config.model_type);
    let (n_items, n_dims, latent_dim) = (config.n_items, config.n_dims, config.latent_dim);
    let (q_t, n_x) = (grids.q_t, grids.n_x);
    let cell = q_t * n_x;
    let mut logp1 = vec![0.0_f64; ctx.n_ctx * n_items * cell];
    let mut logp0 = vec![0.0_f64; ctx.n_ctx * n_items * cell];
    let mut c0 = vec![0.0_f64; ctx.n_ctx * n_dims * cell];
    for s in 0..ctx.n_ctx {
        for i in 0..n_items {
            let d = factor_id[i];
            let off = offset.map(|o| o[s * n_items + i]).unwrap_or(0.0);
            let (shift, scale) = (ctx.shift[s * n_dims + d], ctx.scale[s * n_dims + d]);
            for (t, &node_t) in grids.t_nodes.iter().enumerate() {
                let theta = shift + scale * node_t;
                for x in 0..n_x {
                    let eta = off
                        + eta_at_kind(
                            alpha,
                            b,
                            zeta,
                            tau,
                            free_alpha,
                            kind,
                            latent_dim,
                            config.eps_distance,
                            i,
                            theta,
                            &grids.x_grid[x * latent_dim..(x + 1) * latent_dim],
                        );
                    let idx = (s * n_items + i) * cell + t * n_x + x;
                    logp1[idx] = log_sigmoid(eta);
                    logp0[idx] = log_sigmoid(-eta);
                    c0[(s * n_dims + d) * cell + t * n_x + x] += logp0[idx];
                }
            }
        }
    }
    Tables { logp1, logp0, c0 }
}

/// Assert two tables are bit-identical entry by entry.
fn assert_tables_bit_identical(left: &Tables, right: &Tables) {
    assert_eq!(left.logp1.len(), right.logp1.len());
    for (index, (l, r)) in left.logp1.iter().zip(&right.logp1).enumerate() {
        assert_eq!(l.to_bits(), r.to_bits(), "logp1 differs at {index}");
    }
    for (index, (l, r)) in left.logp0.iter().zip(&right.logp0).enumerate() {
        assert_eq!(l.to_bits(), r.to_bits(), "logp0 differs at {index}");
    }
    for (index, (l, r)) in left.c0.iter().zip(&right.c0).enumerate() {
        assert_eq!(l.to_bits(), r.to_bits(), "c0 differs at {index}");
    }
}

#[test]
fn distance_table_matches_the_scalar_accumulation_bitwise() {
    let (config, _ctx, grids, _factor_id, _alpha, _b, zeta, _tau) =
        table_fixture(ModelType::Mls2plm, 2, 7, 3, 3);
    let table = distance_value_table(
        &zeta,
        config.latent_dim,
        config.eps_distance,
        config.n_items,
        &grids.x_grid,
        grids.n_x,
    );
    for i in 0..config.n_items {
        for x in 0..grids.n_x {
            let x_node = &grids.x_grid[x * config.latent_dim..(x + 1) * config.latent_dim];
            let mut dist2 = config.eps_distance;
            for k in 0..config.latent_dim {
                let diff = x_node[k] - zeta[i * config.latent_dim + k];
                dist2 += diff * diff;
            }
            assert_eq!(table[i * grids.n_x + x].to_bits(), dist2.sqrt().to_bits());
        }
    }
}

#[test]
fn interaction_table_reproduces_the_scalar_eta_by_kind() {
    let (config, _ctx, grids, _factor_id, alpha, b, zeta, tau) =
        table_fixture(ModelType::Mls2plm, 2, 7, 3, 3);
    for (model_type, tolerance) in [
        (ModelType::Mls2plm, 0.0_f64),
        (ModelType::Mirt, 0.0_f64),
        (ModelType::Bifac2plm, 1e-12_f64),
    ] {
        let kind = interaction_kind(model_type);
        let (free_alpha, _) = model_exec_flags(model_type);
        let table = latent_interaction_table(
            &zeta,
            tau,
            kind,
            config.latent_dim,
            config.eps_distance,
            config.n_items,
            &grids.x_grid,
            grids.n_x,
        );
        for i in 0..config.n_items {
            let a = if free_alpha { alpha[i].exp() } else { 1.0 };
            for x in 0..grids.n_x {
                let theta = 0.75;
                let reference_eta = eta_at_kind(
                    &alpha,
                    &b,
                    &zeta,
                    tau,
                    free_alpha,
                    kind,
                    config.latent_dim,
                    config.eps_distance,
                    i,
                    theta,
                    &grids.x_grid[x * config.latent_dim..(x + 1) * config.latent_dim],
                );
                let table_eta = a * theta + b[i] + table[i * grids.n_x + x];
                if tolerance == 0.0 {
                    assert_eq!(
                        table_eta.to_bits(),
                        reference_eta.to_bits(),
                        "eta differs for kind {kind:?} at item {i}, node {x}"
                    );
                } else {
                    assert!(
                        (table_eta - reference_eta).abs()
                            <= tolerance * reference_eta.abs().max(1.0),
                        "eta beyond tolerance for kind {kind:?} at item {i}, node {x}"
                    );
                }
            }
        }
        if kind == InteractionKind::None {
            assert!(table.iter().all(|&term| term == 0.0));
        }
    }
}

#[test]
fn serial_table_build_is_bit_identical_to_the_scalar_reference() {
    let (config, ctx, grids, factor_id, alpha, b, zeta, tau) =
        table_fixture(ModelType::Mls2plm, 3, 11, 5, 3);
    let offsets: Vec<f64> = (0..ctx.n_ctx * config.n_items)
        .map(|index| 0.01 * (index as f64))
        .collect();
    for offset in [None, Some(offsets.as_slice())] {
        let reference =
            reference_tables(&alpha, &b, &zeta, tau, &config, &factor_id, &ctx, &grids, offset);
        let optimized = build_tables_offset_with_workers(
            &alpha, &b, &zeta, tau, &config, &factor_id, &ctx, &grids, offset, 1,
        );
        assert_tables_bit_identical(&reference, &optimized);
    }
}

#[test]
fn sharded_table_build_is_bit_identical_to_the_serial_fill() {
    // 210 items * (7 * 9) cells * 5 contexts exceeds the multithreading floor,
    // so worker_count > 1 genuinely exercises the sharded path.
    let (config, ctx, grids, factor_id, alpha, b, zeta, tau) =
        table_fixture(ModelType::Mls2plm, 5, 210, 7, 3);
    let serial = build_tables_offset_with_workers(
        &alpha, &b, &zeta, tau, &config, &factor_id, &ctx, &grids, None, 1,
    );
    for worker_count in [2, 4, 7] {
        let sharded = build_tables_offset_with_workers(
            &alpha, &b, &zeta, tau, &config, &factor_id, &ctx, &grids, None, worker_count,
        );
        assert_tables_bit_identical(&serial, &sharded);
    }
}

/// Same-head benchmark artifact for issue #403 (runtime, table allocation,
/// dimensions, backend, and the regression threshold), emitted as JSON.
///
/// Run with:
/// `cargo test --release -p mlsirm-core marginal_distance_benchmark -- --ignored --nocapture`
#[test]
#[ignore = "same-head benchmark evidence; run explicitly with --ignored --nocapture"]
fn marginal_distance_benchmark_reports_runtime_and_allocation() {
    let (config, ctx, grids, factor_id, alpha, b, zeta, tau) =
        table_fixture(ModelType::Mls2plm, 6, 60, 21, 11);
    let cell = grids.q_t * grids.n_x;
    let table_bytes = (2 * ctx.n_ctx * config.n_items * cell
        + ctx.n_ctx * config.n_dims * cell)
        * std::mem::size_of::<f64>();
    let interaction_bytes = config.n_items * grids.n_x * std::mem::size_of::<f64>();

    let time_fill = |fill: &dyn Fn() -> Tables| -> f64 {
        let mut best_seconds = f64::INFINITY;
        for _ in 0..3 {
            let started = std::time::Instant::now();
            let tables = fill();
            let elapsed = started.elapsed().as_secs_f64();
            assert!(tables.logp1.iter().all(|value| value.is_finite()));
            best_seconds = best_seconds.min(elapsed);
        }
        best_seconds
    };

    let reference_seconds = time_fill(&|| {
        reference_tables(&alpha, &b, &zeta, tau, &config, &factor_id, &ctx, &grids, None)
    });
    let serial_seconds = time_fill(&|| {
        build_tables_offset_with_workers(
            &alpha, &b, &zeta, tau, &config, &factor_id, &ctx, &grids, None, 1,
        )
    });
    let sharded_seconds = time_fill(&|| {
        build_tables_offset_with_workers(
            &alpha, &b, &zeta, tau, &config, &factor_id, &ctx, &grids, None, 4,
        )
    });

    // Regression threshold: the precomputed serial fill must never be more
    // than 5% slower than the scalar reference on the representative workload.
    let regression_threshold = 1.05;
    assert!(
        serial_seconds <= reference_seconds * regression_threshold,
        "precomputed fill regressed: serial {serial_seconds:.6}s vs reference {reference_seconds:.6}s"
    );

    let artifact = format!(
        concat!(
            "{{\"benchmark_name\": \"marginal_distance_table_build\", ",
            "\"backend_label\": \"rust-cpu\", ",
            "\"workload_dimensions\": {{\"context_count\": {}, \"item_count\": {}, ",
            "\"trait_nodes\": {}, \"latent_nodes\": {}, \"latent_dim\": {}}}, ",
            "\"reference_seconds\": {:.6}, \"serial_seconds\": {:.6}, ",
            "\"sharded_seconds\": {:.6}, \"sharded_workers\": 4, ",
            "\"output_table_bytes\": {}, \"interaction_table_bytes\": {}, ",
            "\"regression_threshold\": {:.2}}}"
        ),
        ctx.n_ctx,
        config.n_items,
        grids.q_t,
        grids.n_x,
        config.latent_dim,
        reference_seconds,
        serial_seconds,
        sharded_seconds,
        table_bytes,
        interaction_bytes,
        regression_threshold,
    );
    println!("{artifact}");
    let artifact_path = std::path::Path::new("target").join("marginal_distance_benchmark.json");
    if std::fs::write(&artifact_path, &artifact).is_err() {
        // The artifact is also printed above, so a read-only target directory
        // must not fail the benchmark evidence itself.
        eprintln!("could not persist {artifact_path:?}; JSON emitted on stdout");
    }
}

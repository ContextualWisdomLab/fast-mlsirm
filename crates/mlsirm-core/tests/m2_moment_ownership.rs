use mlsirm_core::fitstats::{
    cluster_moment_covariance, factorized_multilevel_moments, factorized_trait_moments,
};

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "actual={actual} expected={expected}"
    );
}

#[test]
fn factorized_trait_moments_preserve_shared_factor_products() {
    let moments = factorized_trait_moments(
        &[0.2, 0.4, 0.6, 0.8],
        &[0.5, 0.5],
        &[1.0],
        2,
        &[0, 0],
        1,
        &[vec![0], vec![1], vec![0, 1]],
    )
    .expect("valid simple-structure grid must integrate");

    assert_close(moments[0], 0.3);
    assert_close(moments[1], 0.7);
    assert_close(moments[2], 0.22);
}

#[test]
fn factorized_multilevel_moments_integrate_shared_cluster_nodes() {
    let moments = factorized_multilevel_moments(
        &[
            0.2, 0.4, 0.6, 0.8, // cluster node 0
            0.4, 0.6, 0.8, 1.0, // cluster node 1
        ],
        &[0.5, 0.5],
        &[0.5, 0.5],
        &[1.0],
        2,
        2,
        &[0, 0],
        1,
        &[vec![0], vec![1], vec![0, 1]],
    )
    .expect("valid multilevel grid must integrate");

    assert_close(moments[0], 0.4);
    assert_close(moments[1], 0.8);
    assert_close(moments[2], 0.34);
}

#[test]
fn cluster_moment_covariance_matches_finite_cluster_correction() {
    let covariance = cluster_moment_covariance(
        &[
            1.0, 0.0, 1.0, 0.0, // cluster 0
            0.0, 1.0, 0.0, 1.0, // cluster 1
            1.0, 1.0, 1.0, 1.0, // cluster 2
        ],
        &[0.5, 0.5],
        &[0, 0, 1, 1, 2, 2],
        6,
        2,
        3,
    )
    .expect("valid cluster totals must produce covariance");

    assert_close(covariance[0], 2.0 / 3.0);
    assert_close(covariance[1], -1.0 / 3.0);
    assert_close(covariance[2], -1.0 / 3.0);
    assert_close(covariance[3], 2.0 / 3.0);
}

#[test]
fn moment_and_cluster_kernels_fail_closed_on_shape_and_label_errors() {
    let moment_error = factorized_multilevel_moments(
        &[0.5],
        &[1.0],
        &[1.0],
        &[1.0],
        1,
        1,
        &[0, 0],
        1,
        &[vec![0]],
    )
    .expect_err("short probability grids must fail closed");
    assert!(moment_error.contains("probability grid"));

    let covariance_error = cluster_moment_covariance(
        &[1.0, 0.0, 0.0, 1.0],
        &[0.5, 0.5],
        &[0, 3],
        2,
        2,
        3,
    )
    .expect_err("non-compact cluster labels must fail closed");
    assert!(covariance_error.contains("cluster ids"));

    let overflow_error = cluster_moment_covariance(
        &[1.0, 0.0],
        &[0.5, 0.5],
        &[0],
        1,
        2,
        usize::MAX,
    )
    .expect_err("cluster covariance allocations must reject overflow");
    assert!(overflow_error.contains("overflow"));
}

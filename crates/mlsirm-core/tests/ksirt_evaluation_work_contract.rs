use mlsirm_core::ksirt::{ksirt, KsirtKernel};

#[test]
fn evaluation_work_budget_precedes_response_value_traversal() {
    let mut responses = vec![vec![0.0; 101]; 2];
    responses[0][0] = f64::NAN;

    let error = ksirt(&responses, KsirtKernel::Gaussian, 100_000, None)
        .expect_err("oversized person-item-grid work must fail before value traversal");

    assert_eq!(
        error,
        "ksirt evaluation work exceeds 20000000 person-item-grid terms"
    );
}

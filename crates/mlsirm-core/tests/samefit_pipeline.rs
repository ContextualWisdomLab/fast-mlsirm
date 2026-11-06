#![cfg(feature = "gpu")]
use mlsirm_core::samefit_grm_joint::joint_diagnostics;
#[test]
fn rejects_incomplete_joint_input_before_gpu() {
    assert!(joint_diagnostics(
        &[],
        &[],
        &mlsirm_core::samefit_grm::PackedArchive::new(vec![true], 2, 2, vec![1.0, 0.0, 0.0, 0.0])
            .unwrap(),
        121
    )
    .is_err());
}

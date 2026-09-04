use mlsirm_core::interaction_map_envelope::{
    residual_interaction_map_input_digest, RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
};

#[test]
fn direct_rust_digest_rejects_oversized_identifier_bytes() {
    let oversized = "x".repeat(16 * 1024 * 1024 + 1);
    let persons = vec![oversized];
    let items = vec!["item-a".to_string()];

    let error = residual_interaction_map_input_digest(
        RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
        &persons,
        &items,
        &[1.0],
        &[1.0],
        1,
        1,
        1,
    )
    .expect_err("direct Rust callers must not hash unbounded identifier text");

    assert!(error.contains("person identifier bytes"));
}

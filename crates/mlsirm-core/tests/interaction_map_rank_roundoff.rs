use mlsirm_core::interaction_map_envelope::{
    residual_interaction_map_envelope, residual_interaction_map_input_digest,
    RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
};

#[test]
fn one_row_scaled_residuals_do_not_publish_roundoff_rank() {
    let persons = vec!["person-a".to_string()];
    let items = vec![
        "item-a".to_string(),
        "item-b".to_string(),
        "item-c".to_string(),
    ];
    let observed = [1.0e12, -1.0e12, 1.0];
    let expected = [0.0; 3];
    let digest = residual_interaction_map_input_digest(
        RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
        &persons,
        &items,
        &observed,
        &expected,
        1,
        3,
        2,
    )
    .unwrap();

    let envelope = residual_interaction_map_envelope(
        RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
        &digest,
        &persons,
        &items,
        &observed,
        &expected,
        1,
        3,
        2,
    )
    .expect("roundoff modes must not inflate a one-row residual map above rank one");

    assert_eq!(envelope.map.map_person_count, 1);
    assert_eq!(envelope.map.map_item_count, 3);
    assert_eq!(envelope.map.effective_rank, 1);
    assert_eq!(envelope.map.singular_values.len(), 1);
}

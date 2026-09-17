use mlsirm_core::interaction_map_envelope::{
    residual_interaction_map_envelope, residual_interaction_map_input_digest,
    RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
};

fn ids(prefix: &str, count: usize) -> Vec<String> {
    (0..count).map(|index| format!("{prefix}-{index}")).collect()
}

#[test]
fn one_row_scaled_residuals_do_not_publish_roundoff_rank() {
    let persons = ids("person", 1);
    let items = ids("item", 3);
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

#[test]
fn gram_roundoff_does_not_invent_a_second_rank_one_component() {
    let persons = ids("person", 2);
    let items = ids("item", 3);
    let observed = [
        1.0e100, 3.0e100, -4.0e100,
        2.0e100, 6.0e100, -8.0e100,
    ];
    let expected = [0.0; 6];
    let digest = residual_interaction_map_input_digest(
        RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
        &persons,
        &items,
        &observed,
        &expected,
        2,
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
        2,
        3,
        2,
    )
    .expect("normal-equation roundoff must not create a second rank-one component");

    assert_eq!(envelope.map.effective_rank, 1);
    assert_eq!(envelope.map.singular_values.len(), 1);
}

#[test]
fn finite_inputs_that_overflow_internal_factorization_fail_closed() {
    let persons = ids("person", 3);
    let items = ids("item", 1);
    let observed = [1.0e308, 1.0e308, -1.0e308];
    let expected = [0.0; 3];
    let digest = residual_interaction_map_input_digest(
        RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
        &persons,
        &items,
        &observed,
        &expected,
        3,
        1,
        1,
    )
    .unwrap();

    let error = residual_interaction_map_envelope(
        RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
        &digest,
        &persons,
        &items,
        &observed,
        &expected,
        3,
        1,
        1,
    )
    .expect_err("non-finite internal state must fail instead of publishing zero rank");

    assert!(error.contains("non-finite"));
}

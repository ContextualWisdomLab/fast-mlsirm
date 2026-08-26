use mlsirm_core::interaction_map::residual_interaction_map;
use mlsirm_core::interaction_map_envelope::{
    residual_interaction_map_envelope, RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
    RESIDUAL_INTERACTION_MAP_TIE_POLICY,
};

fn ids(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_string()).collect()
}

#[test]
fn retained_cell_values_are_rust_owned_in_original_index_order() {
    let observed = [9.0, f64::NAN, 2.0, 3.0];
    let expected = [8.0, 7.0, 0.5, 1.5];

    let map = residual_interaction_map(&observed, &expected, 2, 2, 1).unwrap();

    assert_eq!(map.person_indices, vec![1]);
    assert_eq!(map.item_indices, vec![0, 1]);
    assert_eq!(map.observed, vec![2.0, 3.0]);
    assert_eq!(map.expected, vec![0.5, 1.5]);
    assert_eq!(map.residual, vec![1.5, 1.5]);
}

#[test]
fn envelope_persists_requested_axis_count_and_tie_contract() {
    let envelope = residual_interaction_map_envelope(
        RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
        &ids(&["person-0", "person-1"]),
        &ids(&["item-0", "item-1"]),
        &[1.0, 1.0, 1.0, 1.0],
        &[1.0, 1.0, 1.0, 1.0],
        2,
        2,
        2,
    )
    .unwrap();

    assert_eq!(envelope.requested_axis_count, 2);
    assert_eq!(
        envelope.cell_extrema_tie_policy,
        RESIDUAL_INTERACTION_MAP_TIE_POLICY
    );
    assert_eq!(envelope.closest_cell_ids, Some(("person-0".into(), "item-0".into())));
    assert_eq!(envelope.farthest_cell_ids, Some(("person-0".into(), "item-0".into())));
}

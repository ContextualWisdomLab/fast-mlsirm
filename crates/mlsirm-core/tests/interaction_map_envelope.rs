use mlsirm_core::interaction_map::residual_interaction_map;

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

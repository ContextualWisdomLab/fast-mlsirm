use mlsirm_core::interaction_map::residual_interaction_map;
use mlsirm_core::interaction_map_envelope::{
    residual_interaction_map_envelope, ResidualInteractionMapEnvelope,
    RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION, RESIDUAL_INTERACTION_MAP_TIE_POLICY,
};

fn ids(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_string()).collect()
}

fn distance_for(
    envelope: &ResidualInteractionMapEnvelope,
    person_id: &str,
    item_id: &str,
) -> f64 {
    let person = envelope
        .retained_person_ids
        .iter()
        .position(|value| value == person_id)
        .expect("person id must be retained");
    let item = envelope
        .retained_item_ids
        .iter()
        .position(|value| value == item_id)
        .expect("item id must be retained");
    envelope.map.distance[person * envelope.map.map_item_count + item]
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
    assert_eq!(
        envelope.closest_cell_ids,
        Some(("person-0".into(), "item-0".into()))
    );
    assert_eq!(
        envelope.farthest_cell_ids,
        Some(("person-0".into(), "item-0".into()))
    );
}

#[test]
fn envelope_recovers_a_known_rank_one_interaction_without_residual_error() {
    // The true interaction is the zero-mean rank-one matrix
    // [[+1, -1], [-1, +1]].  With two requested axes, the Rust-owned
    // factorization must recover rank one, allocate all inertia to that axis,
    // and reconstruct the complete interaction without unexplained residual.
    let observed = [2.0, 0.0, 0.0, 2.0];
    let expected = [1.0, 1.0, 1.0, 1.0];
    let envelope = residual_interaction_map_envelope(
        RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
        &ids(&["person-a", "person-b"]),
        &ids(&["item-a", "item-b"]),
        &observed,
        &expected,
        2,
        2,
        2,
    )
    .unwrap();

    assert_eq!(envelope.map.effective_rank, 1);
    assert_eq!(envelope.map.person_indices, vec![0, 1]);
    assert_eq!(envelope.map.item_indices, vec![0, 1]);
    assert_eq!(
        envelope.retained_person_ids,
        ids(&["person-a", "person-b"])
    );
    assert_eq!(envelope.retained_item_ids, ids(&["item-a", "item-b"]));
    assert_eq!(envelope.map.singular_values.len(), 1);
    assert!((envelope.map.axis_shares[0] - 1.0).abs() < 1e-12);
    assert_eq!(envelope.map.axis_shares[1], 0.0);

    for ((observed_value, expected_value), fitted) in observed
        .iter()
        .zip(expected.iter())
        .zip(envelope.map.reconstruction.iter())
    {
        let truth = observed_value - expected_value;
        assert!((truth - fitted).abs() < 1e-12);
    }
    assert!(envelope
        .map
        .unexplained
        .iter()
        .all(|value| value.abs() < 1e-12));
}

#[test]
fn envelope_rank_two_recovery_is_deterministic_under_axis_permutations() {
    // This zero-mean 3 x 2 interaction has rank two. Reordering both axes must
    // not change singular values or the requested-axis distance attached to an
    // opaque person/item identity, and two axes must reconstruct the true input.
    let observed = [4.0, 1.0, 2.5, 3.0, 1.5, 0.0];
    let expected = [2.0; 6];
    let original = residual_interaction_map_envelope(
        RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
        &ids(&["person-a", "person-b", "person-c"]),
        &ids(&["item-a", "item-b"]),
        &observed,
        &expected,
        3,
        2,
        2,
    )
    .unwrap();

    let permuted_observed = [0.0, 1.5, 1.0, 4.0, 3.0, 2.5];
    let permuted_expected = [2.0; 6];
    let permuted = residual_interaction_map_envelope(
        RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
        &ids(&["person-c", "person-a", "person-b"]),
        &ids(&["item-b", "item-a"]),
        &permuted_observed,
        &permuted_expected,
        3,
        2,
        2,
    )
    .unwrap();

    assert_eq!(original.map.effective_rank, 2);
    assert_eq!(permuted.map.effective_rank, 2);
    for (left, right) in original
        .map
        .singular_values
        .iter()
        .zip(permuted.map.singular_values.iter())
    {
        assert!((left - right).abs() < 1e-10);
    }
    for person_id in ["person-a", "person-b", "person-c"] {
        for item_id in ["item-a", "item-b"] {
            let left = distance_for(&original, person_id, item_id);
            let right = distance_for(&permuted, person_id, item_id);
            assert!((left - right).abs() < 1e-10);
        }
    }
    for ((observed_value, expected_value), fitted) in observed
        .iter()
        .zip(expected.iter())
        .zip(original.map.reconstruction.iter())
    {
        assert!((observed_value - expected_value - fitted).abs() < 1e-10);
    }
    for ((observed_value, expected_value), fitted) in permuted_observed
        .iter()
        .zip(permuted_expected.iter())
        .zip(permuted.map.reconstruction.iter())
    {
        assert!((observed_value - expected_value - fitted).abs() < 1e-10);
    }
}

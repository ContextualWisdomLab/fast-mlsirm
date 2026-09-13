use mlsirm_core::interaction_map_envelope::{
    residual_interaction_map_input_digest, RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
};

fn ids(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_string()).collect()
}

#[test]
fn input_digest_canonicalizes_quiet_and_signaling_missing_nan_variants() {
    let persons = ids(&["person-a", "person-b"]);
    let items = ids(&["item-a", "item-b"]);
    let expected = [1.0; 4];
    let nan_bits = [
        0x7ff8_0000_0000_0001,
        0x7ff8_0000_0000_0002,
        0xfff8_0000_0000_0001,
        0x7ff0_0000_0000_0001,
        0xfff0_0000_0000_0001,
    ];

    let digests: Vec<String> = nan_bits
        .iter()
        .map(|bits| {
            let missing = f64::from_bits(*bits);
            assert!(missing.is_nan());
            assert_eq!(missing.to_bits(), *bits);
            let observed = [2.0, missing, 0.0, 2.0];
            residual_interaction_map_input_digest(
                RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
                &persons,
                &items,
                &observed,
                &expected,
                2,
                2,
                1,
            )
            .unwrap()
        })
        .collect();

    assert!(digests.iter().all(|digest| digest == &digests[0]));
}

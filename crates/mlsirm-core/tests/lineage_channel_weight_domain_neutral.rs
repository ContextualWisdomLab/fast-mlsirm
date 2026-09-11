use mlsirm_core::lineage_channel_weight::{
    LineageChannelWeightEvidence, LineageCriterionAnchorV1,
    LINEAGE_CHANNEL_WEIGHT_EVIDENCE_SCHEMA,
};

fn payload(anchor_field: &str) -> String {
    format!(
        r#"{{"schema_version":"{LINEAGE_CHANNEL_WEIGHT_EVIDENCE_SCHEMA}","estimation_run_id":"018f47e7-7b5b-7cc0-98c6-15fdf9e3d9b1","source_snapshot_sha256":"{}","knowledge_cutoff":"2026-08-25T00:00:00Z","channel_codes":["temporal","text"],"pair_evidence":[{{"pair_id":"018f47e7-7b5b-7cc0-98c6-015fdf9e3d91","group_id":"group-a","channel_scores":[0.2,0.8]}},{{"pair_id":"018f47e7-7b5b-7cc0-98c6-015fdf9e3d92","group_id":"group-b","channel_scores":[0.7,0.3]}}],"{anchor_field}":{{"contract_version":1,"anchor_kind_code":"lineage_pair_criterion","estimation_run_id":"018f47e7-7b5b-7cc0-98c6-15fdf9e3d9b1","source_snapshot_sha256":"{}","knowledge_cutoff":"2026-08-25T00:00:00Z","criterion_validity_status":"accepted","validated_pair_count":2}}}}"#,
        "a".repeat(64),
        "a".repeat(64),
    )
}

#[test]
fn canonical_anchor_contract_is_domain_neutral() {
    let admitted = LineageChannelWeightEvidence::from_json(&payload("criterion_anchor"))
        .expect("canonical domain-neutral anchor is admitted");
    let anchor: &LineageCriterionAnchorV1 = &admitted.evidence().criterion_anchor;
    assert_eq!(anchor.anchor_kind_code, "lineage_pair_criterion");
    assert_eq!(anchor.validated_pair_count, 2);
}

#[test]
fn legacy_tepp_anchor_wire_name_is_only_a_compatibility_alias() {
    let canonical = LineageChannelWeightEvidence::from_json(&payload("criterion_anchor"))
        .expect("canonical domain-neutral anchor is admitted");
    let legacy = LineageChannelWeightEvidence::from_json(&payload("tepp_anchor"))
        .expect("legacy serialized field remains readable");
    assert_eq!(canonical.evidence(), legacy.evidence());
}

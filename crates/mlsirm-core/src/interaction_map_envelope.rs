//! Versioned, domain-neutral envelope around the Rust residual interaction map.
//!
//! This module adds provenance and opaque caller identifiers without moving any
//! interaction-map arithmetic out of `interaction_map`.

use std::collections::BTreeSet;

use crate::interaction_map::{residual_interaction_map, ResidualInteractionMap};

/// Current public schema accepted by the Rust envelope constructor.
pub const RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION: &str =
    "fast-mlsirm.residual-interaction-map.v1";
/// Stable algorithm identity for the existing complete-case Gabriel map.
pub const RESIDUAL_INTERACTION_MAP_ALGORITHM_ID: &str =
    "gabriel-complete-case-symmetric-residual-map.v1";
/// Package version that produced the envelope.
pub const RESIDUAL_INTERACTION_MAP_IMPLEMENTATION_VERSION: &str = env!("CARGO_PKG_VERSION");
/// Stable calculation-provenance identity for the numerical owner.
pub const RESIDUAL_INTERACTION_MAP_CALCULATION_PROVENANCE: &str =
    "mlsirm-core::interaction_map::residual_interaction_map";

/// Versioned product-neutral result envelope for downstream persistence.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualInteractionMapEnvelope {
    pub schema_version: &'static str,
    pub algorithm_id: &'static str,
    pub implementation_version: &'static str,
    pub calculation_provenance: &'static str,
    /// True only after every returned numeric value has passed a finite-value check.
    pub finite_value_status: bool,
    /// Caller identifiers corresponding to `map.person_indices`.
    pub retained_person_ids: Vec<String>,
    /// Caller identifiers corresponding to `map.item_indices`.
    pub retained_item_ids: Vec<String>,
    /// Caller identifiers for the deterministic closest retained cell.
    pub closest_cell_ids: Option<(String, String)>,
    /// Caller identifiers for the deterministic farthest retained cell.
    pub farthest_cell_ids: Option<(String, String)>,
    /// Rust-owned numerical result; downstream consumers must not recompute it.
    pub map: ResidualInteractionMap,
}

fn validate_identifiers(ids: &[String], expected: usize, axis: &str) -> Result<(), String> {
    if ids.len() != expected {
        return Err(format!(
            "{axis} identifier count must match the declared {axis} dimension"
        ));
    }
    let mut seen = BTreeSet::new();
    for id in ids {
        if !seen.insert(id.as_str()) {
            return Err(format!("duplicate {axis} identifier"));
        }
    }
    Ok(())
}

fn retained_ids(indices: &[usize], ids: &[String], axis: &str) -> Result<Vec<String>, String> {
    indices
        .iter()
        .map(|&index| {
            ids.get(index).cloned().ok_or_else(|| {
                format!("residual interaction map returned an out-of-range {axis} index")
            })
        })
        .collect()
}

fn cell_ids(
    cell: Option<(usize, usize)>,
    person_ids: &[String],
    item_ids: &[String],
) -> Result<Option<(String, String)>, String> {
    let Some((person, item)) = cell else {
        return Ok(None);
    };
    let person_id = person_ids
        .get(person)
        .cloned()
        .ok_or_else(|| "residual interaction map returned an out-of-range person index".to_string())?;
    let item_id = item_ids
        .get(item)
        .cloned()
        .ok_or_else(|| "residual interaction map returned an out-of-range item index".to_string())?;
    Ok(Some((person_id, item_id)))
}

fn map_is_finite(map: &ResidualInteractionMap) -> bool {
    map.person_coordinates.iter().all(|value| value.is_finite())
        && map.item_coordinates.iter().all(|value| value.is_finite())
        && map.singular_values.iter().all(|value| value.is_finite())
        && map.axis_shares.iter().all(|value| value.is_finite())
        && map.observed.iter().all(|value| value.is_finite())
        && map.expected.iter().all(|value| value.is_finite())
        && map.residual.iter().all(|value| value.is_finite())
        && map.distance.iter().all(|value| value.is_finite())
        && map.reconstruction.iter().all(|value| value.is_finite())
        && map.unexplained.iter().all(|value| value.is_finite())
        && map
            .cross_share
            .iter()
            .all(|value| value.map_or(true, |share| share.is_finite()))
}

/// Build one versioned, domain-neutral envelope from Rust-owned map arithmetic.
///
/// Schema and opaque-identifier validation is intentionally completed before
/// numerical work so unsupported consumers and identity mismatches fail closed.
pub fn residual_interaction_map_envelope(
    schema_version: &str,
    person_ids: &[String],
    item_ids: &[String],
    observed: &[f64],
    expected: &[f64],
    n_persons: usize,
    n_items: usize,
    axis_count: usize,
) -> Result<ResidualInteractionMapEnvelope, String> {
    if schema_version != RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION {
        return Err("unsupported residual interaction map schema version".into());
    }
    validate_identifiers(person_ids, n_persons, "person")?;
    validate_identifiers(item_ids, n_items, "item")?;

    let map = residual_interaction_map(observed, expected, n_persons, n_items, axis_count)?;
    if !map_is_finite(&map) {
        return Err("residual interaction map produced non-finite envelope values".into());
    }

    let retained_person_ids = retained_ids(&map.person_indices, person_ids, "person")?;
    let retained_item_ids = retained_ids(&map.item_indices, item_ids, "item")?;
    let closest_cell_ids = cell_ids(map.closest_cell, person_ids, item_ids)?;
    let farthest_cell_ids = cell_ids(map.farthest_cell, person_ids, item_ids)?;

    Ok(ResidualInteractionMapEnvelope {
        schema_version: RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
        algorithm_id: RESIDUAL_INTERACTION_MAP_ALGORITHM_ID,
        implementation_version: RESIDUAL_INTERACTION_MAP_IMPLEMENTATION_VERSION,
        calculation_provenance: RESIDUAL_INTERACTION_MAP_CALCULATION_PROVENANCE,
        finite_value_status: true,
        retained_person_ids,
        retained_item_ids,
        closest_cell_ids,
        farthest_cell_ids,
        map,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn preserves_opaque_ids_and_versioned_provenance() {
        let envelope = residual_interaction_map_envelope(
            RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
            &ids(&["person-0", "person-1"]),
            &ids(&["item-0", "item-1"]),
            &[9.0, f64::NAN, 2.0, 3.0],
            &[8.0, 7.0, 0.5, 1.5],
            2,
            2,
            1,
        )
        .unwrap();

        assert_eq!(envelope.schema_version, RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION);
        assert_eq!(envelope.algorithm_id, RESIDUAL_INTERACTION_MAP_ALGORITHM_ID);
        assert_eq!(
            envelope.implementation_version,
            RESIDUAL_INTERACTION_MAP_IMPLEMENTATION_VERSION
        );
        assert_eq!(
            envelope.calculation_provenance,
            RESIDUAL_INTERACTION_MAP_CALCULATION_PROVENANCE
        );
        assert!(envelope.finite_value_status);
        assert_eq!(envelope.retained_person_ids, ids(&["person-1"]));
        assert_eq!(envelope.retained_item_ids, ids(&["item-0", "item-1"]));
        assert_eq!(
            envelope.closest_cell_ids,
            Some(("person-1".into(), "item-0".into()))
        );
        assert_eq!(
            envelope.farthest_cell_ids,
            Some(("person-1".into(), "item-0".into()))
        );
        assert_eq!(envelope.map.person_indices, vec![1]);
        assert_eq!(envelope.map.item_indices, vec![0, 1]);
    }

    #[test]
    fn rejects_unsupported_schema_before_numerical_validation() {
        let error = residual_interaction_map_envelope(
            "fast-mlsirm.residual-interaction-map.v2",
            &ids(&[]),
            &ids(&[]),
            &[],
            &[],
            usize::MAX,
            usize::MAX,
            0,
        )
        .unwrap_err();
        assert!(error.contains("schema version"));
    }

    #[test]
    fn rejects_duplicate_and_mismatched_identifiers_before_numerical_work() {
        let duplicate = residual_interaction_map_envelope(
            RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
            &ids(&["person", "person"]),
            &ids(&["item"]),
            &[],
            &[],
            2,
            1,
            0,
        )
        .unwrap_err();
        assert!(duplicate.contains("duplicate person identifier"));

        let mismatch = residual_interaction_map_envelope(
            RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
            &ids(&["person"]),
            &ids(&["item"]),
            &[],
            &[],
            2,
            1,
            0,
        )
        .unwrap_err();
        assert!(mismatch.contains("person identifier count"));
    }
}

//! Versioned, domain-neutral envelope around the Rust residual interaction map.
//!
//! This module adds provenance and opaque caller identifiers without moving any
//! interaction-map arithmetic out of `interaction_map`.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use sha2::{Digest, Sha256};

use crate::interaction_map::{residual_interaction_map, ResidualInteractionMap};

/// Current public schema accepted by the Rust envelope constructor.
pub const RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION: &str =
    "fast-mlsirm.residual-interaction-map.v1";
/// Stable algorithm identity for the existing complete-case Gabriel map.
pub const RESIDUAL_INTERACTION_MAP_ALGORITHM_ID: &str =
    "gabriel-complete-case-symmetric-residual-map.v1";
/// Deterministic tie rule for closest/farthest retained-cell selection.
pub const RESIDUAL_INTERACTION_MAP_TIE_POLICY: &str =
    "lexicographic-first-original-index";
/// Package version that produced the envelope.
pub const RESIDUAL_INTERACTION_MAP_IMPLEMENTATION_VERSION: &str = env!("CARGO_PKG_VERSION");
/// Stable calculation-provenance identity for the numerical owner.
pub const RESIDUAL_INTERACTION_MAP_CALCULATION_PROVENANCE: &str =
    "mlsirm-core::interaction_map::residual_interaction_map";

const MAX_INTERACTION_MAP_INPUT_CELLS: usize = 20_000_000;
const MAX_INTERACTION_MAP_AXIS_COUNT: usize = 20_000_000;
const MAX_INTERACTION_MAP_IDENTIFIER_BYTES: usize = 16 * 1024 * 1024;

/// Versioned product-neutral result envelope for downstream persistence.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualInteractionMapEnvelope {
    pub schema_version: &'static str,
    pub algorithm_id: &'static str,
    pub implementation_version: &'static str,
    pub calculation_provenance: &'static str,
    /// SHA-256 identity of the exact validated request evidence.
    pub input_digest: String,
    /// Axis count explicitly requested by the caller and validated by the Rust map.
    pub requested_axis_count: usize,
    /// Public deterministic policy used when closest/farthest distances are tied.
    pub cell_extrema_tie_policy: &'static str,
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

fn validate_input_digest(input_digest: &str) -> Result<(), String> {
    if input_digest.len() != 64
        || !input_digest
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
    {
        return Err("residual interaction map input digest must be lowercase SHA-256".into());
    }
    Ok(())
}

fn validate_identifiers(ids: &[String], expected: usize, axis: &str) -> Result<(), String> {
    if ids.len() != expected {
        return Err(format!(
            "{axis} identifier count must match the declared {axis} dimension"
        ));
    }
    let mut seen = BTreeSet::new();
    let mut identifier_bytes = 0_usize;
    for id in ids {
        identifier_bytes = identifier_bytes
            .checked_add(id.len())
            .ok_or_else(|| format!("{axis} identifier bytes overflow"))?;
        if identifier_bytes > MAX_INTERACTION_MAP_IDENTIFIER_BYTES {
            return Err(format!(
                "{axis} identifier bytes exceed {MAX_INTERACTION_MAP_IDENTIFIER_BYTES}"
            ));
        }
        if !seen.insert(id.as_str()) {
            return Err(format!("duplicate {axis} identifier"));
        }
    }
    Ok(())
}

fn validate_digest_evidence(
    observed: &[f64],
    expected: &[f64],
    n_persons: usize,
    n_items: usize,
    axis_count: usize,
) -> Result<(), String> {
    if axis_count == 0 {
        return Err("axis_count must be positive".into());
    }
    if axis_count > MAX_INTERACTION_MAP_AXIS_COUNT {
        return Err(format!(
            "residual interaction map coordinate request exceeds {MAX_INTERACTION_MAP_AXIS_COUNT} cells"
        ));
    }
    let cell_count = n_persons
        .checked_mul(n_items)
        .ok_or_else(|| "residual interaction map shape overflows".to_string())?;
    if cell_count > MAX_INTERACTION_MAP_INPUT_CELLS {
        return Err(format!(
            "residual interaction map logical-cell count exceeds {MAX_INTERACTION_MAP_INPUT_CELLS}"
        ));
    }
    if observed.len() != cell_count || expected.len() != cell_count {
        return Err("observed and expected lengths must match the declared shape".into());
    }
    if observed.iter().any(|value| value.is_infinite()) {
        return Err("observed values must not be infinite".into());
    }
    if expected.iter().any(|value| !value.is_finite()) {
        return Err("expected values must be finite".into());
    }
    Ok(())
}

fn digest_field(hasher: &mut Sha256, tag: &str, payload: &[u8]) -> Result<(), String> {
    let tag_len = u16::try_from(tag.len())
        .map_err(|_| "residual interaction map digest tag length overflows".to_string())?;
    let payload_len = u64::try_from(payload.len())
        .map_err(|_| "residual interaction map digest payload length overflows".to_string())?;
    hasher.update(tag_len.to_be_bytes());
    hasher.update(tag.as_bytes());
    hasher.update(payload_len.to_be_bytes());
    hasher.update(payload);
    Ok(())
}

fn digest_f64_field(hasher: &mut Sha256, tag: &str, values: &[f64]) -> Result<(), String> {
    let payload_len = values
        .len()
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| "residual interaction map digest payload length overflows".to_string())?;
    let tag_len = u16::try_from(tag.len())
        .map_err(|_| "residual interaction map digest tag length overflows".to_string())?;
    let payload_len = u64::try_from(payload_len)
        .map_err(|_| "residual interaction map digest payload length overflows".to_string())?;
    hasher.update(tag_len.to_be_bytes());
    hasher.update(tag.as_bytes());
    hasher.update(payload_len.to_be_bytes());
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    Ok(())
}

/// Derive the canonical SHA-256 identity for validated interaction-map evidence.
///
/// This is provenance serialization only. It intentionally mirrors the public
/// Python byte contract and does not calculate interaction-map numerics.
pub fn residual_interaction_map_input_digest(
    schema_version: &str,
    person_ids: &[String],
    item_ids: &[String],
    observed: &[f64],
    expected: &[f64],
    n_persons: usize,
    n_items: usize,
    axis_count: usize,
) -> Result<String, String> {
    if schema_version != RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION {
        return Err("unsupported residual interaction map schema version".into());
    }
    validate_identifiers(person_ids, n_persons, "person")?;
    validate_identifiers(item_ids, n_items, "item")?;
    validate_digest_evidence(observed, expected, n_persons, n_items, axis_count)?;

    let axis = u64::try_from(axis_count)
        .map_err(|_| "residual interaction map axis count overflows digest encoding".to_string())?;
    let person_count = u64::try_from(person_ids.len())
        .map_err(|_| "residual interaction map person count overflows digest encoding".to_string())?;
    let item_count = u64::try_from(item_ids.len())
        .map_err(|_| "residual interaction map item count overflows digest encoding".to_string())?;
    let rows = u64::try_from(n_persons)
        .map_err(|_| "residual interaction map row count overflows digest encoding".to_string())?;
    let columns = u64::try_from(n_items)
        .map_err(|_| "residual interaction map column count overflows digest encoding".to_string())?;

    let mut hasher = Sha256::new();
    digest_field(&mut hasher, "schema", schema_version.as_bytes())?;
    digest_field(&mut hasher, "axis_count", &axis.to_be_bytes())?;
    digest_field(&mut hasher, "person_count", &person_count.to_be_bytes())?;
    for identifier in person_ids {
        digest_field(&mut hasher, "person_id", identifier.as_bytes())?;
    }
    digest_field(&mut hasher, "item_count", &item_count.to_be_bytes())?;
    for identifier in item_ids {
        digest_field(&mut hasher, "item_id", identifier.as_bytes())?;
    }
    let mut shape = [0_u8; 16];
    shape[..8].copy_from_slice(&rows.to_be_bytes());
    shape[8..].copy_from_slice(&columns.to_be_bytes());
    digest_field(&mut hasher, "matrix_shape", &shape)?;
    digest_f64_field(&mut hasher, "observed_f64le", observed)?;
    digest_f64_field(&mut hasher, "expected_f64le", expected)?;

    let mut encoded = String::with_capacity(64);
    for byte in hasher.finalize() {
        write!(&mut encoded, "{byte:02x}")
            .map_err(|_| "residual interaction map digest encoding failed".to_string())?;
    }
    Ok(encoded)
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
/// Schema, digest, and opaque-identifier validation is intentionally completed
/// before numerical work so unsupported consumers and identity mismatches fail closed.
pub fn residual_interaction_map_envelope(
    schema_version: &str,
    input_digest: &str,
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
    validate_input_digest(input_digest)?;
    validate_identifiers(person_ids, n_persons, "person")?;
    validate_identifiers(item_ids, n_items, "item")?;
    let canonical_digest = residual_interaction_map_input_digest(
        schema_version,
        person_ids,
        item_ids,
        observed,
        expected,
        n_persons,
        n_items,
        axis_count,
    )?;
    if input_digest != canonical_digest {
        return Err("residual interaction map input digest mismatch".into());
    }

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
        input_digest: canonical_digest,
        requested_axis_count: axis_count,
        cell_extrema_tie_policy: RESIDUAL_INTERACTION_MAP_TIE_POLICY,
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

    const INPUT_DIGEST: &str =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn ids(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn preserves_opaque_ids_and_versioned_provenance() {
        let person_ids = ids(&["person-0", "person-1"]);
        let item_ids = ids(&["item-0", "item-1"]);
        let observed = [9.0, f64::NAN, 2.0, 3.0];
        let expected = [8.0, 7.0, 0.5, 1.5];
        let input_digest = residual_interaction_map_input_digest(
            RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
            &person_ids,
            &item_ids,
            &observed,
            &expected,
            2,
            2,
            1,
        )
        .unwrap();
        let envelope = residual_interaction_map_envelope(
            RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
            &input_digest,
            &person_ids,
            &item_ids,
            &observed,
            &expected,
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
        assert_eq!(envelope.input_digest, input_digest);
        assert_eq!(envelope.requested_axis_count, 1);
        assert_eq!(
            envelope.cell_extrema_tie_policy,
            RESIDUAL_INTERACTION_MAP_TIE_POLICY
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
            INPUT_DIGEST,
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
    fn rejects_malformed_input_digest_before_numerical_validation() {
        let error = residual_interaction_map_envelope(
            RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
            "NOT-A-SHA256",
            &ids(&[]),
            &ids(&[]),
            &[],
            &[],
            usize::MAX,
            usize::MAX,
            0,
        )
        .unwrap_err();
        assert!(error.contains("input digest"));
    }

    #[test]
    fn rejects_duplicate_and_mismatched_identifiers_before_numerical_work() {
        let duplicate = residual_interaction_map_envelope(
            RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
            INPUT_DIGEST,
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
            INPUT_DIGEST,
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

    #[test]
    fn rejects_valid_but_foreign_digest_before_numerical_work() {
        let error = residual_interaction_map_envelope(
            RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
            INPUT_DIGEST,
            &ids(&["person"]),
            &ids(&["item"]),
            &[2.0],
            &[1.0],
            1,
            1,
            1,
        )
        .unwrap_err();
        assert!(error.contains("input digest mismatch"));
    }
}

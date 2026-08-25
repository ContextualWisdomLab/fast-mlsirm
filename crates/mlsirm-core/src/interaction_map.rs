//! Gabriel residual interaction maps after an externally fitted IRT main effect.
//!
//! The caller supplies observed responses and their model expectations. This
//! module owns the numerical residual factorization; product repositories own
//! identifiers, authorization, persistence, and presentation.

use crate::factor::symmetric_eigen_desc;

const SINGULAR_FLOOR: f64 = 1e-12;
const MAX_INTERACTION_MAP_CELLS: usize = 20_000_000;
const MAX_INTERACTION_MAP_COORDINATE_CELLS: usize = 20_000_000;
const MAX_INTERACTION_MAP_EIGEN_WORKSPACE_BYTES: usize = 128 * 1024 * 1024;
const EIGEN_WORKSPACE_MATRIX_COUNT: usize = 3;

fn validate_factorization_workspace(
    rows: usize,
    columns: usize,
    axis_count: usize,
) -> Result<(), String> {
    let coordinate_cells = rows
        .checked_add(columns)
        .and_then(|dimension_sum| dimension_sum.checked_mul(axis_count))
        .ok_or_else(|| "residual interaction map coordinate request overflows".to_string())?;
    if coordinate_cells > MAX_INTERACTION_MAP_COORDINATE_CELLS {
        return Err(format!(
            "residual interaction map coordinate request exceeds {MAX_INTERACTION_MAP_COORDINATE_CELLS} cells"
        ));
    }

    // `symmetric_eigen_desc` holds the caller's Gram matrix plus an internal
    // matrix copy and eigenvector matrix at peak, so budget all three dense
    // `columns x columns` f64 matrices before allocating any of them.
    let eigen_workspace_bytes = columns
        .checked_mul(columns)
        .and_then(|cells| cells.checked_mul(std::mem::size_of::<f64>()))
        .and_then(|bytes| bytes.checked_mul(EIGEN_WORKSPACE_MATRIX_COUNT))
        .ok_or_else(|| "residual interaction map eigen workspace overflows".to_string())?;
    if eigen_workspace_bytes > MAX_INTERACTION_MAP_EIGEN_WORKSPACE_BYTES {
        return Err(format!(
            "residual interaction map eigen workspace exceeds {} MiB",
            MAX_INTERACTION_MAP_EIGEN_WORKSPACE_BYTES / (1024 * 1024)
        ));
    }
    Ok(())
}

/// A complete-case Gabriel biplot of `observed - expected`.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualInteractionMap {
    pub person_indices: Vec<usize>,
    pub item_indices: Vec<usize>,
    pub scored_person_count: usize,
    pub scored_item_count: usize,
    pub person_coordinates: Vec<f64>,
    pub item_coordinates: Vec<f64>,
    pub singular_values: Vec<f64>,
    pub axis_shares: Vec<f64>,
    pub residual: Vec<f64>,
    pub distance: Vec<f64>,
    pub reconstruction: Vec<f64>,
    pub unexplained: Vec<f64>,
    pub cross_share: Vec<Option<f64>>,
    pub axis_count: usize,
}

/// Factor a complete-case residual rectangle using Gabriel's symmetric scaling.
pub fn residual_interaction_map(
    observed: &[f64],
    expected: &[f64],
    n_persons: usize,
    n_items: usize,
    axis_count: usize,
) -> Result<ResidualInteractionMap, String> {
    if axis_count == 0 {
        return Err("axis_count must be positive".into());
    }
    // Even an empty complete-case rectangle returns `axis_shares`, so bound the
    // requested axis vector independently of the surviving matrix dimensions.
    if axis_count > MAX_INTERACTION_MAP_COORDINATE_CELLS {
        return Err(format!(
            "residual interaction map coordinate request exceeds {MAX_INTERACTION_MAP_COORDINATE_CELLS} cells"
        ));
    }

    let cell_count = n_persons
        .checked_mul(n_items)
        .ok_or_else(|| "residual interaction map shape overflows".to_string())?;
    if cell_count > MAX_INTERACTION_MAP_CELLS {
        return Err(format!(
            "residual interaction map logical-cell count exceeds {MAX_INTERACTION_MAP_CELLS}"
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

    let observed_cell = |index: usize| observed[index].is_finite();
    let mut person_indices: Vec<usize> = (0..n_persons)
        .filter(|&person| (0..n_items).any(|item| observed_cell(person * n_items + item)))
        .collect();
    let mut item_indices: Vec<usize> = (0..n_items)
        .filter(|&item| (0..n_persons).any(|person| observed_cell(person * n_items + item)))
        .collect();
    let scored_person_count = person_indices.len();
    let scored_item_count = item_indices.len();
    person_indices.retain(|&person| {
        item_indices
            .iter()
            .all(|&item| observed_cell(person * n_items + item))
    });
    item_indices.retain(|&item| {
        person_indices
            .iter()
            .all(|&person| observed_cell(person * n_items + item))
    });
    if person_indices.is_empty() || item_indices.is_empty() {
        // An empty complete-case rectangle has no row or column coordinate
        // space. Normalize both axes to the same empty rectangle so every
        // returned payload remains shape-consistent at the Python boundary.
        person_indices.clear();
        item_indices.clear();
        return Ok(ResidualInteractionMap {
            person_indices,
            item_indices,
            scored_person_count,
            scored_item_count,
            person_coordinates: Vec::new(),
            item_coordinates: Vec::new(),
            singular_values: Vec::new(),
            axis_shares: vec![0.0; axis_count],
            residual: Vec::new(),
            distance: Vec::new(),
            reconstruction: Vec::new(),
            unexplained: Vec::new(),
            cross_share: Vec::new(),
            axis_count,
        });
    }

    let rows = person_indices.len();
    let columns = item_indices.len();
    validate_factorization_workspace(rows, columns, axis_count)?;

    let mut residual = Vec::with_capacity(rows * columns);
    for &person in &person_indices {
        for &item in &item_indices {
            let index = person * n_items + item;
            residual.push(observed[index] - expected[index]);
        }
    }
    let center = residual.iter().sum::<f64>() / residual.len() as f64;
    let centered: Vec<f64> = residual.iter().map(|value| value - center).collect();
    let mut gram = vec![0.0; columns * columns];
    for left in 0..columns {
        for right in left..columns {
            let value = (0..rows)
                .map(|row| centered[row * columns + left] * centered[row * columns + right])
                .sum();
            gram[left * columns + right] = value;
            gram[right * columns + left] = value;
        }
    }
    let (eigenvalues, eigenvectors) = symmetric_eigen_desc(&gram, columns)?;
    let singular_values: Vec<f64> = eigenvalues
        .iter()
        .take_while(|value| **value > SINGULAR_FLOOR * SINGULAR_FLOOR)
        .map(|value| value.sqrt())
        .collect();
    let retained = singular_values.len();
    let mut person_coordinates = vec![0.0; rows * axis_count];
    let mut item_coordinates = vec![0.0; columns * axis_count];
    for axis in 0..axis_count.min(retained) {
        let root_singular = singular_values[axis].sqrt();
        for item in 0..columns {
            item_coordinates[item * axis_count + axis] =
                eigenvectors[item * columns + axis] * root_singular;
        }
        for person in 0..rows {
            let projection: f64 = (0..columns)
                .map(|item| centered[person * columns + item] * eigenvectors[item * columns + axis])
                .sum();
            person_coordinates[person * axis_count + axis] = projection / root_singular;
        }
    }
    let inertia = singular_values
        .iter()
        .map(|value| value * value)
        .sum::<f64>();
    let axis_shares = (0..axis_count)
        .map(|axis| {
            singular_values
                .get(axis)
                .map_or(0.0, |value| value * value / inertia)
        })
        .collect();
    let mut reconstruction = Vec::with_capacity(rows * columns);
    let mut distance = Vec::with_capacity(rows * columns);
    let mut unexplained = Vec::with_capacity(rows * columns);
    let mut cross_share = Vec::with_capacity(rows * columns);
    for person in 0..rows {
        for item in 0..columns {
            distance.push(
                (0..axis_count)
                    .map(|axis| {
                        let difference = person_coordinates[person * axis_count + axis]
                            - item_coordinates[item * axis_count + axis];
                        difference * difference
                    })
                    .sum::<f64>()
                    .sqrt(),
            );
            let fitted = (0..axis_count)
                .map(|axis| {
                    person_coordinates[person * axis_count + axis]
                        * item_coordinates[item * axis_count + axis]
                })
                .sum::<f64>();
            let raw = residual[person * columns + item];
            let remainder = raw - fitted;
            let share = if raw.abs() > SINGULAR_FLOOR {
                Some(2.0 * fitted * remainder / (raw * raw))
            } else if fitted.abs() <= SINGULAR_FLOOR && remainder.abs() <= SINGULAR_FLOOR {
                Some(0.0)
            } else {
                None
            };
            reconstruction.push(fitted);
            unexplained.push(remainder);
            cross_share.push(share.filter(|value| value.is_finite()));
        }
    }

    Ok(ResidualInteractionMap {
        person_indices,
        item_indices,
        scored_person_count,
        scored_item_count,
        person_coordinates,
        item_coordinates,
        singular_values,
        axis_shares,
        residual,
        distance,
        reconstruction,
        unexplained,
        cross_share,
        axis_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstructs_rank_one_residual_without_inventing_a_second_axis() {
        let observed = [2.0, 0.0, 0.0, 2.0];
        let expected = [1.0, 1.0, 1.0, 1.0];
        let map = residual_interaction_map(&observed, &expected, 2, 2, 2).unwrap();
        assert_eq!(map.person_indices, vec![0, 1]);
        assert_eq!(map.item_indices, vec![0, 1]);
        assert_eq!(map.effective_rank, 1);
        assert_eq!(map.map_person_count, 2);
        assert_eq!(map.map_item_count, 2);
        assert_eq!(map.incomplete_person_count, 0);
        assert_eq!(map.incomplete_item_count, 0);
        assert_eq!(map.singular_values.len(), 1);
        assert!((map.axis_shares[0] - 1.0).abs() < 1e-12);
        assert_eq!(map.axis_shares[1], 0.0);
        for (raw, fitted) in observed
            .iter()
            .zip(expected)
            .map(|(observed, expected)| observed - expected)
            .zip(&map.reconstruction)
        {
            assert!((raw - fitted).abs() < 1e-12);
        }
        assert!(map
            .distance
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
    }

    #[test]
    fn excludes_incomplete_rows_without_zero_filling() {
        let map =
            residual_interaction_map(&[2.0, f64::NAN, 1.0, 2.0], &[1.0, 1.0, 1.0, 1.0], 2, 2, 2)
                .unwrap();
        assert_eq!(map.person_indices, vec![1]);
        assert_eq!(map.item_indices, vec![0, 1]);
        assert_eq!(map.scored_person_count, 2);
        assert_eq!(map.scored_item_count, 2);
        assert_eq!(map.map_person_count, 1);
        assert_eq!(map.map_item_count, 2);
        assert_eq!(map.incomplete_person_count, 1);
        assert_eq!(map.incomplete_item_count, 0);
    }

    #[test]
    fn empty_complete_case_rectangle_has_empty_indices_on_both_axes() {
        let map = residual_interaction_map(
            &[1.0, f64::NAN, f64::NAN, 1.0],
            &[0.0, 0.0, 0.0, 0.0],
            2,
            2,
            2,
        )
        .unwrap();
        assert!(map.person_indices.is_empty());
        assert!(map.item_indices.is_empty());
        assert_eq!(map.effective_rank, 0);
        assert_eq!(map.map_person_count, 0);
        assert_eq!(map.map_item_count, 0);
        assert_eq!(map.incomplete_person_count, map.scored_person_count);
        assert_eq!(map.incomplete_item_count, map.scored_item_count);
        assert_eq!(map.closest_cell, None);
        assert_eq!(map.farthest_cell, None);
        assert!(map.person_coordinates.is_empty());
        assert!(map.item_coordinates.is_empty());
        assert!(map.reconstruction.is_empty());
        assert!(map.unexplained.is_empty());
        assert!(map.cross_share.is_empty());
        assert_eq!(map.axis_shares, vec![0.0, 0.0]);
    }

    #[test]
    fn deterministic_cell_extrema_use_lexicographic_ties() {
        let map = residual_interaction_map(&[1.0, 1.0, 1.0, 1.0], &[1.0, 1.0, 1.0, 1.0], 2, 2, 2)
            .unwrap();
        assert_eq!(map.distance, vec![0.0; 4]);
        assert_eq!(map.closest_cell, Some((0, 0)));
        assert_eq!(map.farthest_cell, Some((0, 0)));
    }

    #[test]
    fn rejects_nonfinite_expected_before_complete_case_filtering() {
        let error = residual_interaction_map(&[1.0], &[f64::NAN], 1, 1, 1).unwrap_err();
        assert!(error.contains("expected"));
    }

    #[test]
    fn rejects_infinite_observed_instead_of_treating_it_as_missing() {
        let error = residual_interaction_map(&[f64::INFINITY], &[0.0], 1, 1, 1).unwrap_err();
        assert!(error.contains("observed"));
    }

    #[test]
    fn rejects_declared_grid_above_resource_ceiling_before_length_validation() {
        let error = residual_interaction_map(&[], &[], 1, 20_000_001, 1).unwrap_err();
        assert!(error.contains("logical-cell"));
    }

    #[test]
    fn rejects_coordinate_overflow_before_allocation() {
        let call = std::panic::catch_unwind(|| {
            residual_interaction_map(&[1.0, 1.0], &[0.0, 0.0], 2, 1, usize::MAX)
        });
        assert!(
            call.is_ok(),
            "resource admission must return Err instead of panicking"
        );
        let error = call.unwrap().unwrap_err();
        assert!(error.contains("coordinate"));
    }

    #[test]
    fn rejects_quadratic_eigen_workspace_before_allocation() {
        let error = validate_factorization_workspace(1, 3_000, 1).unwrap_err();
        assert!(error.contains("eigen workspace"));
        validate_factorization_workspace(1, 2_000, 1).unwrap();
    }
}

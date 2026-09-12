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
    /// Numerical rank of the complete-case centered residual matrix.
    pub effective_rank: usize,
    /// Number of respondents retained in the complete-case map rectangle.
    pub map_person_count: usize,
    /// Number of items retained in the complete-case map rectangle.
    pub map_item_count: usize,
    /// Scored respondents excluded from the complete-case map rectangle.
    pub incomplete_person_count: usize,
    /// Scored items excluded from the complete-case map rectangle.
    pub incomplete_item_count: usize,
    /// Lexicographically first retained cell at the minimum requested-axis distance.
    pub closest_cell: Option<(usize, usize)>,
    /// Lexicographically first retained cell at the maximum requested-axis distance.
    pub farthest_cell: Option<(usize, usize)>,
    pub person_coordinates: Vec<f64>,
    pub item_coordinates: Vec<f64>,
    pub singular_values: Vec<f64>,
    pub axis_shares: Vec<f64>,
    /// Complete-case observed values in retained person-major/item-minor order.
    pub observed: Vec<f64>,
    /// Complete-case model expectations in retained person-major/item-minor order.
    pub expected: Vec<f64>,
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
            effective_rank: 0,
            map_person_count: 0,
            map_item_count: 0,
            incomplete_person_count: scored_person_count,
            incomplete_item_count: scored_item_count,
            closest_cell: None,
            farthest_cell: None,
            person_coordinates: Vec::new(),
            item_coordinates: Vec::new(),
            singular_values: Vec::new(),
            axis_shares: vec![0.0; axis_count],
            observed: Vec::new(),
            expected: Vec::new(),
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

    let mut retained_observed = Vec::with_capacity(rows * columns);
    let mut retained_expected = Vec::with_capacity(rows * columns);
    let mut residual = Vec::with_capacity(rows * columns);
    for &person in &person_indices {
        for &item in &item_indices {
            let index = person * n_items + item;
            retained_observed.push(observed[index]);
            retained_expected.push(expected[index]);
            let value = observed[index] - expected[index];
            if !value.is_finite() {
                return Err("residual interaction map produced a non-finite residual".into());
            }
            residual.push(value);
        }
    }
    let residual_sum = residual.iter().sum::<f64>();
    if !residual_sum.is_finite() {
        return Err("residual interaction map produced a non-finite centering sum".into());
    }
    let center = residual_sum / residual.len() as f64;
    if !center.is_finite() {
        return Err("residual interaction map produced a non-finite center".into());
    }
    let mut centered = Vec::with_capacity(residual.len());
    for value in &residual {
        let centered_value = *value - center;
        if !centered_value.is_finite() {
            return Err("residual interaction map produced a non-finite centered residual".into());
        }
        centered.push(centered_value);
    }
    let mut gram = vec![0.0; columns * columns];
    for left in 0..columns {
        for right in left..columns {
            let value = (0..rows)
                .map(|row| centered[row * columns + left] * centered[row * columns + right])
                .sum::<f64>();
            if !value.is_finite() {
                return Err("residual interaction map produced a non-finite Gram value".into());
            }
            gram[left * columns + right] = value;
            gram[right * columns + left] = value;
        }
    }
    // The shared Jacobi kernel has an absolute convergence tolerance.  A
    // positive scalar multiple has exactly the same eigenvectors, so put the
    // Gram matrix on an O(1) scale before decomposition and restore only the
    // eigenvalues afterward.  This keeps the solver scale-invariant without
    // relaxing numerical-rank admission or allocating another dense matrix.
    let gram_scale = gram
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if gram_scale > 0.0 {
        for value in &mut gram {
            *value /= gram_scale;
        }
    }
    let (mut eigenvalues, eigenvectors) = symmetric_eigen_desc(&gram, columns)?;
    if gram_scale > 0.0 {
        for value in &mut eigenvalues {
            *value *= gram_scale;
        }
    }
    if eigenvalues.iter().any(|value| !value.is_finite())
        || eigenvectors.iter().any(|value| !value.is_finite())
    {
        return Err("residual interaction map eigendecomposition produced non-finite values".into());
    }
    let maximum_rank = rows.min(columns);
    let leading_singular = eigenvalues
        .first()
        .copied()
        .unwrap_or(0.0)
        .max(0.0)
        .sqrt();
    if !leading_singular.is_finite() {
        return Err("residual interaction map produced a non-finite leading singular value".into());
    }
    // This implementation obtains singular values from the Gram matrix, so
    // eigensolver roundoff is in squared-singular-value space. Convert that
    // bound back to the singular-value scale with sqrt(eps * dimension), while
    // retaining the historical absolute floor and the algebraic rank ceiling.
    let gram_roundoff = (f64::EPSILON * rows.max(columns) as f64).sqrt();
    let numerical_singular_floor =
        SINGULAR_FLOOR.max(leading_singular * gram_roundoff);
    let singular_values: Vec<f64> = eigenvalues
        .iter()
        .take(maximum_rank)
        .take_while(|value| **value > 0.0 && (**value).sqrt() > numerical_singular_floor)
        .map(|value| value.sqrt())
        .collect();
    let retained = singular_values.len();
    let mut person_coordinates = vec![0.0; rows * axis_count];
    let mut item_coordinates = vec![0.0; columns * axis_count];
    for axis in 0..axis_count.min(retained) {
        let root_singular = singular_values[axis].sqrt();
        for item in 0..columns {
            let coordinate = eigenvectors[item * columns + axis] * root_singular;
            if !coordinate.is_finite() {
                return Err("residual interaction map produced a non-finite item coordinate".into());
            }
            item_coordinates[item * axis_count + axis] = coordinate;
        }
        for person in 0..rows {
            let projection: f64 = (0..columns)
                .map(|item| centered[person * columns + item] * eigenvectors[item * columns + axis])
                .sum();
            let coordinate = projection / root_singular;
            if !projection.is_finite() || !coordinate.is_finite() {
                return Err("residual interaction map produced a non-finite person coordinate".into());
            }
            person_coordinates[person * axis_count + axis] = coordinate;
        }
    }
    let inertia = singular_values
        .iter()
        .map(|value| value * value)
        .sum::<f64>();
    if !inertia.is_finite() {
        return Err("residual interaction map produced non-finite inertia".into());
    }
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
    let mut closest_cell = None;
    let mut farthest_cell = None;
    let mut closest_distance = f64::INFINITY;
    let mut farthest_distance = f64::NEG_INFINITY;
    for person in 0..rows {
        for item in 0..columns {
            let cell_distance = (0..axis_count)
                .map(|axis| {
                    let difference = person_coordinates[person * axis_count + axis]
                        - item_coordinates[item * axis_count + axis];
                    difference * difference
                })
                .sum::<f64>()
                .sqrt();
            if !cell_distance.is_finite() {
                return Err("residual interaction map produced a non-finite distance".into());
            }
            // `person_indices` and `item_indices` are ascending, so strict
            // comparisons retain the lexicographically first cell on ties.
            let cell_identity = (person_indices[person], item_indices[item]);
            if cell_distance < closest_distance {
                closest_distance = cell_distance;
                closest_cell = Some(cell_identity);
            }
            if cell_distance > farthest_distance {
                farthest_distance = cell_distance;
                farthest_cell = Some(cell_identity);
            }
            distance.push(cell_distance);
            let fitted = (0..axis_count)
                .map(|axis| {
                    person_coordinates[person * axis_count + axis]
                        * item_coordinates[item * axis_count + axis]
                })
                .sum::<f64>();
            let raw = residual[person * columns + item];
            let remainder = raw - fitted;
            if !fitted.is_finite() || !remainder.is_finite() {
                return Err("residual interaction map produced a non-finite reconstruction".into());
            }
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
        effective_rank: retained,
        map_person_count: rows,
        map_item_count: columns,
        incomplete_person_count: scored_person_count - rows,
        incomplete_item_count: scored_item_count - columns,
        closest_cell,
        farthest_cell,
        person_coordinates,
        item_coordinates,
        singular_values,
        axis_shares,
        observed: retained_observed,
        expected: retained_expected,
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
        assert!(map.observed.is_empty());
        assert!(map.expected.is_empty());
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
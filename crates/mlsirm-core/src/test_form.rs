//! Fixed-form greedy maximum-information assembly with content constraints.
//!
//! Public Python `assemble_test_form` validates and marshals; ordering,
//! exclusion, and content-feasibility decisions are owned here so form
//! construction stays single-sourced on the compiled numeric path.
//!
//! The procedure ranks eligible items by Fisher (or precomputed) information
//! descending and greedily admits the next item that preserves look-ahead
//! feasibility of minimum content counts under maximum caps (van der Linden,
//! 2005, ch. 4 greedy heuristic for constrained assembly).
//!
//! # References (APA 7th ed.)
//!
//! van der Linden, W. J. (2005). *Linear models for optimal test design*.
//! Springer. https://doi.org/10.1007/0-387-29054-0

use std::collections::{HashMap, HashSet};

/// Assemble a fixed-length form by greedy maximum-information selection.
///
/// `information` is one finite-or-nonfinite score per item. Non-finite scores
/// and indices listed in `exclude` are skipped. When `content` is `Some`, it
/// must have the same length as `information`; empty `min`/`max` maps mean no
/// constraints of that kind. Returns selected item indices in admission order.
pub fn assemble_test_form_greedy(
    information: &[f64],
    length: usize,
    content: Option<&[String]>,
    min_per_content: &HashMap<String, i64>,
    max_per_content: &HashMap<String, i64>,
    exclude: &[i64],
) -> Result<Vec<i64>, String> {
    let n = information.len();
    if n == 0 {
        return Err("information must be a non-empty 1D array".into());
    }
    if length < 1 || length > n {
        return Err("length must be between 1 and the number of items".into());
    }
    if let Some(labels) = content {
        if labels.len() != n {
            return Err("content length must match information".into());
        }
    } else if !min_per_content.is_empty() || !max_per_content.is_empty() {
        return Err("content labels are required for content constraints".into());
    }
    for (label, &minimum) in min_per_content {
        if minimum < 0 {
            return Err(format!("minimum content constraint for {label} must be non-negative"));
        }
        if let Some(&maximum) = max_per_content.get(label) {
            if minimum > maximum {
                return Err(format!(
                    "minimum content constraint cannot exceed maximum for {label}"
                ));
            }
        }
    }
    for (label, &maximum) in max_per_content {
        if maximum < 0 {
            return Err(format!("maximum content constraint for {label} must be non-negative"));
        }
    }

    let mut excluded: HashSet<usize> = HashSet::new();
    for &raw in exclude {
        if raw < 0 {
            return Err("exclude indices must be non-negative".into());
        }
        let idx = raw as usize;
        if idx >= n {
            return Err("exclude index out of range".into());
        }
        excluded.insert(idx);
    }

    // Descending information, stable on ties by ascending index (deterministic).
    let mut order: Vec<usize> = (0..n)
        .filter(|&i| !excluded.contains(&i) && information[i].is_finite())
        .collect();
    order.sort_by(|&a, &b| {
        information[b]
            .partial_cmp(&information[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let mut selected: Vec<usize> = Vec::with_capacity(length);
    let mut counts: HashMap<String, i64> = HashMap::new();
    let length_i = length as i64;

    for _ in 0..length {
        let mut admitted = false;
        for &item in &order {
            if selected.contains(&item) {
                continue;
            }
            let label = content.map(|labels| labels[item].as_str());
            let mut next_counts = counts.clone();
            if let Some(label) = label {
                let cap = max_per_content
                    .get(label)
                    .copied()
                    .unwrap_or(length_i);
                let current = next_counts.get(label).copied().unwrap_or(0);
                if current >= cap {
                    continue;
                }
                next_counts.insert(label.to_owned(), current + 1);
            }
            if constraints_feasible(
                &order,
                &selected,
                item,
                &excluded,
                content,
                &next_counts,
                length,
                min_per_content,
                max_per_content,
            ) {
                selected.push(item);
                counts = next_counts;
                admitted = true;
                break;
            }
        }
        if !admitted {
            return Err("could not assemble a form that satisfies constraints".into());
        }
    }

    for (label, &minimum) in min_per_content {
        let have = counts.get(label).copied().unwrap_or(0);
        if have < minimum {
            // Defensive: look-ahead should already enforce minima.
            return Err(format!("minimum content constraint not met: {label}"));
        }
    }

    Ok(selected.into_iter().map(|i| i as i64).collect())
}

fn constraints_feasible(
    order: &[usize],
    selected: &[usize],
    candidate: usize,
    excluded: &HashSet<usize>,
    content: Option<&[String]>,
    counts: &HashMap<String, i64>,
    length: usize,
    min_counts: &HashMap<String, i64>,
    max_counts: &HashMap<String, i64>,
) -> bool {
    let mut trial_selected = selected.to_vec();
    trial_selected.push(candidate);
    let slots_left = length.saturating_sub(trial_selected.len()) as i64;
    let required_left: i64 = min_counts
        .iter()
        .map(|(label, &minimum)| (minimum - counts.get(label).copied().unwrap_or(0)).max(0))
        .sum();
    if required_left > slots_left {
        return false;
    }
    let Some(labels) = content else {
        return true;
    };
    let length_i = length as i64;
    let blocked: HashSet<usize> = trial_selected
        .iter()
        .copied()
        .chain(excluded.iter().copied())
        .collect();
    for (label, &minimum) in min_counts {
        let needed = (minimum - counts.get(label).copied().unwrap_or(0)).max(0);
        if needed == 0 {
            continue;
        }
        let cap = max_counts.get(label).copied().unwrap_or(length_i);
        let mut available: i64 = 0;
        for &item in order {
            if blocked.contains(&item) || labels[item] != *label {
                continue;
            }
            // counts already include the candidate when label matches.
            if counts.get(label).copied().unwrap_or(0) + available >= cap {
                break;
            }
            available += 1;
        }
        if available < needed {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_map() -> HashMap<String, i64> {
        HashMap::new()
    }

    #[test]
    fn unconstrained_picks_highest_information() {
        let info = [5.0, 4.0, 3.0, 2.0, 1.0];
        let form = assemble_test_form_greedy(&info, 3, None, &empty_map(), &empty_map(), &[])
            .expect("form");
        assert_eq!(form, vec![0, 1, 2]);
    }

    #[test]
    fn exclude_skips_top_item() {
        let info = [5.0, 4.0, 3.0, 2.0, 1.0];
        let form =
            assemble_test_form_greedy(&info, 2, None, &empty_map(), &empty_map(), &[0]).expect("form");
        assert_eq!(form, vec![1, 2]);
    }

    #[test]
    fn content_min_max_respected() {
        let info = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let content = ["a", "a", "b", "b", "c", "c"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let min = HashMap::from([("c".into(), 1)]);
        let max = HashMap::from([("a".into(), 1)]);
        let form =
            assemble_test_form_greedy(&info, 3, Some(&content), &min, &max, &[]).expect("form");
        let mut a = 0;
        let mut c = 0;
        for &idx in &form {
            match content[idx as usize].as_str() {
                "a" => a += 1,
                "c" => c += 1,
                _ => {}
            }
        }
        assert!(a <= 1);
        assert!(c >= 1);
    }

    #[test]
    fn infeasible_min_raises() {
        let info = [5.0, 4.0];
        let content = ["a".into(), "a".into()];
        let min = HashMap::from([("b".into(), 1)]);
        let err = assemble_test_form_greedy(&info, 2, Some(&content), &min, &empty_map(), &[])
            .unwrap_err();
        assert!(err.contains("could not assemble") || err.contains("constraint"));
    }

    #[test]
    fn ownership_sentinel_shape_matches_public_contract() {
        // Length-2 form with B-min / A-max and exclude=3 should be feasible
        // and prefer high-info B then residual A when constraints force it.
        let info = [1.0, 4.0, 3.0, 2.0];
        let content = ["A", "A", "B", "B"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let min = HashMap::from([("B".into(), 1)]);
        let max = HashMap::from([("A".into(), 1)]);
        let form =
            assemble_test_form_greedy(&info, 2, Some(&content), &min, &max, &[3]).expect("form");
        assert_eq!(form.len(), 2);
        assert!(!form.contains(&3));
        // Item 1 (A, info=4) is highest remaining; item 2 (B, info=3) satisfies min B.
        // Greedy admits highest first when still feasible: 1 then 2.
        assert_eq!(form, vec![1, 2]);
    }
}

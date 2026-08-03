#!/usr/bin/env python3
"""Apply two exact mathematical contract corrections before compilation."""

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    """Replace one exact source block and reject ambiguous patch state."""
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"{path}: expected one exact mathematical patch marker")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    """Correct the GPA oblique chain rule and NaN-partial target alignment."""
    replace_once(
        "crates/mlsirm-core/src/rotation/optimizer.rs",
        """    let product = matmul(
        &pattern_gradient,
        factors,
        factors,
        inverse_transform,
        factors,
    );
    let mut gradient = transpose(&product, factors, factors);
    for value in &mut gradient {
        *value = -*value;
    }
""",
        """    // GPArotation's oblique chain rule is
    // G = -t(T^{-T} Gq' A T^{-T}) = -T^{-1} A' Gq T^{-1}.
    // `pattern_gradient` is already T^{-1} A' Gq, so the product is the
    // transform-space gradient and must not be transposed a second time.
    let mut gradient = matmul(
        &pattern_gradient,
        factors,
        factors,
        inverse_transform,
        factors,
    );
    for value in &mut gradient {
        *value = -*value;
    }
""",
    )
    replace_once(
        "crates/mlsirm-core/src/rotation/selector.rs",
        """fn aligned_target_rmse(
    pattern: &[f64],
    target: &[f64],
    rows: usize,
    factors: usize,
) -> Result<f64, String> {
    let aligned = align_to_reference(pattern, target, rows, factors, false)?;
    let mut sum = 0.0;
    let mut count = 0_usize;
    for (value, target_value) in aligned.iter().zip(target) {
        if target_value.is_nan() {
            continue;
        }
        sum += (value - target_value).powi(2);
        count += 1;
    }
    if count == 0 {
        return Err("theory_target contains no specified cells".into());
    }
    Ok((sum / count as f64).sqrt())
}
""",
        """fn aligned_target_rmse(
    pattern: &[f64],
    target: &[f64],
    rows: usize,
    factors: usize,
) -> Result<f64, String> {
    let (assignment, signs) = partial_target_assignment(pattern, target, rows, factors)?;
    let mut sum = 0.0;
    let mut count = 0_usize;
    for target_column in 0..factors {
        let pattern_column = assignment[target_column];
        for i in 0..rows {
            let target_value = target[i * factors + target_column];
            if target_value.is_nan() {
                continue;
            }
            let value = signs[target_column] * pattern[i * factors + pattern_column];
            sum += (value - target_value).powi(2);
            count += 1;
        }
    }
    if count == 0 {
        return Err("theory_target contains no specified cells".into());
    }
    Ok((sum / count as f64).sqrt())
}

fn partial_target_assignment(
    pattern: &[f64],
    target: &[f64],
    rows: usize,
    factors: usize,
) -> Result<(Vec<usize>, Vec<f64>), String> {
    let mut assignment = vec![usize::MAX; factors];
    let mut signs = vec![1.0; factors];
    let mut used = vec![false; factors];
    for target_column in 0..factors {
        let mut best: Option<(usize, f64, f64)> = None;
        for pattern_column in 0..factors {
            if used[pattern_column] {
                continue;
            }
            let mut dot = 0.0;
            let mut pattern_ss = 0.0;
            let mut target_ss = 0.0;
            let mut specified = 0_usize;
            for i in 0..rows {
                let target_value = target[i * factors + target_column];
                if target_value.is_nan() {
                    continue;
                }
                let pattern_value = pattern[i * factors + pattern_column];
                dot += target_value * pattern_value;
                pattern_ss += pattern_value * pattern_value;
                target_ss += target_value * target_value;
                specified += 1;
            }
            if specified == 0 || pattern_ss <= 1e-24 || target_ss <= 1e-24 {
                continue;
            }
            let congruence = dot / (pattern_ss * target_ss).sqrt();
            if best
                .as_ref()
                .map(|(_, current, _)| congruence.abs() > *current)
                .unwrap_or(true)
            {
                best = Some((pattern_column, congruence.abs(), congruence.signum()));
            }
        }
        let (selected, _, sign) = best.ok_or_else(|| {
            format!("theory_target column {target_column} has insufficient specified variance")
        })?;
        assignment[target_column] = selected;
        signs[target_column] = if sign == 0.0 { 1.0 } else { sign };
        used[selected] = true;
    }
    Ok((assignment, signs))
}
""",
    )


if __name__ == "__main__":
    main()

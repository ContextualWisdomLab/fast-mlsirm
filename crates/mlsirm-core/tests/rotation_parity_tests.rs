#[test]
fn simple_structure_metrics_deterministic_f64_parity() {
    fn reference_metrics(pattern: &[f64], rows: usize, factors: usize, skip_general: bool) -> (f64, f64) {
        let first = usize::from(skip_general);
        let active = factors - first;
        let mut complexity_numerator = 0.0;
        let mut complexity_denominator = 0.0;
        let mut factor_ss = vec![0.0; active];
        for i in 0..rows {
            let row = &pattern[i * factors + first..(i + 1) * factors];
            let row_ss: f64 = row.iter().map(|x| x * x).sum();
            let row_fourth: f64 = row.iter().map(|x| x.powi(4)).sum();
            complexity_numerator += row_ss * row_ss - row_fourth;
            complexity_denominator += row_ss * row_ss;
            for (j, value) in row.iter().enumerate() {
                factor_ss[j] += value * value;
            }
        }
        let row_complexity = if complexity_denominator > 0.0 {
            complexity_numerator / complexity_denominator
        } else {
            1.0
        };
        let minimum = factor_ss.iter().copied().fold(f64::INFINITY, f64::min);
        let maximum = factor_ss.iter().copied().fold(0.0_f64, f64::max);
        let factor_balance = if maximum > 0.0 {
            minimum / maximum
        } else {
            0.0
        };
        (row_complexity, factor_balance)
    }

    fn optimized_metrics(pattern: &[f64], rows: usize, factors: usize, skip_general: bool) -> (f64, f64) {
        let first = usize::from(skip_general);
        let active = factors - first;
        let mut complexity_numerator = 0.0;
        let mut complexity_denominator = 0.0;
        let mut factor_ss = vec![0.0; active];
        for i in 0..rows {
            let row = &pattern[i * factors + first..(i + 1) * factors];
            let (row_ss, row_fourth) = row.iter().fold((0.0, 0.0), |(s2, s4), &x| {
                let x2 = x * x;
                (s2 + x2, s4 + x2 * x2)
            });
            complexity_numerator += row_ss * row_ss - row_fourth;
            complexity_denominator += row_ss * row_ss;
            for (j, value) in row.iter().enumerate() {
                factor_ss[j] += value * value;
            }
        }
        let row_complexity = if complexity_denominator > 0.0 {
            complexity_numerator / complexity_denominator
        } else {
            1.0
        };
        let minimum = factor_ss.iter().copied().fold(f64::INFINITY, f64::min);
        let maximum = factor_ss.iter().copied().fold(0.0_f64, f64::max);
        let factor_balance = if maximum > 0.0 {
            minimum / maximum
        } else {
            0.0
        };
        (row_complexity, factor_balance)
    }

    let fixtures = vec![
        vec![0.5, 0.2, -0.3, 0.8, -0.1, 0.9, 0.05, -0.05],
        vec![0.0, -0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5],
        vec![1e-310, -1e-310, 2e-310, 0.0, 1e-310, -1e-310, 2e-310, 0.0],
        vec![1e150, -1e150, 1e150, 1e150, 1e150, 1e150, -1e150, -1e150],
        vec![1e10, 1e-10, -1e10, -1e-10, 0.5, -0.5, 0.0, 0.0],
    ];

    for fixture in fixtures {
        for factors in [2, 4] {
            let rows = fixture.len() / factors;
            for skip_general in [false, true] {
                if skip_general && factors < 2 { continue; }
                let (ref_comp, ref_bal) = reference_metrics(&fixture, rows, factors, skip_general);
                let (opt_comp, opt_bal) = optimized_metrics(&fixture, rows, factors, skip_general);

                if ref_comp.is_finite() {
                    assert_eq!(ref_comp.to_bits(), opt_comp.to_bits());
                } else {
                    assert_eq!(ref_comp.is_nan(), opt_comp.is_nan());
                }

                if ref_bal.is_finite() {
                    assert_eq!(ref_bal.to_bits(), opt_bal.to_bits());
                } else {
                    assert_eq!(ref_bal.is_nan(), opt_bal.is_nan());
                }
            }
        }
    }
}

use mlsirm_core::binary64_mean::{
    correctly_rounded_finite_mean, Binary64MeanError, BINARY64_MEAN_CONTRACT,
};

fn value_bits(values: &[f64]) -> (u64, bool) {
    let result = correctly_rounded_finite_mean(values).expect("finite mean must be admitted");
    (result.value.to_bits(), result.exact_zero)
}

#[test]
fn publishes_versioned_domain_neutral_contract() {
    assert_eq!(BINARY64_MEAN_CONTRACT, "fast_mlsirm.binary64_mean@1.0.0");
}

#[test]
fn tepp_half_ulp_tail_changes_the_final_rounding() {
    let dominant = f64::from_bits(0x4698_0000_0000_0000);
    assert_eq!(
        value_bits(&[dominant, -(2_f64).powi(53), -1.0]),
        (0x467f_ffff_ffff_ffff, false)
    );
    assert_eq!(
        value_bits(&[-dominant, (2_f64).powi(53), 1.0]),
        (0xc67f_ffff_ffff_ffff, false)
    );
}

#[test]
fn preserves_low_order_mass_in_ordinary_mixed_sign_input() {
    assert_eq!(
        value_bits(&[1.0e16, -1.0, -1.0]),
        (0x4327_af4c_4a80_aaa9, false)
    );
}

#[test]
fn distinguishes_exact_zero_from_signed_nonzero_underflow() {
    let q = f64::from_bits(1);
    assert_eq!(value_bits(&[f64::MAX, -f64::MAX]), (0, true));
    assert_eq!(value_bits(&[-0.0, 0.0]), (0, true));
    assert_eq!(value_bits(&[q, 0.0]), (0, false));
    assert_eq!(value_bits(&[-q, 0.0]), (1_u64 << 63, false));
    assert_eq!(value_bits(&[f64::MAX, q, -f64::MAX]), (0, false));
}

#[test]
fn subnormal_rounding_covers_below_half_ties_and_residue() {
    let q = f64::from_bits(1);
    assert_eq!(value_bits(&[q, 0.0, 0.0]), (0, false));
    assert_eq!(value_bits(&[q, 0.0]), (0, false));
    assert_eq!(value_bits(&[q, 2.0 * q]), (2, false));
    assert_eq!(value_bits(&[q, q, 0.0]), (1, false));
    assert_eq!(value_bits(&[q, q]), (1, false));
    assert_eq!(
        value_bits(&[f64::MAX, 2.0 * q, 2.0 * q, -f64::MAX]),
        (1, false)
    );
}

#[test]
fn subnormal_rounding_can_carry_into_the_minimum_normal() {
    let max_subnormal = f64::from_bits((1_u64 << 52) - 1);
    let min_normal = f64::from_bits(1_u64 << 52);
    assert_eq!(
        value_bits(&[max_subnormal, min_normal]),
        (min_normal.to_bits(), false)
    );
}

#[test]
fn same_sign_near_maximum_mean_does_not_overflow() {
    assert_eq!(value_bits(&[f64::MAX, f64::MAX]), (f64::MAX.to_bits(), false));
}

#[test]
fn final_division_uses_round_to_nearest_ties_to_even() {
    let one = 1.0_f64;
    let next = f64::from_bits(one.to_bits() + 1);
    let next_next = f64::from_bits(one.to_bits() + 2);

    assert_eq!(value_bits(&[one, next]), (one.to_bits(), false));
    assert_eq!(
        value_bits(&[next, next_next]),
        (next_next.to_bits(), false)
    );
    assert_eq!(value_bits(&[one, one, next]), (one.to_bits(), false));
    assert_eq!(value_bits(&[one, next, next]), (next.to_bits(), false));
}

#[test]
fn rounding_covers_the_minimum_normal_binade_and_binade_carry() {
    let min_normal = f64::from_bits(1_u64 << 52);
    let min_next = f64::from_bits((1_u64 << 52) + 1);
    let min_next_next = f64::from_bits((1_u64 << 52) + 2);
    assert_eq!(
        value_bits(&[min_normal, min_next]),
        (min_normal.to_bits(), false)
    );
    assert_eq!(
        value_bits(&[min_next, min_next_next]),
        (min_next_next.to_bits(), false)
    );

    let two = 2.0_f64;
    let below_two = f64::from_bits(two.to_bits() - 1);
    assert_eq!(value_bits(&[below_two, two]), (two.to_bits(), false));
}

#[test]
fn permutation_does_not_change_the_result() {
    let dominant = f64::from_bits(0x4698_0000_0000_0000);
    let expected = value_bits(&[dominant, -(2_f64).powi(53), -1.0]);
    assert_eq!(
        value_bits(&[-1.0, dominant, -(2_f64).powi(53)]),
        expected
    );
    assert_eq!(
        value_bits(&[-(2_f64).powi(53), -1.0, dominant]),
        expected
    );
}

#[test]
fn refuses_empty_and_nonfinite_evidence() {
    assert_eq!(
        correctly_rounded_finite_mean(&[]),
        Err(Binary64MeanError::EmptyInput)
    );
    assert_eq!(
        correctly_rounded_finite_mean(&[1.0, f64::NAN]),
        Err(Binary64MeanError::NonFiniteValue { index: 1 })
    );
    assert_eq!(
        correctly_rounded_finite_mean(&[f64::INFINITY]),
        Err(Binary64MeanError::NonFiniteValue { index: 0 })
    );
}

//! Correctly-rounded arithmetic mean for finite IEEE-754 binary64 evidence.
//!
//! A finite binary64 value is an exact signed integer multiple of `2^-1074`.
//! This module accumulates those integers without floating-point rounding and
//! performs the original-count division before the single final
//! round-to-nearest, ties-to-even projection back to `f64`.
//!
//! The representation is deliberately domain-neutral. Temporal/event semantics,
//! psychometric estimands, missingness policy, and scientific acceptance remain
//! caller responsibilities.
//!
//! # Capacity invariant
//!
//! The largest finite binary64 value is `(2^53 - 1) * 2^971`, or
//! `(2^53 - 1) * 2^2045` units of `2^-1074`, which needs 2,098 magnitude bits.
//! For every slice cardinality representable by a supported `usize <= 64` bits,
//! the exact same-sign sum is strictly smaller than `2^2162`. Thirty-four
//! `u64` limbs provide 2,176 magnitude bits, so accumulation cannot overflow the
//! fixed representation without imposing a model-specific sample ceiling.

use core::cmp::Ordering;

/// Versioned public numerical contract for correctly-rounded finite means.
pub const BINARY64_MEAN_CONTRACT: &str = "fast_mlsirm.binary64_mean@1.0.0";

/// Number of 64-bit limbs needed by the proved 2,176-bit magnitude bound.
const LIMBS: usize = 34;
/// Binary64 fraction-field mask.
const FRACTION_MASK: u64 = (1_u64 << 52) - 1;
/// Implicit leading significand bit for normal binary64 values.
const HIDDEN_BIT: u64 = 1_u64 << 52;
/// Binary64 sign bit.
const SIGN_BIT: u64 = 1_u64 << 63;

const _: () = assert!(usize::BITS <= 64);

/// Result of a correctly-rounded finite binary64 mean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrectlyRoundedMean {
    /// One final IEEE-754 round-to-nearest, ties-to-even projection.
    pub value: f64,
    /// Whether the exact mathematical sum of all represented inputs is zero.
    ///
    /// This remains `false` when a nonzero exact mean is too small to represent
    /// and therefore rounds to signed zero.
    pub exact_zero: bool,
}

/// Admission failures for [`correctly_rounded_finite_mean`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binary64MeanError {
    /// A mean has no defined original-count denominator for an empty slice.
    EmptyInput,
    /// A caller supplied NaN or infinity; non-finite policy belongs upstream.
    NonFiniteValue { index: usize },
}

/// Fixed-width unsigned magnitude in exact `2^-1074` units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FixedMagnitude {
    /// Little-endian 64-bit limbs; capacity is fixed by the module proof.
    limbs: [u64; LIMBS],
}

impl FixedMagnitude {
    /// Construct the additive identity.
    const fn zero() -> Self {
        Self { limbs: [0; LIMBS] }
    }

    /// Return whether every limb is zero.
    fn is_zero(&self) -> bool {
        self.limbs.iter().all(|&limb| limb == 0)
    }

    /// Add one exact significand shifted by `shift` powers of two.
    ///
    /// The module capacity proof guarantees that carry propagation cannot run
    /// beyond the final limb for any admitted materializable slice.
    fn add_shifted_significand(&mut self, significand: u64, shift: usize) {
        let limb_index = shift / 64;
        let offset = shift % 64;
        let shifted = (significand as u128) << offset;

        let low_total = self.limbs[limb_index] as u128 + shifted as u64 as u128;
        self.limbs[limb_index] = low_total as u64;

        let high_total = self.limbs[limb_index + 1] as u128
            + (shifted >> 64)
            + (low_total >> 64);
        self.limbs[limb_index + 1] = high_total as u64;

        let mut carry = high_total >> 64;
        let mut index = limb_index + 2;
        while carry != 0 {
            let total = self.limbs[index] as u128 + carry;
            self.limbs[index] = total as u64;
            carry = total >> 64;
            index += 1;
        }
    }

    /// Compare exact unsigned magnitudes without conversion to floating point.
    fn cmp_magnitude(&self, other: &Self) -> Ordering {
        for index in (0..LIMBS).rev() {
            match self.limbs[index].cmp(&other.limbs[index]) {
                Ordering::Equal => {}
                ordering => return ordering,
            }
        }
        Ordering::Equal
    }

    /// Subtract `other` exactly in place when `self >= other`.
    fn subtract_assign(&mut self, other: &Self) {
        let mut borrow = 0_u128;
        for index in 0..LIMBS {
            let minuend = self.limbs[index] as u128;
            let subtrahend = other.limbs[index] as u128 + borrow;
            self.limbs[index] = minuend.wrapping_sub(subtrahend) as u64;
            borrow = u128::from(minuend < subtrahend);
        }
    }

    /// Divide the exact magnitude by a nonzero `u64`, returning quotient and remainder.
    fn div_rem_u64(&self, divisor: u64) -> (Self, u64) {
        let mut quotient = Self::zero();
        let mut remainder = 0_u128;
        let divisor = divisor as u128;
        for index in (0..LIMBS).rev() {
            let dividend = (remainder << 64) | self.limbs[index] as u128;
            quotient.limbs[index] = (dividend / divisor) as u64;
            remainder = dividend % divisor;
        }
        (quotient, remainder as u64)
    }

    /// Return the exact magnitude bit length, or zero for the additive identity.
    fn bit_len(&self) -> usize {
        self.limbs
            .iter()
            .rposition(|&limb| limb != 0)
            .map_or(0, |index| {
                index * 64 + (64 - self.limbs[index].leading_zeros() as usize)
            })
    }

    /// Read one bit from the exact magnitude.
    fn bit(&self, index: usize) -> bool {
        (self.limbs[index / 64] & (1_u64 << (index % 64))) != 0
    }

    /// Return whether any bit strictly below `index` is set.
    fn any_bits_below(&self, index: usize) -> bool {
        let whole_limbs = index / 64;
        if self.limbs[..whole_limbs].iter().any(|&limb| limb != 0) {
            return true;
        }
        let partial_bits = index % 64;
        partial_bits != 0
            && (self.limbs[whole_limbs] & ((1_u64 << partial_bits) - 1)) != 0
    }

    /// Return the low 64 bits after an exact right shift by `shift`.
    fn shifted_low_u64(&self, shift: usize) -> u64 {
        let limb_index = shift / 64;
        let offset = shift % 64;
        let pair = self.limbs[limb_index] as u128
            | ((self.limbs[limb_index + 1] as u128) << 64);
        (pair >> offset) as u64
    }
}

/// Decompose one finite binary64 value into sign, significand, and exact-unit shift.
fn split_finite_magnitude(value: f64) -> (bool, u64, usize) {
    let bits = value.to_bits();
    let negative = (bits & SIGN_BIT) != 0;
    let exponent = ((bits >> 52) & 0x7ff) as usize;
    let fraction = bits & FRACTION_MASK;
    if exponent == 0 {
        (negative, fraction, 0)
    } else {
        (negative, HIDDEN_BIT | fraction, exponent - 1)
    }
}

/// Decide ties-to-even rounding for an integer plus `remainder / divisor`.
fn round_scalar_fraction_up(integer: u64, remainder: u64, divisor: u64) -> bool {
    match (2_u128 * remainder as u128).cmp(&(divisor as u128)) {
        Ordering::Less => false,
        Ordering::Greater => true,
        Ordering::Equal => integer & 1 == 1,
    }
}

/// Round a quotient in the subnormal range, including carry into minimum normal.
fn round_subnormal_or_min_normal(quotient: &FixedMagnitude, remainder: u64, divisor: u64) -> u64 {
    let integer = quotient.limbs[0];
    let rounded = integer + u64::from(round_scalar_fraction_up(integer, remainder, divisor));
    if rounded == HIDDEN_BIT {
        1_u64 << 52
    } else {
        rounded
    }
}

/// Round a normal-range quotient to binary64 with exact sticky information.
fn round_normal(quotient: &FixedMagnitude, remainder: u64, divisor: u64) -> u64 {
    let bit_len = quotient.bit_len();
    let mut shift = bit_len - 53;
    let mut significand = quotient.shifted_low_u64(shift);

    let round_up = if shift == 0 {
        round_scalar_fraction_up(significand, remainder, divisor)
    } else {
        let half_bit = shift - 1;
        if !quotient.bit(half_bit) {
            false
        } else if quotient.any_bits_below(half_bit) || remainder != 0 {
            true
        } else {
            significand & 1 == 1
        }
    };

    if round_up {
        significand += 1;
        if significand == 1_u64 << 53 {
            significand = HIDDEN_BIT;
            shift += 1;
        }
    }

    let exponent = (shift + 1) as u64;
    let fraction = significand - HIDDEN_BIT;
    (exponent << 52) | fraction
}

/// Return the correctly-rounded arithmetic mean of finite binary64 values.
///
/// Every input is accumulated exactly in units of `2^-1074`. The exact signed
/// sum is divided by the original slice cardinality using integer quotient and
/// remainder, and only then projected once to binary64 with IEEE-754
/// round-to-nearest, ties-to-even. Consequently the result is independent of
/// input permutation and does not overflow merely because an intermediate sum
/// would exceed `f64::MAX` while the mean remains representable.
///
/// `exact_zero` distinguishes exact cancellation from a nonzero exact mean that
/// rounds to signed zero. Exact cancellation is canonicalized to `+0.0`; a
/// negative nonzero underflow preserves the IEEE-754 negative-zero sign.
///
/// # Errors
///
/// Empty input and any NaN or infinity are rejected. Missingness or other
/// domain-specific non-finite semantics must be resolved by the caller before
/// invoking this numerical primitive.
pub fn correctly_rounded_finite_mean(
    values: &[f64],
) -> Result<CorrectlyRoundedMean, Binary64MeanError> {
    if values.is_empty() {
        return Err(Binary64MeanError::EmptyInput);
    }

    let mut positive = FixedMagnitude::zero();
    let mut negative = FixedMagnitude::zero();

    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(Binary64MeanError::NonFiniteValue { index });
        }
        let (is_negative, significand, shift) = split_finite_magnitude(value);
        if significand == 0 {
            continue;
        }
        if is_negative {
            negative.add_shifted_significand(significand, shift);
        } else {
            positive.add_shifted_significand(significand, shift);
        }
    }

    let (is_negative, mut magnitude) = match positive.cmp_magnitude(&negative) {
        Ordering::Equal => {
            return Ok(CorrectlyRoundedMean {
                value: 0.0,
                exact_zero: true,
            })
        }
        Ordering::Greater => {
            positive.subtract_assign(&negative);
            (false, positive)
        }
        Ordering::Less => {
            negative.subtract_assign(&positive);
            (true, negative)
        }
    };

    debug_assert!(!magnitude.is_zero());
    let divisor = values.len() as u64;
    let (quotient, remainder) = magnitude.div_rem_u64(divisor);
    magnitude = quotient;

    let unsigned_bits = if magnitude.bit_len() <= 52 {
        round_subnormal_or_min_normal(&magnitude, remainder, divisor)
    } else {
        round_normal(&magnitude, remainder, divisor)
    };
    let signed_bits = if is_negative {
        unsigned_bits | SIGN_BIT
    } else {
        unsigned_bits
    };

    Ok(CorrectlyRoundedMean {
        value: f64::from_bits(signed_bits),
        exact_zero: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_magnitude_carry_and_ordering_are_exact() {
        let mut carried = FixedMagnitude::zero();
        carried.add_shifted_significand(u64::MAX, 0);
        carried.add_shifted_significand(1, 0);
        assert_eq!(carried.limbs[0], 0);
        assert_eq!(carried.limbs[1], 1);

        let mut larger = FixedMagnitude::zero();
        larger.add_shifted_significand(2, 64);
        assert_eq!(larger.cmp_magnitude(&carried), Ordering::Greater);
        assert_eq!(carried.cmp_magnitude(&larger), Ordering::Less);
        assert_eq!(carried.cmp_magnitude(&carried), Ordering::Equal);
    }

    #[test]
    fn fixed_magnitude_subtraction_borrows_across_limbs() {
        let mut lhs = FixedMagnitude::zero();
        lhs.limbs[1] = 1;
        let mut rhs = FixedMagnitude::zero();
        rhs.limbs[0] = 1;
        lhs.subtract_assign(&rhs);
        assert_eq!(lhs.limbs[0], u64::MAX);
        assert_eq!(lhs.limbs[1], 0);
    }

    #[test]
    fn low_bit_queries_cover_half_classification() {
        let mut value = FixedMagnitude::zero();
        value.limbs[0] = 0b1010;
        assert!(value.bit(3));
        assert!(!value.bit(2));
        assert!(value.any_bits_below(3));
        assert!(!value.any_bits_below(1));
        assert!(!FixedMagnitude::zero().any_bits_below(0));
    }
}

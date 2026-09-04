# Oblimax deterministic binary64 reference

## Claim boundary

`RotationCriterion::Oblimax` owns a deterministic CPU-binary64 arithmetic route for its criterion value and analytic gradient. For identical finite input bits, the implementation fixes the integer-power route to explicit multiplication and the logarithm route to package-owned IEEE-754 bit normalization plus a fixed 24-term Kahan-compensated `atanh` series. Before second/fourth moments are formed, one exact power-of-two multiplier derived from the maximum absolute loading conditions the represented range; the scale-free ratio `(sum4 / sum2) / sum2` is then formed before logarithm evaluation. This prevents scale-equivalent finite loading matrices from failing only because raw fourth moments overflow or underflow, while avoiding a decimal normalization step that would redefine significands. The repository therefore has a reviewable operator sequence and golden binary64 identity rather than delegating these operations to a platform transcendental implementation.

This is a criterion-local reproducibility contract. It does not imply that every rotation criterion, GPU kernel, foreign package, compiler configuration, or mathematical-library function is bitwise reproducible. Cross-target release evidence remains a separate requirement: a supported target/toolchain is not covered until its exact-head CI executes the golden-bit contract successfully. Formula agreement and factor-recovery validity also remain separate from bitwise identity.

## Why the route is explicit

Rust 1.98.1 documents `f64` as IEEE-754 binary64. Primitive addition, subtraction, multiplication, and division use IEEE round-to-nearest, ties-to-even semantics. By contrast, the Rust `f64::powi` and `f64::ln` documentation explicitly marks their precision as unspecified and potentially varying by platform, Rust version, and invocation. The Oblimax reference therefore does not use either operation in its owned reference span.

The logarithm decomposes a positive finite input into a binary64 significand and exponent, uses exact power-of-two normalization, bounds the transformed series variable around unity, and evaluates the same finite reduction in the same order. Oblimax first conditions the loading range with an exact binary exponent shift, reduces second and fourth moments on those conditioned values, forms their scale-free ratio, and then applies the logarithm once. This avoids two different scale hazards: cancellation from separately materializing `ln(sum4)` and `2 ln(sum2)`, and overflow/underflow from forming raw moments of an otherwise finite scale-equivalent loading matrix.

The range conditioner deliberately does **not** divide by an arbitrary floating-point maximum. A general division would introduce another rounded normalization route into ordinary fixtures. Instead, `binary_exponent_positive` reads the IEEE-754 exponent directly and `exact_power_of_two` constructs the shared multiplier from bits. The gradient is evaluated in the conditioned loading coordinates and multiplied by the same exact scale to map the derivative back to the caller's original loading coordinates. Zero loadings remain rejected as degenerate; non-finite caller loadings are rejected at the public criterion boundary; conditioned moments must remain finite and strictly positive.

## Evidence contract

The current-head acceptance suite requires all of the following without substituting one form of evidence for another:

- a source-level guard covering the deterministic logarithm helper and Oblimax body, rejecting `.powi(` and `.ln()` in that owner span;
- an exact golden-bit criterion value and all analytic-gradient components for a fixed finite loading matrix;
- repeated evaluation with bit-identical outputs;
- analytic-gradient versus central finite-difference coverage in the Rust criterion suite;
- ordinary common-scale invariance within the declared binary64 tolerance;
- bit-identical objective value under an exact `2^188` common scale, which exercises exponent cancellation without decimal scale-rounding ambiguity;
- successful, bit-identical objective evaluation under exact `2^300` and `2^-300` common scaling even though the preceding raw-moment route overflows `sum4` or underflows it to zero, plus inverse-scale gradient identity after mapping back to original loading coordinates;
- explicit exponent/power-of-two construction coverage at normal and subnormal binary64 boundaries;
- fail-closed handling for zero, non-finite input, invalid logarithm domain, and any conditioned moment that cannot be represented as a finite positive value;
- hosted supported-target execution before a cross-target release claim is made.

The extreme-scale fixtures are numerical-contract evidence, not a claim that such loading magnitudes are operationally typical. They isolate whether a mathematically scale-invariant criterion is spuriously narrowed by the binary64 range of an avoidable intermediate. The golden fixture is a regression identity for the declared arithmetic route, not an empirical psychometric validation result. Rotation validity and recovery must continue to be established through the existing criterion, optimizer, invariance, and realistic recovery evidence.

## Decision record

Problem: after the deterministic logarithm and scale-free log assembly were repaired, second and fourth moments were still formed from raw loadings. Exact `2^300` scaling can overflow the fourth moment and exact `2^-300` scaling can underflow it to zero although the input loadings are finite and the Oblimax objective is scale invariant.

Constraint: the fix must preserve the established ordinary-input golden identity, keep the Rust CPU-f64 route explicit, avoid unspecified transcendental behavior, and preserve the analytic gradient with respect to the original caller coordinates.

Alternatives considered: keep the raw-moment range and document a narrower input envelope; divide by an arbitrary finite maximum; or condition with one exact power-of-two derived from the represented exponent. The first option makes a scale-invariant criterion fail for avoidable implementation-range reasons. The second adds a rounded normalization operation and changed the arithmetic route unnecessarily. The selected exact power-of-two conditioner preserves binary significands, leaves the existing golden fixture unchanged, prevents the demonstrated raw-moment overflow/underflow class, and has a simple inverse-coordinate gradient transformation.

Risk: this does not prove that every possible finite loading vector yields representable conditioned second/fourth moments, nor does it substitute for supported-target execution or psychometric recovery. Those remain independent gates.

## Traceability

- Production owner: `crates/mlsirm-core/src/rotation/criteria.rs` (`RotationCriterion::Oblimax`).
- Executable contract: `crates/mlsirm-core/tests/rotation_moment_contract.rs` plus the criterion finite-difference suite.
- Extreme-scale source RED: `557789cd3e9e170c6d0eb46a8baebad49a00bb23`.
- Extreme-scale causal GREEN: `350c59fe99ccc53f0ce2c4234cb4e35c7686b5e8`.
- Repair issue: #1747.
- Canonical PR lane: #1736.
- Release note: `docs/changelog.d/1747-oblimax-deterministic-reference.md`.

## References

Institute of Electrical and Electronics Engineers. (2019). *IEEE standard for floating-point arithmetic* (IEEE Std 754-2019). https://standards.ieee.org/ieee/315/6210/

International Organization for Standardization, International Electrotechnical Commission, & Institute of Electrical and Electronics Engineers. (2020). *ISO/IEC/IEEE international standard—Floating-point arithmetic* (ISO/IEC/IEEE 60559:2020). https://standards.ieee.org/ieee/60559/10226/

Rust Project Developers. (2026, September 1). *Primitive type f64 (Rust 1.98.1)*. https://doc.rust-lang.org/stable/core/primitive.f64.html

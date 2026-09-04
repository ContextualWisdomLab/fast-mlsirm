# Oblimax deterministic binary64 reference

## Claim boundary

`RotationCriterion::Oblimax` owns a deterministic CPU-binary64 arithmetic route for its criterion value and analytic gradient. For identical finite input bits, the implementation fixes the integer-power route to explicit multiplication and the logarithm route to package-owned IEEE-754 bit normalization plus a fixed 24-term Kahan-compensated `atanh` series. The repository therefore has a reviewable operator sequence and golden binary64 identity rather than delegating these operations to a platform transcendental implementation.

This is a criterion-local reproducibility contract. It does not imply that every rotation criterion, GPU kernel, foreign package, compiler configuration, or mathematical-library function is bitwise reproducible. Cross-target release evidence remains a separate requirement: a supported target/toolchain is not covered until its exact-head CI executes the golden-bit contract successfully. Formula agreement and factor-recovery validity also remain separate from bitwise identity.

## Why the route is explicit

Rust 1.98.1 documents `f64` as IEEE-754 binary64. Primitive addition, subtraction, multiplication, and division use IEEE round-to-nearest, ties-to-even semantics. By contrast, the Rust `f64::powi` and `f64::ln` documentation explicitly marks their precision as unspecified and potentially varying by platform, Rust version, and invocation. The Oblimax reference therefore does not use either operation.

The logarithm decomposes the positive finite input into a binary64 significand and exponent, uses exact power-of-two normalization, bounds the transformed series variable around unity, and evaluates the same finite reduction in the same order. The helper rejects zero, negative, NaN, and infinite inputs. Oblimax separately rejects zero or non-finite moments before returning a criterion result.

## Evidence contract

The current-head acceptance suite requires all of the following without substituting one form of evidence for another:

- a source-level guard that the Oblimax implementation contains neither `.powi(` nor `.ln()`;
- an exact golden-bit criterion value and all analytic-gradient components for a fixed finite loading matrix;
- repeated evaluation with bit-identical outputs;
- analytic-gradient versus central finite-difference coverage in the Rust criterion suite;
- common-scale invariance within the declared binary64 tolerance;
- fail-closed handling for zero, non-finite input, and invalid logarithm domain;
- hosted supported-target execution before a cross-target release claim is made.

The golden fixture is a regression identity for the declared arithmetic route, not an empirical psychometric validation result. Rotation validity and recovery must continue to be established through the existing criterion, optimizer, invariance, and realistic recovery evidence.

## Traceability

- Production owner: `crates/mlsirm-core/src/rotation/criteria.rs` (`RotationCriterion::Oblimax`).
- Executable contract: `crates/mlsirm-core/tests/rotation_moment_contract.rs` plus the criterion finite-difference suite.
- Repair issue: #1747.
- Canonical PR lane: #1736.
- Release note: `docs/changelog.d/1747-oblimax-deterministic-reference.md`.

## References

Institute of Electrical and Electronics Engineers. (2019). *IEEE standard for floating-point arithmetic* (IEEE Std 754-2019). https://standards.ieee.org/ieee/315/6210/

International Organization for Standardization, International Electrotechnical Commission, & Institute of Electrical and Electronics Engineers. (2020). *ISO/IEC/IEEE international standard—Floating-point arithmetic* (ISO/IEC/IEEE 60559:2020). https://standards.ieee.org/ieee/60559/10226/

Rust Project Developers. (2026, September 1). *Primitive type f64 (Rust 1.98.1)*. https://doc.rust-lang.org/stable/core/primitive.f64.html

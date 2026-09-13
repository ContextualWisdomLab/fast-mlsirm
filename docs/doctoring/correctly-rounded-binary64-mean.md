# Correctly-rounded finite binary64 mean

## Claim

`mlsirm_core::binary64_mean::correctly_rounded_finite_mean` is a domain-neutral numerical primitive for non-empty finite `f64` slices. It preserves every represented addend exactly, divides the exact signed sum by the original slice cardinality, and performs one final IEEE-754 round-to-nearest, ties-to-even conversion. It also reports whether the exact represented sum is zero, so callers can distinguish exact cancellation from a nonzero mean that rounds to zero.

This claim is numerical only. It does not define missingness, temporal aggregation, psychometric estimands, sampling designs, Validation Evidence admission, or scientific acceptance. The governing implementation decision and rejected alternatives are recorded in Proposed [ADR-0029](../adr/0029-correctly-rounded-binary64-mean.md); the ADR must remain Proposed until protected integration and immutable-release evidence satisfy its acceptance conditions.

## Representation and capacity proof

Every finite binary64 value is an exact integer multiple of `q = 2^-1074`.

For a normal value with exponent-field integer `E`, the exact integer coefficient is

`I = (2^52 + fraction) * 2^(E - 1)`.

For a subnormal value it is simply the fraction field. The largest finite coefficient is therefore

`I_max = (2^53 - 1) * 2^2045`,

which needs 2,098 magnitude bits. On supported Rust targets `usize::BITS <= 64`; a materializable slice has at most `2^64 - 1` elements. Hence either same-sign accumulator is bounded by

`n * I_max < 2^64 * 2^2098 = 2^2162`.

The implementation allocates 34 `u64` limbs, or 2,176 magnitude bits, independently for positive and negative inputs. Positive and negative exact totals are compared and subtracted once after ingestion. The fixed width is therefore a proved representation bound rather than a product-specific sample ceiling.

## Final rounding

Let `S` be the exact signed coefficient sum, `M = |S|`, and `n` the original input count. Long division by the scalar `n` gives

`M = nQ + r`, with `0 <= r < n`.

No rounded floating-point sum is formed. For subnormal results, `Q` and `r/n` directly determine the nearest integer multiple of `q`. For normal results, the top 53 significand bits of `Q` are retained and the discarded integer bits plus `r/n` are classified as below half, above half, or exactly half. Exact half cases use the retained significand's least significant bit for ties-to-even. A carry out of the 53-bit significand advances the exponent binade. Exact cancellation is canonicalized to `+0.0`; nonzero negative underflow retains the negative-zero sign.

## Test trace

The public integration contract is `crates/mlsirm-core/tests/binary64_mean_contract.rs`. It includes:

- the TEPP consumer counterexample `0x1.8p+106, -2^53, -1`, whose exact mean rounds to `0x467fffffffffffff`, plus the mirrored negative case;
- `[1e16, -1, -1]` low-order mass;
- the required ordinary tiny-residue cancellation fixture `[f64::MAX, 1e-16, -f64::MAX]`, whose exact represented mean retains the `1e-16` residue before division;
- exact cancellation versus positive and negative nonzero underflow;
- minimum-subnormal residue after `f64::MAX` cancellation;
- subnormal ties-to-even and carry into the minimum normal;
- normal ties-to-even, less-than-half and greater-than-half cases, and carry across a binade;
- same-sign `f64::MAX` inputs, whose mean remains finite although a naive intermediate sum overflows;
- permutation invariance;
- empty and positive/negative non-finite fail-closed admission;
- 10,000 deterministic subnormal-domain vectors checked against an independent test-only exact integer/rational oracle that sums signed `2^-1074` coefficients in `i128`, divides by the exact slice cardinality, and applies ties-to-even without calling the production accumulator.

Private fixed-magnitude tests cover carry propagation, ordering, cross-limb borrow, and rounding-bit classification. The exact-oracle property test deliberately stays in a bounded subnormal domain so that its independent `i128` representation is itself trivial to audit; it complements rather than reimplements the production 2,176-bit accumulator. Hosted exact-head Rust, rustdoc/Clippy, statement/branch coverage, security, package/fuzz and review evidence remain release gates; this document does not substitute for those gates.

## Ownership and consumer rule

The reusable arithmetic belongs to `fast-mlsirm`. TEPP may retain its consumer RED while this branch is mutable, but it must not copy this implementation or pin a PR head. Consumption begins only from an immutable released/versioned fast-mlsirm contract with SBOM/provenance/reproducibility and rollback evidence.

## Research and standards trace

ISO/IEC 60559:2020 is the current published ISO floating-point arithmetic standard as checked on 2026-09-11. IEEE has an active P754 revision project superseding IEEE 754-2019; therefore the implementation records the published 2019/2020 arithmetic contract and does not present the in-progress revision as a released standard.

The error-free-transformation literature is supporting design context rather than a proof that a floating partials implementation would satisfy this API. Ogita, Rump, and Oishi (2005) develop accurate sum/dot-product methods; Rump, Ogita, and Oishi (2008a) analyze faithful summation and underflow; Rump, Ogita, and Oishi (2008b) give rounding-to-nearest summation algorithms. Proposed ADR-0029 records why those techniques, compensation/coalescing, arbitrary-precision production dependencies, and pre-scaling were considered but not selected for version 1.0.0. The selected implementation instead uses exact integer accumulation because the complete binary64 exponent range and 64-bit slice-cardinality bound give a small fixed representation with a direct original-count rounding proof.

### References

International Organization for Standardization. (2020). *Information technology—Microprocessor systems—Floating-point arithmetic* (ISO/IEC 60559:2020). https://www.iso.org/standard/80985.html

Institute of Electrical and Electronics Engineers. (2019). *IEEE standard for floating-point arithmetic* (IEEE Std 754-2019). https://standards.ieee.org/ieee/315/6210/

Ogita, T., Rump, S. M., & Oishi, S. (2005). Accurate sum and dot product. *SIAM Journal on Scientific Computing, 26*(6), 1955–1988. https://doi.org/10.1137/030601818

Rump, S. M., Ogita, T., & Oishi, S. (2008a). Accurate floating-point summation part I: Faithful rounding. *SIAM Journal on Scientific Computing, 31*(1), 189–224. https://doi.org/10.1137/050645671

Rump, S. M., Ogita, T., & Oishi, S. (2008b). Accurate floating-point summation part II: Sign, K-fold faithful and rounding to nearest. *SIAM Journal on Scientific Computing, 31*(2), 1269–1302. https://doi.org/10.1137/07068816X

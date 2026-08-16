# Paired rating-range scalar control boundary

## Problem

`paired_rating_range_evidence()` accepts a caller-controlled `category_count` that determines the permitted ordinal-label domain before data are marshalled into the Rust core. The protected-main implementation admitted any `int`/`numpy.integer` subclass and then called `int(category_count)`. Python's numeric conversion protocol can execute special methods supplied by a subclass, so broad subclass admission makes conversion behavior caller-programmable instead of a sealed scalar-control boundary.

This is a validation and marshalling defect, not a psychometric-formula defect. The Rust implementation remains the sole owner of descriptive rating-range arithmetic.

## Contract

The public wrapper now accepts only these exact scalar identities:

- built-in `int`;
- NumPy `int8`, `int16`, `int32`, `int64`;
- NumPy `uint8`, `uint16`, `uint32`, `uint64`.

Python and NumPy integer subclasses are rejected before conversion, label validation, native-core discovery, or Rust dispatch. Accepted NumPy scalars are normalized once to a built-in `int`, then the established inclusive domain `2..1000` is enforced. Boolean values remain invalid.

The change intentionally does **not** alter paired-rating labels, Rust arithmetic, result fields, category semantics, scoring thresholds, or downstream policy.

## Test evidence

The fail-first regression at commit `4adfc0a41d1136242a73da380b3c2e345057d729` introduces hostile Python and NumPy integer subclasses whose `__int__` and `__repr__` methods raise if executed. The GREEN implementation must reject those values with the package-owned `ValueError`, leave the callback ledger empty, and avoid native-core discovery. Separate parameterized cases preserve exact built-in NumPy integer scalar support and require a built-in `int` at Rust dispatch.

Hosted current-head tests remain authoritative for integration; predecessor-head test, review, or check evidence is historical only.

## Security and reliability rationale

The boundary follows the repository's fail-closed pattern: validate the identity and domain of a control before invoking extensible conversion behavior or privileged/native execution. This is consistent with NIST SSDF's emphasis on preventing vulnerabilities through secure design, implementation, and verification. NIST SP 800-218 version 1.1 remains the current final SSDF publication; version 1.2 is available as an Initial Public Draft and is tracked as prospective guidance rather than represented as final.

## References

National Institute of Standards and Technology. (2022). *Secure Software Development Framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218). U.S. Department of Commerce. https://doi.org/10.6028/NIST.SP.800-218

National Institute of Standards and Technology. (2025). *Secure Software Development Framework (SSDF) version 1.2: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218 Rev. 1, Initial Public Draft). U.S. Department of Commerce. https://csrc.nist.gov/pubs/sp/800/218/r1/ipd

NumPy Developers. (2026). *Data types*. NumPy documentation. Retrieved August 16, 2026, from https://numpy.org/doc/stable/user/basics.types.html

Python Software Foundation. (2026). *Built-in functions: `int`*. Python 3.14.6 documentation. Retrieved August 16, 2026, from https://docs.python.org/3.14/library/functions.html#int

# Mixed item-model input validation

## Decision

`fit_mixed_items(..., item_models=...)` treats response-family names as a finite semantic control, not as values that may be coerced from arbitrary caller objects. A built-in `str` may broadcast to every item. Other inputs are consumed as an iterable with at most `n_items + 1` requests, which is sufficient to distinguish an exact-length item-model vector from an overlong stream without eagerly materializing an unbounded iterable.

Every model token must be an exact built-in `str` before normalization through the package-owned alias table. Ordinary iterator construction or iteration failures are normalized to stable package-owned `ValueError` messages. `KeyboardInterrupt`, `SystemExit`, and `GeneratorExit` remain process-control signals because the validator catches `Exception`, not `BaseException`. Unsupported model text and caller exception text are not copied into public validation errors.

This is a reliability, privacy, and error-surface hardening decision. It does not change any mixed-format likelihood, EM update, missingness contract, category handling, convergence rule, thread behavior, parameter interpretation, or numerical result. Those computations remain Rust-owned.

## Security and reliability rationale

Python's data model specifies that `str(object)` invokes the object's `__str__()` method, so converting an arbitrary semantic-control object is executable caller-defined behavior rather than passive validation. Likewise, eagerly applying `list(...)` to a caller-controlled iterable has no intrinsic item-count bound and can keep consuming an infinite or unexpectedly large stream.

The package therefore validates the finite control domain before normalization and uses one bounded look-ahead entry to detect overlength. Public errors identify the package-owned field and contract while avoiding rejected caller content. CWE-209 is used only as an engineering taxonomy for minimizing sensitive information in error messages; it is not certification, severity evidence, or proof of exploitability.

## Verification contract

Acceptance requires all of the following on the same product implementation:

- overlong or infinite model iterables are consumed at most `n_items + 1` entries before rejection;
- short iterables fail deterministically without additional unbounded consumption;
- ordinary iterator-construction and iterator-execution failures become non-reflective package errors;
- `KeyboardInterrupt`, `SystemExit`, and `GeneratorExit` propagate unchanged;
- hostile non-string model objects are rejected without invoking `__str__()` or `__repr__()`;
- unsupported built-in strings are rejected without echoing the rejected value;
- built-in string broadcast plus accepted list, tuple, and generator aliases preserve canonical dispatch exactly; and
- invalid semantic controls fail before the compiled mixed-format numerical entrypoint is invoked.

The pre-documentation product head `bebbc3b244da6b4222de02d6e06bc1978b1358bc` passed the complete Python suite, Rust workspace/PyO3 tests, package/build/reinstall/release-acceptance checks, explicit GPU no-skip evidence, fuzz, Security Scan, and SAST. Final acceptance must be recreated after documentation/changelog synchronization; predecessor success is not final-head evidence.

## References — APA 7

MITRE. (2026). *CWE-209: Generation of Error Message Containing Sensitive Information (Version 4.20).* https://cwe.mitre.org/data/definitions/209.html

Python Software Foundation. (2026). *Built-in exceptions — Python 3.14.6 documentation.* https://docs.python.org/3.14/library/exceptions.html

Python Software Foundation. (2026). *Data model — Python 3.14.6 documentation.* https://docs.python.org/3.14/reference/datamodel.html

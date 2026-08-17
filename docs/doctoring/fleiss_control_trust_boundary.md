# Fleiss kappa control trust boundary

## Scope

Issue #961 is a Python marshalling-boundary correction for the public `fleiss_kappa` adapter. It does **not** change Fleiss/Conger agreement arithmetic, chance-agreement definitions, category-wise kappas, z statistics, p values, missing-row semantics, or numerical ownership. Those result-affecting computations remain in the Rust `mlsirm_core::agreement::fleiss_kappa` implementation.

The bounded correction establishes result-affecting controls before caller data is materialized and before the compiled core is discovered:

- explicit `k` admits only exact built-in `int` and the package-trusted concrete NumPy integer scalar identities;
- `bool`, integer subclasses, arbitrary `__int__`/`__index__` providers, and values outside `2..=10000` are rejected without executing caller callbacks;
- `exact` admits only exact built-in `bool` and concrete `np.bool_` and rejects arbitrary truthiness providers without executing `__bool__`;
- inferred `k` is capped at 10,000 before native dispatch, matching the dense category-allocation guard;
- accepted controls are normalized to exact built-in `int`/`bool` values before PyO3 marshalling.

## Evidence contract

`tests/test_validation_fleiss_control_safety.py` provides public-boundary evidence for hostile integer subclasses, index-only providers, truthiness providers, pre-core rejection, genuine Python/NumPy scalar compatibility, and inferred-category overflow. A fake native core records only accepted normalized controls, so the regression distinguishes control admission from agreement arithmetic.

The expected security invariant is: **a rejected semantic control performs zero caller-defined conversion/truthiness callbacks and reaches neither ratings materialization nor native Fleiss dispatch.** Valid controls preserve the existing Rust-owned results.

## Ownership and interoperability

Python owns trust-boundary validation, array marshalling, and result-object construction. Rust owns the Fleiss/Conger statistics. This separation avoids moving psychometric formulas into Python while making the package boundary deterministic for Python and NumPy scalar callers.

## References

Conger, A. J. (1980). Integration and generalization of kappas for multiple raters. *Psychological Bulletin, 88*(2), 322–328.

Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters. *Psychological Bulletin, 76*(5), 378–382.

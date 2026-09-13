# Multilevel contextual-effect finiteness and sparse marshalling

## Decision

The public multiple-membership contextual-effect predictor treats only context effects referenced by the supplied sparse design as numerical inputs. Referenced non-finite values and non-finite weighted outputs fail closed in the Rust numerical boundary. Unreferenced effect-table capacity is not scanned because it cannot contribute to the supplied design. Python is limited to package-contract verification and deterministic marshalling: each required mapping value is read once without a separate membership probe, and caller-controlled lookup or conversion exceptions are normalized before native dispatch.

This boundary does **not** redefine the multiple-membership model or its weighting semantics. It protects the implementation contract around the existing weighted contextual contribution while preserving sparse work proportional to referenced memberships.

## Evidence and rationale

Browne, Goldstein, and Rasbash (2001) formalize multiple-membership/multiple-classification models in which an observation can belong to more than one member of a classification. The predictor implemented here preserves that structural premise: membership weights and referenced contextual effects determine the contribution, so unused effect-table entries are not part of the realized sparse design.

IEEE 754-2019 is the current IEEE standard for floating-point arithmetic and explicitly defines special floating-point values and behavior including infinities and NaNs. Those values are legitimate floating-point representations, but they are not acceptable realized contextual-effect parameters or public predictor outputs for this package contract. The implementation therefore validates referenced parameter finiteness before numerical work and validates output finiteness before returning it.

The Python mapping boundary is an interoperability/reliability concern rather than psychometric arithmetic. A caller-supplied `Mapping` may implement `__contains__`, `__getitem__`, or numeric coercion with arbitrary code. Performing a membership probe and then a second lookup creates unnecessary duplicate callback execution and a time-of-check/time-of-use surface. The marshaller now reads each required key once, preserves the governed missing-key contract, and replaces other callback failures with package-owned non-reflective errors. Numerical finiteness and weighted summation remain Rust-owned.

## Verification contract

- referenced `NaN`, `+Inf`, and `-Inf` effects are rejected before weighted output is returned;
- finite inputs whose weighted sum overflows are rejected;
- non-finite values in unreferenced effect-table slots do not change output or expand sparse validation work;
- Python marshalling does not invoke caller-defined membership probes for required keys;
- hostile lookup and numeric-coercion callbacks cannot reflect caller-controlled exception text through the public package error;
- ordinary finite results remain unchanged across the public Python → PyO3 → Rust path.

## References

Browne, W. J., Goldstein, H., & Rasbash, J. (2001). Multiple membership multiple classification (MMMC) models. *Statistical Modelling, 1*(2), 103–124. https://doi.org/10.1177/1471082X0100100202

IEEE Standards Association. (2019). *IEEE standard for floating-point arithmetic* (IEEE Std 754-2019). IEEE. https://doi.org/10.1109/IEEESTD.2019.8766229

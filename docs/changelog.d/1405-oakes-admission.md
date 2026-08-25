# Callback-safe and bounded Oakes uncertainty evidence admission

## Fixed

- Validate the Oakes finite-difference step `h` as a finite positive, losslessly representable Rust `f64` control before any caller response, factor, or mask evidence is inspected.
- Seal Oakes response and item-to-dimension evidence before NumPy materialization. Exact real-numeric NumPy arrays and inert built-in list/tuple evidence with concrete Python/NumPy numeric scalars remain supported; arbitrary array/numeric providers, subclasses, object/text storage, and concrete complex evidence fail closed with stable field-specific diagnostics.
- Seal optional observation masks before Boolean coercion so caller truth-value protocols cannot alter which response cells enter uncertainty estimation. Built-in mask cells use the same NumPy typecode-derived exact scalar universe as the response/factor admission path, preserving concrete integer aliases such as `longlong`/`ulonglong` without reopening subclass or protocol callbacks.
- Bound Oakes response evidence to 20,000,000 logical cells and built-in container traversal to 40,000,000 structural nodes before dense `float64` marshalling, while preserving `NaN`/`-1` response missingness and signed-64 factor narrowing contracts.
- Preserve Rust ownership of the Oakes information identity, finite-difference cross term, covariance, and standard-error arithmetic; Python changes are validation, bounded marshalling, and regression evidence only.

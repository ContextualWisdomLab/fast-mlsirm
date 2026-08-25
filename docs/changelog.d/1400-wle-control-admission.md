# Seal Warm WLE controls and observation masks before caller evidence

## Fixed

Dichotomous and polytomous Warm WLE entry points now validate and normalize `theta_bound`, `tol`, category-count, and model-family controls before caller array materialization or compiled-core discovery. Callback-bearing scalar providers fail closed without executing their conversion protocols, accepted Python/NumPy numeric controls must preserve their exact value through the Rust `f64`/native-integer boundary, and supported NumPy string model identities are normalized to package-owned strings.

Explicit `observed` masks now use a callback-free Boolean admission boundary as well. Exact Boolean NumPy arrays and exact built-in list/tuple masks containing concrete Python/NumPy Boolean values are normalized to contiguous package-owned Boolean arrays; generic array providers, container/array subclasses, object/text/complex storage, non-Boolean cells, and shape mismatches fail closed before truth coercion or Rust discovery. Warm correction, information, root-search, standard-error, GRM, and GPCM numerical arithmetic remain Rust-owned.
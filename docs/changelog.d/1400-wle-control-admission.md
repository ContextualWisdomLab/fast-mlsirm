# Seal Warm WLE semantic controls before caller evidence

## Fixed

Dichotomous and polytomous Warm WLE entry points now validate and normalize `theta_bound`, `tol`, category-count, and model-family controls before caller array materialization or compiled-core discovery. Callback-bearing scalar providers fail closed without executing their conversion protocols, accepted Python/NumPy numeric controls must preserve their exact value through the Rust `f64`/native-integer boundary, and supported NumPy string model identities are normalized to package-owned strings. Warm correction, information, root-search, standard-error, GRM, and GPCM numerical arithmetic remain Rust-owned.
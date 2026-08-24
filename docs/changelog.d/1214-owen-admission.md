# Owen CAT evidence admission

## Fixed

- Establish Owen posterior/CAT scalar, Boolean, item-array, and binary-response trust boundaries before compiled-core discovery or caller-controlled coercion. Caller-defined scalar/truth callbacks, complex/text/object item or response storage, and arbitrary array providers now fail closed while supported NumPy scalar/array evidence is normalized to inert built-in/contiguous representations. Owen posterior moments, b-matching, variance stopping, and all result-affecting psychometric arithmetic remain Rust-owned.

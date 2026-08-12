# Infit/outfit and ordinary M2 require Rust ownership

## Fixed

- Public `infit_outfit()` fails closed when the compiled Rust core or required `infit_outfit_stat` entrypoint is missing, instead of silently selecting pure-Python residual arithmetic.
- Ordinary public `m2()` fails closed when the compiled Rust core or required `m2_stat` entrypoint is missing, instead of silently selecting the private NumPy reference.
- Retargeted legacy S-X² / person-fit / infit-outfit NumPy-fallback coverage tests to the fail-closed ownership contract.

# Validate Rasch CML controls before data materialization

## Changed

- Validate `max_iter` and `tol` before caller-owned response or group arrays are materialized by the public Rasch CML and Andersen LR entry points.
- Reject complex-valued response matrices and Andersen group labels before `float64` coercion can discard imaginary components and silently admit altered data.
- Preserve exact built-in and supported NumPy scalar compatibility and existing validation errors while preventing invalid semantic controls from triggering response-array callbacks or unnecessary data work.
- Keep conditional likelihood, optimization, information, LR, p-value, and all other production psychometric arithmetic in the Rust core.

# Validate Rasch CML controls before data materialization

## Changed

- Validate `max_iter` and `tol` before caller-owned response or group arrays are materialized by the public Rasch CML and Andersen LR entry points.
- Reject complex-valued response matrices and Andersen group labels before `float64` coercion can discard imaginary components and silently admit altered data.
- Establish exact package-trusted response/group container and scalar identities before NumPy materialization so arbitrary `__array__` providers, ndarray/container/numeric subclasses, and object/text storage cannot execute caller conversion protocols while defining the scientific evidence analyzed by Rust.
- Preserve exact NumPy Boolean/integer/unsigned/real arrays, ordinary built-in response rows, finite non-negative integer group labels, and supported concrete NumPy scalar compatibility.
- Keep conditional likelihood, optimization, information, LR, p-value, and all other production psychometric arithmetic in the Rust core.

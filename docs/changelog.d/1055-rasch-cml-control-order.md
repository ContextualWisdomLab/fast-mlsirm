# Validate Rasch CML controls before data materialization

## Changed

- Validate `max_iter` and `tol` before caller-owned response or group arrays are materialized by the public Rasch CML and Andersen LR entry points.
- Reject complex-valued response matrices and Andersen group labels before `float64` coercion can discard imaginary components and silently admit altered data.
- Establish exact package-trusted response/group container and scalar identities before NumPy materialization so arbitrary `__array__` providers, ndarray/container/numeric subclasses, and object/text storage cannot execute caller conversion protocols while defining the scientific evidence analyzed by Rust.
- Preserve Andersen external group identities as exact package-owned integers before deterministic dense-ID construction, so distinct labels above the `float64` exact-integer boundary do not collapse and large unsigned labels do not wrap through signed narrowing.
- Preserve exact NumPy Boolean/integer/unsigned/real arrays, ordinary built-in response rows, finite non-negative integral group labels, and supported concrete NumPy scalar compatibility.
- Keep conditional likelihood, optimization, information, LR, p-value, and all other production psychometric arithmetic in the Rust core.

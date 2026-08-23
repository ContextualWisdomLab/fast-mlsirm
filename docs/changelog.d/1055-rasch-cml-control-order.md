# Validate Rasch CML controls before data materialization

## Changed

- Validate `max_iter` and `tol` before caller-owned response or group arrays are materialized by the public Rasch CML and Andersen LR entry points.
- Reject complex-valued response matrices and Andersen group labels before `float64` coercion can discard imaginary components and silently admit altered data.
- Establish exact package-trusted response/group container and scalar identities before NumPy materialization so arbitrary `__array__` providers, ndarray/container/numeric subclasses, and object/text storage cannot execute caller conversion protocols while defining the scientific evidence analyzed by Rust.
- Preserve Andersen external group identities as exact package-owned integers before deterministic dense-ID construction, so distinct labels above the `float64` exact-integer boundary do not collapse and large unsigned labels do not wrap through signed narrowing.
- Reject response evidence above 20,000,000 logical cells before NumPy stacking, `float64` materialization, or signed-`int64` allocation, including oversized exact broadcast matrices and exact NumPy row leaves nested inside trusted built-in response matrices.
- Preserve exact NumPy Boolean/integer/unsigned/real arrays, ordinary built-in response rows, finite non-negative integral group labels, and supported concrete NumPy scalar compatibility inside the explicit response resource envelope.
- Keep conditional likelihood, optimization, information, LR, p-value, and all other production psychometric arithmetic in the Rust core.

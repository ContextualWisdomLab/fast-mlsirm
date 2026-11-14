## 2024-05-24 - JSON.loads Float Overflow Vulnerability
**Vulnerability:** A malicious actor could inject infinity or non-finite values into the system by passing huge float strings like `1e999` into `json.loads` because the default python float parsing will parse them as `inf`.
**Learning:** Adding a `parse_constant` hook to `json.loads` is insufficient to prevent infinity values from entering if they are constructed from large exponents. `parse_float` needs to be overridden to check for finiteness.
**Prevention:** Always implement a `parse_float` override that runs `float(literal)` and verifies it using `math.isfinite(val)` when reading untrusted JSON data.

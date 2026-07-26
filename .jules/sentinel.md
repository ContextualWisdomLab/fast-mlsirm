## 2026-06-29 - [np.load Insecure Deserialization Risk & Assertion Optimization Removal]
**Vulnerability:**
1. `numpy.load()` was used without explicitly specifying `allow_pickle=False`. This could lead to insecure deserialization and arbitrary code execution if a malicious pickle file is loaded (especially critical depending on the environment's NumPy version).
2. `assert` was used for critical control flow (`assert best is not None`). Assertions are stripped out when Python is run with the `-O` optimization flag, potentially leading to undefined behavior and masking errors in production environments.

**Learning:**
Explicitly defining `allow_pickle=False` is a robust defense-in-depth practice. Relying on `assert` for necessary runtime checks is dangerous; standard exceptions like `RuntimeError` should be used instead.

**Prevention:**
- Always add `allow_pickle=False` to `np.load` unless explicitly required and verified.
- Replace critical `assert` statements with `if` condition checks that raise appropriate runtime exceptions.

## 2026-07-06 - [DoS via Unconstrained Array Dimension Allocation]
**Vulnerability:** In `fast_mlsirm/fit.py`, the number of dimensions `n_dims` was calculated using the maximum value provided in user input (`factor_id.max()`). A maliciously crafted large integer in `factor_id` causes `np.zeros((n_persons, n_dims))` to attempt allocating an impossibly large array (e.g. hundreds of GiB), crashing the application via Out-Of-Memory (OOM) and causing a Denial of Service (DoS).
**Learning:** Never trust user input to define unconstrained array dimensions, especially when derived from maximum values within the data.
**Prevention:** Add explicit boundary checks (e.g. `n_dims > n_items`) to ensure derived dimensions remain mathematically sound and computationally feasible before memory allocation.
## 2024-07-04 - [Defense in Depth] Validate URI Schemes in Link Generation
**Vulnerability:** A script (`scripts/build_pr_queue_governance.py`) used `escape()` to sanitize URLs placed directly in the `href` attribute of an `<a>` tag. However, `escape()` alone is insufficient to prevent XSS if the URL uses an unsafe protocol such as `javascript:` or `data:`.
**Learning:** This is a classic case where escaping HTML special characters provides a false sense of security for URI-based injection contexts. An attacker could potentially inject a malicious script by providing an unsafe protocol.
**Prevention:** Always validate URI schemes and restrict them to safe protocols (e.g., `http:`, `https:`) before using them in contexts like `href` or `src`. If an unsafe scheme is detected, the URL should be neutralized (e.g., replaced with `#`). I implemented a `_safe_url` helper function to enforce this.
## 2026-07-12 - [Bandit B324: Use of weak MD5 hash for security]
**Vulnerability:** MD5 hashing in `fast_mlsirm/report.py` triggered a high severity warning by Bandit, because by default it is assumed to be used for security purposes which is unsafe due to weak hashing.
**Learning:** For non-security purposes like generating unique dom ids, `hashlib.md5()` triggers a vulnerability warning unless `usedforsecurity=False` is passed. This allows bypassing FIPS compliance limitations as well as suppressing false positive warnings.
**Prevention:** Always add `usedforsecurity=False` parameter to `hashlib.md5` and other weak hashing functions unless they are genuinely used for secure cryptography (which they shouldn't be).
## 2026-07-20 - [Gradient Poisoning via NaN/Inf Hyperparameters Bypass]
**Vulnerability:** In `fast_mlsirm/config.py`, bounding checks on numeric hyperparameters (like `learning_rate <= 0` or `eps_distance <= 0`) failed to explicitly check for `NaN`. Because `NaN <= 0` evaluates to `False`, a malicious or malformed `NaN` value could bypass validation, poison gradients (e.g. producing `NaN` or `inf` during iterative calculations), and cause unhandled crashes later in the execution flow.
**Learning:** Python numerical boundary checks (like `<` or `<=`) will evaluate to `False` when comparing against `NaN`. This allows `NaN` to silently bypass constraints intended to ensure positive numerical values, breaking the assumption that the validated values are computationally safe.
**Prevention:** Always explicitly check for `NaN` and `inf` using `math.isfinite()` or `np.isfinite()` before or alongside enforcing numerical limits on parameters used in critical calculations.

## 2024-05-15 - [Assert Removal in Python]
**Vulnerability:** Use of `assert` in application logic (`assert best is not None` in `python/fast_mlsirm/fit.py`).
**Learning:** Python `assert` statements are stripped when the interpreter is run with optimization (`-O`). This can cause application logic to silently skip checks and lead to undefined behaviors or security issues.
**Prevention:** Replace `assert` with explicit `if` conditions that raise exceptions like `RuntimeError` or `ValueError` to ensure checks are enforced regardless of interpreter optimization flags.

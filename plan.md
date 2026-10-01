1. **Optimize boolean array counting in `python/fast_mlsirm/ata.py`**
   - Replace `int(np.sum(labels[eligible_now] == lbl))` with `int(np.count_nonzero(labels[eligible_now] == lbl))`.
   - `np.count_nonzero` is significantly faster and more memory-efficient than `np.sum` on boolean arrays because it avoids allocating an intermediate integer array.
   - Also add an inline comment explaining this optimization as per Bolt guidelines.
2. **Execute testing**
   - Run tests using `uv run pytest tests/` to confirm the change does not break existing functionality.
3. **Pre-commit**
   - Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.

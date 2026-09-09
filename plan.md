1. **Identify Bottleneck**: The loop in `fast_mlsirm/estimators/marginal.py` at line 1045 calculates `dist = np.sqrt(eps_distance + np.sum(diff * diff, axis=1))`. This involves an intermediate array allocation of `diff * diff` which can be large, slowing down the distance computation loop.
2. **Implement Vectorization**: Replace `np.sum(diff * diff, axis=1)` with `np.einsum("ij,ij->i", diff, diff)` to avoid intermediate array allocations and improve performance, which is documented in `.jules/bolt.md`.
3. **Verify Modification**: Execute format, lint, and test scripts to guarantee the code quality and logic remain correct. Measure the performance impact.
4. **Complete Pre-Commit Steps**: Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.
5. **Document Learnings**: (If novel) add entry in `.jules/bolt.md`.
6. **Submit PR**: Create a performance optimization PR with correct prefix and description.

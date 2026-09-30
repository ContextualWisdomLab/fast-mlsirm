We have 4 files that use `json.loads` for deserializing unconstrained JSON and missing a `parse_float` hook which could lead to DoS by causing overflow.

Files to modify:
1. `python/fast_mlsirm/cross_engine_conformance.py`
2. `python/fast_mlsirm/io.py`
3. `python/fast_mlsirm/llm_judge.py`
4. `python/fast_mlsirm/rubric/candidates.py`

In each file, we'll need to define a `_reject_float_nonfinite(value)` function (and import `math` where missing), and update the `json.loads` calls with `parse_float=_reject_float_nonfinite`.

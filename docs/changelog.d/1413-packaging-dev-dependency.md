# Declare packaging as an explicit dev dependency

## Changed

- Add `packaging>=23` to the `dev` extra in `pyproject.toml` so `tests/test_packaging_python_floor.py` (which imports `packaging.markers` and `packaging.version`) resolves an explicit, declared dependency instead of relying on it arriving transitively through `pytest`.
- Regenerate `uv.lock` so `packaging` appears as a direct `dev`-extra dependency of `fast-mlsirm`, keeping `uv lock --check` green.

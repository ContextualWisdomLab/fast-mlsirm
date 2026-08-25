# Crossed continuous evidence admission

## Fixed

- Preserve Boolean response compatibility while rejecting Python and NumPy Boolean values in crossed/MMMC item intercepts, item slopes, and person offsets before native discovery, preventing silent `False`/`True` to `0.0`/`1.0` reinterpretation of continuous scientific evidence.

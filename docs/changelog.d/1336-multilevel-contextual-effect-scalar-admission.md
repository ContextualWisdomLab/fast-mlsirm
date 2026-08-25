# Contextual-effect scalar admission

## Fixed

- Seal `weighted_contextual_effect()` continuous contextual-effect values before numeric conversion callbacks. Package-trusted Python/NumPy integer and floating scalars are normalized losslessly to inert binary64 values, while Boolean, complex, callback-bearing, non-finite, and lossy values fail closed before native discovery.

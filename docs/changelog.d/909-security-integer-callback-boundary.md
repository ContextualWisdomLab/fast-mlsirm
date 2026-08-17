# Answer-copying integer control boundary

## Security

- Harden Wollack omega, K-index, and K1/K2/S1/S2 row/count controls so only exact built-in integers and genuine supported NumPy integer scalars are normalized before compiled-core discovery; reject booleans, integer subclasses, and arbitrary coercion providers without executing caller callbacks.

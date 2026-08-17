# Reliability integer control boundary

## Security

- Harden Finn `s_levels`, Guttman `n_sample_splits`/`seed`, and Feldt `n_persons`/`n_items` so only exact built-in integers and genuine supported NumPy integer scalars are normalized before compiled-core discovery or ratings materialization; reject booleans, integer subclasses, and arbitrary coercion providers without executing caller callbacks.

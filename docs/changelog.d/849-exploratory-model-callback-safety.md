# Exploratory model factor-count callback safety

## Fixed

- Accept only exact built-in Python integers and genuine NumPy integer scalar types for exploratory factor counts, rejecting caller-defined integer subclasses before conversion callbacks can execute while preserving the existing positive-factor and multidimensional-support contracts.

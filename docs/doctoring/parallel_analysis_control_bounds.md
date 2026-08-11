# Parallel analysis public control and workspace bounds

## Standards and literature

Horn, J. L. (1965). A rationale and test for the number of factors in factor analysis. *Psychometrika, 30*(2), 179–185. https://doi.org/10.1007/BF02289447

Glorfeld, L. W. (1995). An improvement on Horn's parallel analysis methodology for selecting the correct number of factors to retain. *Educational and Psychological Measurement, 55*(3), 377–393. https://doi.org/10.1177/0013164495055003002

Open Web Application Security Project. (2021). *OWASP API security top 10 2023: API4 — unrestricted resource consumption*. OWASP Foundation. https://owasp.org/API-Security/

## Product application

Public `parallel_analysis` controls (`n_iterations`, `centile`, `seed`) are validated as exact integers of admitted types before PyO3 dispatch. Hostile `__int__` converters, booleans, floats, and strings fail closed. Iteration counts that would request an unbounded random-eigenvalue workspace are rejected in both the Python boundary and the Rust kernel allocation path so callers cannot exhaust memory with a single control value.

## Verification

- `tests/test_parallel_analysis_control_bounds.py`

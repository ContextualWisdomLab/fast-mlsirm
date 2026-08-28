# Rasch CML response structural budget

## Fixed

- Bound exact built-in Rasch CML response-tree traversal independently of logical numeric cells, so malformed empty-row fan-out cannot consume unbounded Python preflight work while keeping the response-cell count at zero.
- Preserve the existing 20,000,000-cell response envelope and every valid non-empty persons-by-items matrix inside it by applying a conservative 40,000,000 structural-node ceiling before NumPy materialization.
- Replay Rust-owned Rasch CML and Andersen result mappings before NumPy/scalar marshalling, rejecting callback-bearing, malformed-cardinality, or non-finite native payloads with package-owned runtime diagnostics while preserving the current valid Rust result shape.
- Bind Andersen per-group retained-person evidence to the admitted person partition: each exact native count must remain within the admitted-person bound and the aggregate retained count cannot exceed `n_persons`, preventing stale/foreign native payloads from claiming more retained persons than were admitted.
- Preserve logical-cell error precedence, exact NumPy/list/tuple compatibility, complete 0/1 response semantics, existing dimensionality/minimum-item diagnostics, and Andersen split behavior.
- Keep Rasch conditional likelihood, CML estimation, conditional-information standard-error arithmetic, Andersen likelihood-ratio arithmetic, convergence, and uncertainty unchanged in the Rust numerical core.
- Primary research basis for the unchanged Rasch/CML semantics: Andersen, E. B. (1970). “Asymptotic properties of conditional maximum-likelihood estimators.” *Journal of the Royal Statistical Society: Series B (Methodological), 32*(2), 283–301. https://doi.org/10.1111/j.2517-6161.1970.tb00842.x. This establishes conditional maximum likelihood by conditioning on sufficient statistics for incidental parameters; the structural-node ceiling and native-result replay are implementation trust/resource guards and do not alter that likelihood.
- Primary research basis for the unchanged split-group fit test: Andersen, E. B. (1973). “A goodness of fit test for the Rasch model.” *Psychometrika, 38*(1), 123–140. https://doi.org/10.1007/BF02291180. This derives the conditional likelihood-ratio comparison of group-specific versus overall item difficulties; the PR preserves that statistic and only bounds/replays the Python↔Rust boundary.

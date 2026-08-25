# Interaction-map empty complete-case result

## Fixed

- Normalize an empty complete-case interaction rectangle to empty respondent and item index sets so the Rust result remains shape-consistent with zero-length person/item coordinates and zero-by-zero reconstruction, unexplained-residual, and cross-share arrays at the Python boundary.
- Preserve the requested bounded `axis_shares` length for an empty map without inventing a maximal-complete-submatrix selection rule or retaining one non-empty axis after the other has collapsed.

Gabriel factorization, singular values, coordinate arithmetic, reconstruction, unexplained residuals, and cross-term calculations for non-empty complete-case rectangles remain unchanged and Rust-owned.

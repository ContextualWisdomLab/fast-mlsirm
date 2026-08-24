# BRATT control admission

## Fixed

- Validate and normalize Bradley-Terry-with-ties reference, iteration, and tolerance controls before comparison-data materialization or compiled-core discovery, rejecting callback-bearing scalar subclasses and protocol providers while preserving trusted built-in and NumPy scalar inputs.
- Keep BRATT probability, MM-update, reference-rescaling, convergence, and log-likelihood arithmetic unchanged in the Rust core.

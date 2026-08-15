### Security

- Harden `parallel_analysis()` control validation so caller-defined Python and NumPy integer subclasses cannot execute conversion or representation callbacks while establishing `n_iterations`, `centile`, or `seed`; invalid controls now fail before compiled-core discovery while genuine supported NumPy integer scalars, existing bounds, the 128 MiB random-workspace ceiling, and Rust-owned Horn/Glorfeld factor-retention arithmetic are preserved.

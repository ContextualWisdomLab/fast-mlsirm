# Compensatory 2PL control trust hardening

## Security

- Validate and normalize `q`, `estimate_corr`, `max_iter`, `tol`, `xi_points`, and `xi_seed` before response-array materialization or native-core discovery, rejecting caller-defined scalar subclasses and arbitrary conversion/truth-value providers without executing their callbacks.
- Preserve documented built-in and concrete NumPy scalar compatibility, Gauss-Hermite node choices, positive finite tolerance, iteration and QMC/MC point limits, and the full unsigned-64 integration-seed domain while passing only normalized built-in primitives to Rust.
- Keep compensatory 2PL likelihood, integration, ECM correlation estimation, convergence, and EAP arithmetic unchanged in the Rust core.

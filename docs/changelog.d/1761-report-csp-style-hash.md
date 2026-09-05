## Security

- Harden standalone HTML reports by replacing broad `style-src 'unsafe-inline'` admission in the reusable report renderers with a SHA-256 CSP hash-source bound to the exact inline stylesheet bytes. Validation and facets-calibration renderers that reuse the essay report CSP helper now pass the same `_css()` payload explicitly, preventing the shared-helper signature change from breaking those report paths.
- Add regression coverage that replays the exact CSS hash and the shared validation/calibration call sites. This is inline-style defense in depth; it does not claim that the former `style-src` policy independently enabled JavaScript execution.

## Security

- Harden standalone HTML reports by replacing broad `style-src 'unsafe-inline'` admission in the reusable report renderers with a SHA-256 CSP hash-source bound to the exact inline stylesheet bytes. Validation and facets-calibration renderers that reuse the essay report CSP helper now pass the same `_css()` payload explicitly, preventing the shared-helper signature change from breaking those report paths.
- Preserve data-dependent diagnostic bar widths without CSP-blocked `style` attributes by rendering bounded `progress` values instead. Exact numerical values remain available in the adjacent labels and governed exact-value tables; the chart stays decorative.
- Add regression coverage that replays the exact CSS hash, shared validation/calibration call sites, and absence of data-dependent inline style attributes. This is inline-style defense in depth; it does not claim that the former `style-src` policy independently enabled JavaScript execution.

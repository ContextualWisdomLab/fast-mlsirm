# Standalone report CSP style hashing

## Security

- Harden standalone HTML reports by replacing broad `style-src 'unsafe-inline'` admission in the reusable report renderers with a SHA-256 CSP hash-source bound to the exact inline stylesheet bytes. Validation and facets-calibration renderers that reuse the essay report CSP helper now pass the same `_css()` payload explicitly, preventing the shared-helper signature change from breaking those report paths.
- Preserve data-dependent diagnostic bar widths without CSP-blocked `style` attributes by rendering bounded `progress` values instead. Exact numerical values remain available in the adjacent labels and governed exact-value tables; the chart stays decorative.
- Keep meta-delivered CSP claims standards-accurate: portable report metadata no longer includes `frame-ancestors`, which CSP Level 3 requires browsers to ignore in a `<meta>` policy. Deployments that require anti-framing must enforce `frame-ancestors 'none'` in the HTTP `Content-Security-Policy` response header.
- Add regression coverage that replays the exact CSS hash, shared validation/calibration call sites, absence of data-dependent inline style attributes, and absence of ineffective meta `frame-ancestors`. This is inline-style defense in depth; it does not claim that the former `style-src` policy independently enabled JavaScript execution.

# Cross-engine conformance report accessibility

## Changed

- Add a keyboard skip link, explicit main-content focus target, focus-visible treatment, reduced-motion handling, and readable table styling to the standalone cross-engine conformance evidence report.
- Keep the self-contained report fail-closed under Content Security Policy by authorizing only the exact inline stylesheet bytes through a SHA-256 `style-src` hash; scripts and unsafe inline styles remain disallowed.
- Add regression evidence that the rendered stylesheet hash matches the emitted CSP and that the skip-link/focus contracts remain present.

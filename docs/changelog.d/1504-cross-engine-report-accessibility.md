# Cross-engine conformance report accessibility

## Changed

- Add a keyboard skip link, explicit main-content focus target, focus-visible treatment, reduced-motion handling, and readable table styling to the standalone cross-engine conformance evidence report.
- Bind the rendered light, dark-screen, and print canvases plus table boundaries to package-owned `--canvas`, `--text`, and `--line` tokens, and derive the 3:1 boundary-contrast regression from those exact production token values rather than assumed user-agent system colors.
- Keep the self-contained report fail-closed under Content Security Policy by authorizing only the exact inline stylesheet bytes through a SHA-256 `style-src` hash; scripts and unsafe inline styles remain disallowed.
- Add regression evidence that the rendered stylesheet hash matches the emitted CSP and that the skip-link/focus contracts remain present.

## Source basis

- W3C, *Understanding Success Criterion 2.4.1: Bypass Blocks* (WCAG 2.2), https://www.w3.org/WAI/WCAG22/Understanding/bypass-blocks.html — motivates the report's keyboard skip link to the main-content region so sequential navigation can bypass preceding/repeated material.
- W3C, *Understanding Success Criterion 2.4.7: Focus Visible* (WCAG 2.2), https://www.w3.org/WAI/WCAG22/Understanding/focus-visible.html — supports retaining an explicit visible keyboard-focus treatment for the programmatically focusable report target.
- W3C, *Understanding Success Criterion 2.3.3: Animation from Interactions* (WCAG 2.2), https://www.w3.org/WAI/WCAG22/Understanding/animation-from-interactions.html — identifies user motion preference, including `prefers-reduced-motion`, as a mechanism for suppressing non-essential interaction motion.
- W3C, *Understanding Success Criterion 1.4.11: Non-text Contrast* (WCAG 2.2), https://www.w3.org/WAI/WCAG22/Understanding/non-text-contrast.html — provides the 3:1 visual-boundary contrast basis; executable evidence computes contrast from the exact rendered `--line` and `--canvas` tokens for light, dark-screen, and print states.
- W3C, *Content Security Policy Level 3*, https://www.w3.org/TR/CSP3/ — defines hash-source matching for inline style content; the report authorizes the SHA-256 digest of the exact `_css()` UTF-8 bytes emitted inside its single `<style>` element instead of broad `unsafe-inline` styling.

These standards/guidance are cited, linked, and summarized rather than redistributed as repository PDFs.

# Essay-validation empty-state status semantics

## Changed

- Marked empty identifier evidence in standalone essay-validation HTML reports as a WAI-ARIA `status` region with explicit `aria-atomic="true"`, while preserving visible text and avoiding focus movement.
- Added a deterministic regression for the exact status markup and documented the interoperability boundary: live-region semantics improve assistive-technology exposure for status updates, but a pre-populated static report is not claimed to trigger an initial announcement.

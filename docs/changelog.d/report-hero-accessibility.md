# Accessible diagnostics-report hero metadata

## Changed

- Generated HTML diagnostics reports now expose the source filename as a
  semantic description list and hide redundant decorative branding from the
  accessibility tree.
- Regression coverage pins the `aria-hidden`, `dl`, `dt`, and `dd` markup so
  future template changes cannot silently remove the accessibility contract.

# Item-bank report skip-link focus visibility (#2025)

## Fixed

- Keep `<main id="main-content" tabindex="-1">` as the programmatic skip-link
  target without suppressing its focus indicator.
- Reject compact or whitespace-separated `outline: none` reintroduction in the
  generated report contract.
- Browser, assistive-technology, responsive, locale, and performance acceptance
  remains pending; source assertions are not real-browser evidence.

## 2025-02-18 - Semantic Metrics & Redundant Charts
**Learning:** Using semantic `<dl>`, `<dt>`, `<dd>` elements for key-value pairs (like metric cards) significantly improves screen reader accessibility. Also, hiding purely visual components (like compact bar charts) using `aria-hidden="true"` when the exact same data is available in a subsequent accessible table reduces screen reader noise.
**Action:** Always prefer description lists for key-value displays and hide visual-only redundancies from screen readers to keep the auditory experience clean.

## 2024-05-15 - Focus Visible For Scrollable Tables\n**Learning:** Using :focus-visible instead of :focus for scrollable containers improves accessibility for keyboard users without causing jarring outlines for mouse users.\n**Action:** Use :focus-visible on focusable generic containers like `.table-wrap[tabindex="0"]` in Python HTML report templates.
## 2024-05-15 - Smooth Table Row Hover Transitions
**Learning:** Table row hover states without transitions feel abrupt and unpolished in data-heavy static reports, and adding hover states helps users track their reading position across wide tables.
**Action:** Apply a `transition: background-color 0.15s ease-in-out` to `tbody tr` elements and a subtle `background` color on `tbody tr:hover` in Python HTML report templates.
## 2024-07-12 - Print Optimization for HTML Reports
**Learning:** Static HTML data reports are often printed or exported to PDF, but default browser settings strip background colors, which completely hides CSS-based bar charts and removes visual grouping from metric cards.
**Action:** Use `@media print` with `print-color-adjust: exact` to preserve data visualizations, set body background to white to save ink, hide screen-reader skip links, and apply `break-inside: avoid` on sections to prevent awkward page breaks.

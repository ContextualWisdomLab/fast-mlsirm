## 2024-05-15 - Focus Visible For Scrollable Tables\n**Learning:** Using :focus-visible instead of :focus for scrollable containers improves accessibility for keyboard users without causing jarring outlines for mouse users.\n**Action:** Use :focus-visible on focusable generic containers like `.table-wrap[tabindex="0"]` in Python HTML report templates.
## 2024-05-15 - Smooth Table Row Hover Transitions
**Learning:** Table row hover states without transitions feel abrupt and unpolished in data-heavy static reports, and adding hover states helps users track their reading position across wide tables.
**Action:** Apply a `transition: background-color 0.15s ease-in-out` to `tbody tr` elements and a subtle `background` color on `tbody tr:hover` in Python HTML report templates.
## 2024-07-12 - Print Optimization for HTML Reports
**Learning:** Static HTML data reports are often printed or exported to PDF, but default browser settings strip background colors, which completely hides CSS-based bar charts and removes visual grouping from metric cards.
**Action:** Use `@media print` with `print-color-adjust: exact` to preserve data visualizations, set body background to white to save ink, hide screen-reader skip links, and apply `break-inside: avoid` on sections to prevent awkward page breaks.
## 2024-10-24 - CSS Bar Chart Animation
**Learning:** CSS animations can enhance static data visualizations without requiring JavaScript, providing visual polish and reducing perceived loading times for data.
**Action:** Use CSS keyframe animations for simple visual improvements in static reports.
## 2026-07-23 - Tabular Nums for Data Reports
**Learning:** In data-heavy reports, numbers with varying widths can be hard to read and scan when stacked vertically.
**Action:** Use `font-variant-numeric: tabular-nums;` in body styles or data-heavy components within HTML reports to ensure numbers align properly vertically for improved readability and scanning.

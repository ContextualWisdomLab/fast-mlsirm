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
## 2024-11-20 - ARIA Status Role for Empty States
**Learning:** Using `role="status"` on empty state containers (e.g., `<div class="empty-state" role="status">`) instead of standard paragraph tags ensures assistive technologies actively announce when no data or rows are recorded, without interrupting the user's flow.
**Action:** Use `role="status"` on empty states in HTML components instead of standard `p` tags.

## 2024-11-20 - Tabular Nums for Readability
**Learning:** Numbers of varying widths can be hard to scan vertically in data-heavy reports, negatively impacting readability and usability.
**Action:** Always include `font-variant-numeric: tabular-nums;` in body styles or data-heavy components within HTML reports to ensure numbers align properly vertically.

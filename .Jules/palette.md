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
## 2025-01-28 - ARIA Live Regions for Empty States
**Learning:** Using standard `<p>` tags for empty states in dynamic or conditionally rendered tables can lead to screen readers silently skipping the section, leaving users confused about missing data.
**Action:** Use `<div class="empty-state" role="status">` for empty states to leverage ARIA live regions, ensuring assistive technologies actively announce that no rows were recorded.
## 2025-02-12 - Native Dark Mode for Static Reports
**Learning:** Hardcoded light colors in static HTML reports ignore OS-level theme preferences, leading to poor UX and increased eye strain for users who rely on dark mode in data-heavy environments.
**Action:** Extract hardcoded hex colors into CSS variables and add a `@media (prefers-color-scheme: dark)` block to support native dark mode in Python-generated HTML report templates.
## 2026-08-03 - [HTML Table Numeric Typography]
**Learning:** For standalone HTML reports that display large volumes of numeric data (like metrics and diagnostic data), the default proportional fonts can cause the decimals and numbers to snake horizontally, hindering scan-ability.
**Action:** When adding or styling tables with metrics or numeric values in HTML reports, inject the 'font-variant-numeric: tabular-nums;' CSS property into table cells ('th', 'td') to align numbers beautifully and enhance readability.
## 2026-08-04 - Focus Visible For Scrollable Code Blocks
**Learning:** Code blocks (`<pre>`) that contain wide text (like JSON or CSV exports) require horizontal scrolling, but by default they cannot receive keyboard focus, locking keyboard-only users out of viewing the full content.
**Action:** Always add `tabindex="0"`, `role="region"`, `aria-label`, and a `:focus-visible` outline to scrollable `<pre>` or code containers to ensure full keyboard navigability and clear visual focus feedback.
## 2026-08-04 - Status Semantics and Numeric Alignment for Reports
**Learning:** Explicit status semantics can make conditionally rendered empty states easier to discover with assistive technology, while tabular numerals improve visual comparison of metric columns. Focus-reveal behavior must not depend only on `:focus-visible`, and hover styling must not reduce the contrast of unrelated rows.
**Action:** Use `role="status"` for genuine conditionally rendered status messages, apply `font-variant-numeric: tabular-nums` to numeric report tables, reveal skip links on `:focus`, retain a visible `:focus-visible` indicator, and avoid opacity-based dimming of non-hovered content.
## 2026-08-05 - Fail-Safe Pointer Focus Outline Suppression
**Learning:** Globally suppressing focus outlines for mouse users with `:focus { outline: none; }` destroys keyboard accessibility on user agents that do not support `:focus-visible`. A fail-safe selector approach uses `:focus:not(:focus-visible)` to explicitly target non-keyboard interaction, preserving the user agent default when `:focus-visible` is unsupported. Pointer-only ring suppression is a visual polish feature, not a complete solution for whole-product WCAG conformance; pointer users may still benefit from focus indication.
**Action:** When suppressing pointer focus rings, always use `:focus:not(:focus-visible) { outline: none; }`.
References:
- W3C. (2025). *Understanding Success Criterion 2.4.7: Focus Visible*.
- W3C. (2025). *C45: Using CSS :focus-visible to provide keyboard focus indication*.
- W3C. (2025). *Selectors Level 4*.

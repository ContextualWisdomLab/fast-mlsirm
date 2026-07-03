## 2025-03-03 - Focus outlines for scrolling tables
**Learning:** Using `tabindex="0"` on table wrappers allows keyboard users to scroll horizontal tables, which is excellent for a11y. However, applying outlines via the `:focus` pseudo-class creates a distracting visual ring when mouse users click the table.
**Action:** Use `:focus-visible` instead of `:focus` for container elements like `.table-wrap`. This preserves keyboard accessibility outlines while hiding them for mouse clicks.

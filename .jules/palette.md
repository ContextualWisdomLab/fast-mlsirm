## 2025-05-25 - Focus-Visible for Scrollable Containers
**Learning:** Using `:focus` on elements with `tabindex="0"` (like `.table-wrap`) causes a jarring visual outline when a mouse user clicks on the scrollable area to scroll, which degrades the UX. Keyboard users still need this outline for a11y navigation.
**Action:** Always use `:focus-visible` instead of `:focus` for generic focusable containers like scrollable areas, ensuring focus outlines only appear for keyboard navigation and not on mouse click.

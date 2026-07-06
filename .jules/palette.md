## 2025-05-18 - Focus-visible for scrollable containers
**Learning:** Using `:focus` for focusable scroll containers (like tables with `tabindex="0"`) creates a visually jarring outline when mouse users click on them to scroll. This negatively impacts the UX for non-keyboard users.
**Action:** Always use `:focus-visible` instead of `:focus` for generic focusable containers to ensure the outline only appears during keyboard navigation, preserving accessibility while avoiding visual clutter for mouse interactions.

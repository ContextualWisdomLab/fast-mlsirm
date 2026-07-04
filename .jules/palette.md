## 2026-07-04 - Keyboard Focus UX Improvement
**Learning:** Using `:focus-visible` on interactive scrollable containers (like `tabindex="0"` tables) ensures focus outlines only appear for keyboard navigation, avoiding distracting focus rings on mouse click.
**Action:** Always prefer `:focus-visible` over `:focus` for custom focus styles unless it is explicitly an input element where focus style is expected on click.

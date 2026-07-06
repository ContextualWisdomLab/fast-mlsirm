## 2024-10-24 - Improve focus styles for pointer users
**Learning:** Using standard `:focus` on scrollable regions (like `.table-wrap`) can create visual noise because clicking inside the table with a mouse triggers the focus ring.
**Action:** Use `:focus-visible` instead of `:focus` so that focus outlines only appear for keyboard navigation and not on mouse clicks, providing a cleaner UI without sacrificing accessibility.

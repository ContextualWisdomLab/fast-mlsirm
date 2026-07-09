## 2024-05-15 - Add Empty State for Diagnostics Report Table
**Learning:** Tables generated without rows resulted in broken structural sections where a caption exists without meaningful data, confusing the report context.
**Action:** Always include an empty-state block (e.g. `<p class="empty-state">No rows...</p>`) when rendering UI components that dynamically consume arrays/lists so users aren't left with empty space and can quickly verify there is no data to evaluate.
## 2024-05-15 - Use :focus-visible for Scrollable Regions
**Learning:** Using `:focus` on scrollable regions like `.table-wrap` causes a distracting focus ring when a user simply clicks into the area to scroll with a mouse/trackpad.
**Action:** Use `:focus-visible` for scrollable containers so that keyboard users (tabbing) still get the necessary focus ring for accessibility, while mouse/pointer users do not.

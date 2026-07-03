
## 2024-05-18 - HTML Report Focus Styling
**Learning:** Hardcoding `:focus` outlines in generated HTML reports creates unnecessary visual noise for mouse users clicking on interactive table wrappers or elements.
**Action:** Always use `:focus-visible` instead of `:focus` when generating static HTML reports with embedded CSS to ensure keyboard navigation accessibility without compromising mouse/touch user experience.

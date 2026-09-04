## 2024-07-10 - HTML Report Accessibility

## 2024-07-10 - Table Summary Associations via aria-describedby
**Learning:** In the HTML report generation, tables with truncated rows previously appended a visual "<p>Showing X of Y rows.</p>" note outside the `<table>` element. Screen reader users navigating to the `<div class="table-wrap" role="region" tabindex="0">` would not have this contextual truncation limit read out to them.
**Action:** When creating accessible regions that summarize data, ensure secondary explanatory text (like row counts/truncation warnings) is programmatically associated with the main region container using `aria-describedby="[note-id]"` so that assistive technologies announce the context alongside the container's `aria-label`.

## 2024-07-11 - Skip-to-Content Links in HTML Reports
**Learning:** Standalone HTML reports require skip-to-content links for keyboard/screen reader users, just like standard web applications, to avoid forcing users to navigate through repetitive or non-essential visual elements at the top of the page.
**Action:** Always include a `skip-link` right after the body tag and set `id="main-content"` on the primary content container.

## 2024-07-11 - Skip-to-Content Link Target Focus
**Learning:** Adding a `skip-link` pointing to `#main-content` is not enough for keyboard accessibility; the target `<main>` element must be programmatically focusable (`tabindex="-1"`) and its default focus outline should be removed (`outline: none;`) to ensure the user's focus correctly shifts into the main content area without displaying an unnecessary visual artifact.
**Action:** Always add `tabindex="-1"` and `outline: none;` to the primary content container that serves as the skip link target.

## 2024-07-12 - HTML 리포트의 화면 전환 효과 최소화 (Reduced Motion)
**Learning:** 독립형 HTML 리포트에는 시각적 부드러움을 위해 CSS 전환 효과(예: skip-link 슬라이딩, 테이블 행 호버 효과)가 포함되어 있으나, 이는 전정기관 장애가 있는 사용자에게 불편함을 줄 수 있습니다. 접근성을 완전히 확보하려면 시스템 설정에서 애니메이션 최소화(prefers-reduced-motion)를 선택한 사용자를 위해 이를 비활성화하는 미디어 쿼리가 필수적이라는 점을 배웠습니다.
**Action:** 생성되는 HTML 리포트의 CSS에 항상 `@media (prefers-reduced-motion: reduce)` 블록을 포함하여, 접근성을 고려한 사용자 환경에서는 `transition-duration`, `animation-duration`, `scroll-behavior`가 즉시 처리되도록 적용합니다.

## 2024-07-13 - CLI Debugging Stack Traces
**Learning:** Adding a `FAST_MLSIRM_DEBUG` bypass to user-friendly `try/except` blocks is crucial for DX. Otherwise, unexpected runtime errors during development will be swallowed into generic stderr messages, hiding the stack trace needed to actually fix the bug.
**Action:** When adding `try-except` blocks to Python CLI subcommands to improve Developer Experience (DX) by preventing raw tracebacks for users, include a debug bypass (e.g., `if os.environ.get("FAST_MLSIRM_DEBUG"): raise`) in *all* catch blocks (including `RuntimeError` and `Exception`) to ensure tracebacks aren't swallowed during local development and debugging.

## 2025-02-12 - Table Body `<th scope="row">` and Tabular Numbers
**Learning:** Using `<td>` for the first column in data tables makes it difficult for screen reader users to associate row data with its identifying header. Additionally, numbers in data tables can jitter horizontally if proportional fonts are used.
**Action:** When generating HTML data tables, always use `<th scope="row">` for the first identifying column to improve screen reader accessibility. Apply `font-variant-numeric: tabular-nums;` to base table cell styles to ensure numbers align properly, and explicitly update primary cell CSS selectors (e.g., from `th, td { ... }` to `thead th, tbody th, td { ... }`) and add `tbody th { font-weight: normal; }` to maintain consistent baseline styling.

## 2025-02-12 - Semantic `<section>` Accessible Names
**Learning:** Screen readers only treat `<section>` as a landmark region if it has an accessible name. Without an accessible name, the region is not easily navigable via screen reader rotor menus.
**Action:** Always provide an accessible name to `<section>` using `aria-labelledby` pointing to its main heading's `id`.

## 2025-02-12 - CSS Hover-Focus Isolation for Dense Visualizations
**Learning:** Dense bar charts or lists can be difficult to visually parse. Highlighting the currently hovered row by dimming the surrounding rows greatly improves visual focus and UX.
**Action:** Use a CSS pattern like `.container:hover .item:not(:hover) { opacity: 0.5; }` (along with `transition` properties on the item) to isolate visual focus during interaction with dense data visualizations.

## 2024-08-01 - Focus Visible Styles for Skip-to-Content Targets
**Learning:** While `outline: none;` on a `<main>` container properly removes the visual artifact when users click inside the content area, it completely breaks keyboard accessibility for users navigating via the "Skip to main content" link because no focus indicator is shown when the target is focused.
**Action:** When overriding the focus outline on semantic containers like `<main>`, always provide a `.element:focus-visible` rule (e.g., `outline: 3px solid var(--primary-color)`) after the `:focus { outline: none; }` rule to ensure keyboard navigation remains visibly accessible without disrupting mouse interactions.

## 2025-02-12 - CSS Hover-Focus Isolation for Dense Visualizations
**Learning:** Dense bar charts or lists can be difficult to visually parse. Highlighting the currently hovered row by dimming the surrounding rows greatly improves visual focus and UX.
**Action:** Use a CSS pattern like `.container:hover .item:not(:hover) { opacity: 0.5; }` (along with `transition` properties on the item) to isolate visual focus during interaction with dense data visualizations.

## 2024-07-10 - HTML Report Semantic Definition Lists
**Learning:** Using `<span>` and `<p>` elements for displaying key-value metadata pairs (like "Source: file.json") in HTML reports misses an opportunity to provide semantic structure. Additionally, purely decorative text preceding main headings (like a small subtitle) can add noise for screen reader users when read out immediately before the main `<h1>`.
**Action:** In HTML reports, prefer semantic `<dl>`, `<dt>`, `<dd>` elements for key-value data (explicitly resetting default `<dt>` font-weight for cross-browser visual consistency), and use `aria-hidden="true"` to hide purely visual, redundant components from screen readers.

## 2025-02-12 - Exposing Unrounded Numeric Representations for Formatted Floats
**Learning:** In data-heavy HTML reports, formatting floats to a fixed number of significant digits can obscure the original Python float representation. A native `title` tooltip can help pointer users inspect that unrounded representation, but it is not a reliable keyboard, touch, or assistive-technology disclosure mechanism.
**Action:** Add the unrounded Python float representation as a supplemental native `title` tooltip on formatted values. Preserve the report's accessible exact-value disclosure and JSON/CSV exports as the authoritative non-hover paths; never claim that `title` alone provides accessibility.

## 2026-08-11 - Do Not Use Opacity Dimming for Focus Isolation
**Learning:** Adding hover-focus isolation to dense visualizations by dropping the opacity of non-hovered elements (e.g., `tbody:hover tr:not(:hover) { opacity: 0.5; }`) breaks project accessibility rules regarding peer contrast and causes CI tests (e.g., `test_hover_does_not_dim_unrelated_chart_or_table_content`) to fail. Tests that strictly enforce contrast constraints must not be modified just to pass CI.
**Action:** Do not apply CSS hover-focus isolation patterns (e.g., dimming non-hovered rows via `opacity`) in dense data visualizations like bar charts or list grids.

## 2026-08-11 - Replace Hardcoded System Colors with CSS Variables
**Learning:** Hardcoding system colors like `GrayText` in HTML reports bypasses the project's contrast-managed CSS variables, leading to inconsistent UI styling and potential contrast issues across light and dark themes.
**Action:** When defining border or text colors for elements like `.empty-state`, `<pre>`, or `<table>`, always use defined CSS variables such as `var(--muted)` for text and `var(--line)` for borders. Ensure these variables are declared in both the default `:root` and dark mode `@media` contexts.

## 2026-08-11 - Enforcing Contrast and High-Contrast Mode for CSS Variables
**Learning:** When defining palette tokens like `--muted` (for text) and `--line` (for borders), arbitrary hex codes may fail strict contrast tests (e.g., 4.5:1 for text, 3:1 for borders). Additionally, author-defined RGB values interfere with OS-level high-contrast settings unless explicitly delegated.
**Action:** Always measure hex codes against pure white (`#ffffff`) and pure black (`#000000`) to ensure contrast ratio boundaries are respected. Furthermore, always add an `@media (forced-colors: active)` block to delegate decorative variables (e.g., `--muted: CanvasText; --line: CanvasText;`) to semantic system colors, ensuring the UI remains accessible for users relying on forced color modes.

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

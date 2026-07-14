## 2024-07-10 - HTML Report Accessibility

## 2024-07-10 - Table Summary Associations via aria-describedby
**Learning:** In the HTML report generation, tables with truncated rows previously appended a visual "<p>Showing X of Y rows.</p>" note outside the `<table>` element. Screen reader users navigating to the `<div class="table-wrap" role="region" tabindex="0">` would not have this contextual truncation limit read out to them.
**Action:** When creating accessible regions that summarize data, ensure secondary explanatory text (like row counts/truncation warnings) is programmatically associated with the main region container using `aria-describedby="[note-id]"` so that assistive technologies announce the context alongside the container's `aria-label`.

## 2024-07-11 - Skip-to-Content Links in HTML Reports
**Learning:** Standalone HTML reports require skip-to-content links for keyboard/screen reader users, just like standard web applications, to avoid forcing users to navigate through repetitive or non-essential visual elements at the top of the page.
**Action:** Always include a `skip-link` right after the body tag and set `id="main-content"` on the primary content container.

## 2024-07-11 - Skip-to-Content Link Target Focus
**Learning:** `#main-content`를 가리키는 `skip-link`를 추가하는 것만으로는 키보드 접근성에 충분하지 않습니다; 마우스 클릭 시 불필요한 시각적 아티팩트 없이 사용자의 초점이 올바르게 메인 콘텐츠 영역으로 이동할 수 있도록 대상 `<main>` 요소는 프로그래밍 방식으로 초점을 맞출 수 있어야 하며(`tabindex="-1"`), 기본 초점 윤곽선을 제거(`outline: none;`)해야 합니다. 그러나 키보드 사용자가 skip link가 작동했음을 시각적으로 인지할 수 있도록 `:focus-visible` 상태가 필요합니다.
**Action:** skip link 대상 역할을 하는 기본 콘텐츠 컨테이너에 `:focus` 상태를 위한 `tabindex="-1"` 및 `outline: none;`을 항상 추가하고, 키보드 사용자가 초점이 이동한 위치를 볼 수 있도록 `:focus-visible` 윤곽선을 함께 적용하세요.

## 2024-07-12 - HTML 리포트의 화면 전환 효과 최소화 (Reduced Motion)
**Learning:** 독립형 HTML 리포트에는 시각적 부드러움을 위해 CSS 전환 효과(예: skip-link 슬라이딩, 테이블 행 호버 효과)가 포함되어 있으나, 이는 전정기관 장애가 있는 사용자에게 불편함을 줄 수 있습니다. 접근성을 완전히 확보하려면 시스템 설정에서 애니메이션 최소화(prefers-reduced-motion)를 선택한 사용자를 위해 이를 비활성화하는 미디어 쿼리가 필수적이라는 점을 배웠습니다.
**Action:** 생성되는 HTML 리포트의 CSS에 항상 `@media (prefers-reduced-motion: reduce)` 블록을 포함하여, 접근성을 고려한 사용자 환경에서는 `transition-duration`, `animation-duration`, `scroll-behavior`가 즉시 처리되도록 적용합니다.

## 2024-07-13 - CLI Debugging Stack Traces
**Learning:** Adding a `FAST_MLSIRM_DEBUG` bypass to user-friendly `try/except` blocks is crucial for DX. Otherwise, unexpected runtime errors during development will be swallowed into generic stderr messages, hiding the stack trace needed to actually fix the bug.
**Action:** When adding `try-except` blocks to Python CLI subcommands to improve Developer Experience (DX) by preventing raw tracebacks for users, include a debug bypass (e.g., `if os.environ.get("FAST_MLSIRM_DEBUG"): raise`) in *all* catch blocks (including `RuntimeError` and `Exception`) to ensure tracebacks aren't swallowed during local development and debugging.

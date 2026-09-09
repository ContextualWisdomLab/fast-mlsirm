# Commercial report table focus visibility

## Changed

- 마우스 클릭 시 시각적으로 거슬리는 기본 포커스 링을 제거하면서 키보드 접근성(`.table-wrap:focus-visible`)은 엄격하게 보존하기 위해, benchmark, buyer packet, PR queue governance, procurement due diligence, release evidence index 리포트의 `.table-wrap:focus:not(:focus-visible)` 규칙에 `outline: none`을 추가했습니다.
- 다섯 리포트를 실제 Chrome/ChromeDriver에서 렌더링하는 PR-head E2E를 추가해 키보드 Tab 포커스, 포인터 포커스, 포커스 표시와 중심부 비가림, 모바일 내부 가로 스크롤, 중간 너비·데스크톱의 페이지 오버플로를 검증하고 브라우저/드라이버 버전과 각 HTML SHA-256을 90일 증거 아티팩트로 보존합니다. 검증기는 Python 표준 라이브러리로 W3C WebDriver를 직접 호출하므로 Selenium/Playwright 런타임 의존성을 제품 또는 테스트 의존성에 추가하지 않습니다.
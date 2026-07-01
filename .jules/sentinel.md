## 2026-07-01 - Add Content-Security-Policy to HTML reports
**Vulnerability:** Missing Content-Security-Policy (CSP) in dynamically generated standalone HTML reports.
**Learning:** `python/fast_mlsirm/report.py` generates local, standalone HTML reports. Although the inputs are numerical and JSON, adding a strict CSP (`default-src 'none'; style-src 'unsafe-inline'`) prevents unexpected resource loading and potential XSS if untrusted input is somehow passed in the future.
**Prevention:** Include restrictive CSP tags in all generated HTML, even for offline data visualization.

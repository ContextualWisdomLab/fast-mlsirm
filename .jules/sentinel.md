## 2026-08-28 - Strix LLM Outages are Non-Actionable
**Learning:** Strix CI check failures stating "LLM Provider NOT provided" or "provider/backend was unavailable" are caused by intermittent third-party LLM infrastructure outages, not codebase regressions.
**Action:** Do not attempt to alter codebase logic to fix these specific external connection errors. Instead, use the `message_user` tool with `continue_working: false` to inform the user and halt execution.

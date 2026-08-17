## 2025-02-21 - [Prevent subprocess hang DoS]
**Vulnerability:** External `subprocess.run` calls without timeouts can hang indefinitely due to network disruptions, creating a Denial of Service (DoS) risk.
**Learning:** Adding a bounded `timeout` to `subprocess.run` (e.g. `timeout=60`) prevents operations from hanging indefinitely and safely mitigates the risk by capturing `subprocess.TimeoutExpired` properly.
**Prevention:** Always supply a configured or defaulted `timeout` argument to `subprocess.run` executions across all automation scripts and core functionalities.

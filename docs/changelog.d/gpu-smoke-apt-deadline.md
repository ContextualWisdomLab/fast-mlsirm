# Bound GPU smoke package provisioning

## CI / Reliability

- Bound Vulkan package index and installation network/lock waits with explicit APT request, retry, lock, and whole-command deadlines so a hosted-runner mirror stall fails with actionable provisioning evidence instead of consuming the full GPU job timeout.
- Preserve the existing llvmpipe Vulkan adapter proof and explicit CPU/GPU parity test; this changes CI provisioning reliability only, not production numerical behavior.

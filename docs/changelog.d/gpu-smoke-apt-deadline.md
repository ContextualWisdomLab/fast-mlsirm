# Bound GPU smoke package provisioning

## Fixed

- Bound Vulkan package index and installation network/lock waits with explicit APT request, retry, lock, and whole-command deadlines so a hosted-runner mirror stall fails with actionable provisioning evidence instead of consuming the full GPU job timeout.
- Route the GPU smoke job through an isolated deb822 source list backed by the canonical Ubuntu archive and security endpoints, preventing the hosted runner's `mirror+file` registry from repeatedly selecting a black-holed Azure mirror for package payloads after metadata fallback.
- Preserve the existing llvmpipe Vulkan adapter proof and explicit CPU/GPU parity test; this changes CI provisioning reliability only, not production numerical behavior.

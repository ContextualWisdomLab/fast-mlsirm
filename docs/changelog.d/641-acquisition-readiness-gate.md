# Acquisition readiness gate

## Changed

- Sales readiness no longer fabricates a KRW 2B contract value by default; deal scenarios must be supplied explicitly, and a generic `--require-acquisition-readiness` profile now activates buyer-packet, benchmark, release-evidence, procurement, PR-queue, and Figma validators without the legacy 20B file/token bundle.
- The legacy `--require-20b-product` profile remains a deprecated compatibility mode, while the manifest records generic readiness, compatibility mode, and transaction-scenario identity separately.

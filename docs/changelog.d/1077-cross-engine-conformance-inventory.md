# Cross-engine conformance inventory contract

## Added

- Add a provider-neutral, source-free `ConformanceInventory` contract for independent numerical conformance coverage. The first slice records public estimands, parameterization and identification scope, isolated engine/version/license identity, versioned parameter-mapping and fixture/environment fingerprints, and explicit passed/failed/indeterminate/not-executed states without adding external engines as runtime, build, package, or release dependencies. This is Python validation/provenance schema work only; production psychometric and statistical arithmetic remains Rust-owned.
- Accept both full Git SHA-1 and SHA-256 commit identities so protected-main and harness provenance remains valid across repository hash-format migrations.
- Require at least one executed evidence row before a capability can claim
  `covered` or `partially_covered` status.
- Revalidate exact package-owned engine, evidence, capability, and inventory records before manifest or fingerprint replay so post-construction field rebinding cannot bypass semantic-control, fingerprint, or collection admission; hostile enum controls and container subclasses fail closed before their callbacks execute.

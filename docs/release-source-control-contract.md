# Release source and execution identities

The release source R, caller execution commit C, pinned helper H, and reusable
attestation signer S are distinct identities. This candidate adds only the
missing R-ancestor-C check to direct package publication, after the full-history
checkout and before version/tag verification. The earlier control-plane step
already requires C to equal the event SHA on the default-branch ref. Missing
commit objects fail closed; no fetch fallback or ancestor inference is used.

The existing release-tag workflow checks the same ancestry before dispatch.
The direct workflow_dispatch entry point must enforce it independently. Existing
tag, version, exact checkout, admission, and publication dependencies remain
required. Streaming transport is inherited from b4f897d0 without modification.

Central attestation at 1916e95a still requires source SHA equal event SHA at two
boundaries. Changing that contract remains pending. Its helper revision is
00c6551183cca101cfc97c43656a17cc2491c1b4, not an assertion of the signer revision.
Any future separation must verify the signer repository/path/digest, authentic
certificate source identity, and independently bind R to the exact built bytes.
Same-run immutable artifact IDs and complete matrix evidence remain required.

Primary documentation read 2026-09-24:

- https://cli.github.com/manual/gh_attestation_verify — Understanding
  Verification, Additional Policy Enforcement, and options source-digest,
  signer-digest, signer-workflow. Certificate fields derive from OIDC;
  predicate data can be controlled by the originating workflow. Signing an SBOM
  does not by itself establish successful license/Strix checks or complete
  platform coverage.
- https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/increase-security-rating
  — reusable builder and attestation signing requirements. This candidate does
  not claim SLSA level 3.

Actual hosted certificate behavior when R differs from C, effective caller
permissions, trusted same-run aggregation, and twelve-wheel plus sdist dependency
closure remain unverified. Central equality checks and source-digest arguments
are outside this candidate. No hosted run or publication acceptance follows
from the local Git guard tests.

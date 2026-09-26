# Release scope identity: declarations, not closure acceptance

This first wiring increment extends the existing reproducibility-record artifact;
it does not add jobs or resolve dependencies. The existing twelve wheel matrix
legs and separate sdist produce the same distributions and TSV rows. The record
job checks out the approved control commit for the helper and the exact release
source for immutable declaration blobs, then writes `release-scope-identities.json`
alongside the TSV. Its Ubuntu host is not evidence about target dependencies.

Each identity binds the release SHA, distribution filename/hash, recorded build
environment, target/Python/ABI/platform/WHEEL tags, actual METADATA or PKG-INFO
member hash, and tracked lock/pyproject/Cargo manifest/requirements blob hashes.
Wheel metadata must be unique and its WHEEL tags agree with its filename and leg.
The sdist has a separate source identity, unique PKG-INFO/pyproject members under
one root, and pyproject bytes equal to the release source. These checks never
execute package code. Existing transport hashing remains chunked.

`release-admission` first verifies the selected immutable artifact IDs, ZIP and
member digests and existing distribution/sealed evidence checks. It then consumes
the JSON from the same selected reproducibility artifact and recomputes every
identity using exact release blobs and transported distribution bytes. Missing,
duplicate, changed, cross-platform or promoted records refuse admission before
the existing final trusted-full-gate HOLD. No name-only fallback is added.

Runtime, build, dev, optional, native and bundled scope entries are explicitly
`UNKNOWN` with null evidence. Complete declarations, an empty dependency array
or an asserted boolean cannot certify any scope. In particular, hashing every
Cargo lock does not establish the features/targets used in compiled wheels;
METADATA hashes do not establish resolved marker or optional dependency closure.
No trusted collector is connected in this increment, so there is deliberately
no accepted positive scope sentinel. A future owner must bind collector evidence
and full expected-set checks before adding that positive path. The source/control
and attestation signer contracts, complete license verdicts, platform-specific
native inspection and full same-run gate aggregation remain separate blockers.

Backward compatibility: prior reproducibility artifacts lacking the JSON cannot
be admitted. The trusted control commit must carry the updated helper and
workflow together. Existing final HOLD, tag/publish ordering and central verifier
remain. Focused synthetic tests exercise the actual workflow producer block,
ID-bound transport and scope-consumer block; they do not run a build or certify
real dependencies. No hosted run or full license acceptance is claimed.

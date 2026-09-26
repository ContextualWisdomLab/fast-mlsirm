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

`release-admission` first requires the pinned central gate job to succeed, then
verifies the selected immutable artifact IDs, ZIP and member digests. The
central full-set verdict must match the same run and attempt, exact release
source, all thirteen distribution rows, and the complete Strix binding set. It
then consumes
the JSON from the same selected reproducibility artifact and recomputes every
identity using exact release blobs and transported distribution bytes. Missing,
duplicate, changed, cross-platform or promoted records refuse admission before
the remaining platform-scope HOLD. No name-only fallback is added.

Runtime, build, dev, optional, native and bundled scope entries are explicitly
`UNKNOWN` with null evidence. Complete declarations, an empty dependency array
or an asserted boolean cannot certify any scope. In particular, hashing every
Cargo lock does not establish the features/targets used in compiled wheels;
METADATA hashes do not establish resolved marker or optional dependency closure.
No trusted target-scope collector is connected yet, so there is deliberately no
accepted positive scope sentinel. The central licence and Strix verdict now has
its own authenticated path; it does not claim platform scope completeness.

## Evidence needed to remove the scope HOLD

Each of the twelve wheel build jobs and the sdist job must produce a scope
inventory from its actual build environment. Put each inventory beside its
existing TSV in that leg's `repro-digest-*` artifact; the record job must bind
the exact thirteen digest-artifact IDs and archive digests to the corresponding
distribution SHA, source SHA, build-environment identity, run ID and attempt.
Admission must download those selected IDs, verify archive and member digests,
and check the inventories against the distributions. A missing, duplicate,
stale, foreign-run, wrong-target or changed inventory must refuse release.
Keep these evidence artifacts separate from the thirteen distribution artifacts
passed to the central licence and Strix gate.

The six scopes have distinct sources of truth:

- **Runtime and optional Python:** record the target Python interpreter and
  installed or otherwise fully resolved distribution files, versions, markers,
  extras and hashes. Compare them with the wheel's `Requires-Dist` and the
  applicable `uv.lock` resolution. The lock is universal; its mere presence is
  not a target installation. The `fuzz` extra now limits Atheris to its locked
  CPython/Linux x86_64 wheel targets; other targets retain Hypothesis only.
  That marker is a support boundary, not proof of an installed optional closure.
- **Build and dev:** record the build environment's installed Python tools and
  target-filtered Cargo resolution with the features used by the wheel build.
  `requirements/package.txt` and `Cargo.lock` constrain these scopes but do not
  prove which packages the build loaded. For Linux wheels, collect inside the
  digest-pinned manylinux container used by `maturin-action`; the surrounding
  runner's Python and Cargo inventories describe a different environment. The
  sdist needs its own build scope.
- **Native and bundled:** inspect every binary and packaged library in the
  *finished* wheel on its target runner. Record imported shared-library names,
  resolved paths or explicit system-provided identities, and hashes of bundled
  files. A source manifest or Linux-only inspection cannot establish the macOS
  and Windows wheels' native closure. The macOS `universal2` wheel contains two
  architecture slices; evidence for one runner architecture does not establish
  the other slice.

The validator must require a complete, nonempty evidence record for each
applicable scope and compare it with the exact distribution and target. Until
target-side producers, immutable transfer and negative tests exist, every
`UNKNOWN` entry continues to refuse admission. The current `uv.lock` and
`requirements/package.txt` pin different NumPy versions; they describe
different environments and must not be silently substituted for each other.
An sdist describes future target builds rather than one installed runtime;
its inventory must distinguish the inspected source/build environment from
platform-dependent consumer resolutions instead of claiming one universal
runtime closure.

The tool semantics behind this split are documented by the
[uv universal-resolution guide](https://docs.astral.sh/uv/concepts/resolution/),
[Cargo metadata reference](https://doc.rust-lang.org/cargo/commands/cargo-metadata.html)
and [auditwheel's binary inspection description](https://github.com/pypa/auditwheel).

Backward compatibility: prior reproducibility artifacts lacking the JSON cannot
be admitted. The trusted control commit must carry the updated helper and
workflow together. The scope HOLD and tag/publish ordering remain. Focused
synthetic tests exercise the actual workflow producer, ID-bound transport,
central-verdict comparison and scope-consumer blocks; they do not certify real
target dependencies. No hosted acceptance is claimed.

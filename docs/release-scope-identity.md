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
The macOS universal2 leg accepts maturin's per-architecture x86_64 and arm64
tags alongside a required universal2 tag; the locally built CPython 3.14 wheel
carried all three. A filename tag alone still does not prove that both binary
slices and their native imports were inspected.
The sdist has a separate source identity, unique PKG-INFO/pyproject members under
one root, and pyproject bytes equal to the release source. These checks never
execute package code. Existing transport hashing remains chunked.

`release-admission` first requires the pinned central gate job to succeed, then
verifies the selected immutable artifact IDs, ZIP and member digests. The
central full-set verdict must match the same run and attempt, exact release
source, all thirteen distribution rows, and the complete Strix binding set. Its
selected artifact also carries the central dependency report. Admission checks
the report hash named by the verdict and requires each installed dependency's
exact wheel archive hash to appear in the licensed, Strix-bound dependency set.
Admission then consumes the JSON from the same selected reproducibility artifact and recomputes
every identity using exact release blobs and transported distribution bytes. Missing,
duplicate, changed, cross-platform or promoted records refuse admission before
the remaining platform-scope HOLD. No name-only fallback is added.

Runtime, build, dev, optional, native and bundled scope entries are explicitly
`UNKNOWN` with null evidence. Complete declarations, an empty dependency array
or an asserted boolean cannot certify any scope. In particular, hashing every
Cargo lock does not establish the features/targets used in compiled wheels;
METADATA hashes do not establish resolved marker or optional dependency closure.
The central licence and Strix verdict has its own authenticated path; it does
not claim platform scope completeness.
The current `requirements/package.txt` pins NumPy 2.5.2 while `uv.lock`
installs NumPy 2.5.1. The central report therefore cannot yet cover that runtime
receipt. Actual target closure evidence must resolve this mismatch before B3
can pass; changing one lock's version string would not establish coverage.

## Evidence needed to remove the scope HOLD

The build jobs now hash every regular member of each finished distribution and
place a bundle inventory beside the TSV in their existing `repro-digest-*`
artifacts. The record job binds the exact thirteen digest-artifact IDs and
archive digests to the source SHA, control SHA, run ID and attempt.
Admission selects those IDs, verifies their archive and member digests, and
compares each TSV row and bundle inventory with the downloaded distribution.
A missing, duplicate, stale, foreign-run or changed digest artifact refuses
release. These evidence
artifacts remain separate from the thirteen distribution artifacts passed to
the central licence and Strix gate.

Each wheel build job now exports its locked runtime and `fuzz` requirements
with pinned uv, downloads their hash-verified binary distributions, and installs
them without network access or a reused cache into an isolated environment for
the runner's Python. It then installs the exact finished wheel and checks the
environment and native extension import. The selected `repro-digest-*` artifact
contains the downloaded dependency wheels, their hashes, package lists,
interpreter identity, requirements hash, `uv.lock` hash and imported extension
hash. Admission rehashes each dependency wheel, checks its metadata against the
installed package list, and binds the extension hash to the published wheel.
A changed, missing or cross-target receipt refuses admission before the scope
HOLD. The macOS universal2 receipt exercises the runner architecture only; it
does not prove installation on its other binary slice. The central gate
prescreens each transported runtime wheel SHA before Strix, scans a distinct
fixture for each approved SHA, and seals the licence report and Strix artifact
identities into its full-set verdict. Admission checks that set against the
twelve installed-wheel receipts. A missing or mismatched platform wheel
remains HOLD.

The sdist and wheel builds still need build and native dependency evidence.
File hashes establish the bundled bytes, but do not identify the origin or
licence of each member or the libraries it loads.
Admission must validate those claims against the corresponding distribution
SHA and build environment before the scope HOLD can be removed.

The six scopes have distinct sources of truth:

- **Runtime and optional Python:** record the target Python interpreter and
  installed or otherwise fully resolved distribution files, versions, markers,
  extras and hashes. Compare them with the wheel's `Requires-Dist` and the
  applicable `uv.lock` resolution. The lock is universal; its mere presence is
  not a target installation. The `fuzz` extra now limits Atheris to its locked
  CPython/Linux x86_64 wheel targets; other targets retain Hypothesis only.
  The target installation receipt exercises this marker on the runner. It is
  not yet promoted to an accepted scope identity.
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

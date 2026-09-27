# Atheris 3.1.0 immutable source lead

Status: HOLD. Technical evidence review, not legal review or a release verdict.

The last setup.py change before the first 3.1.0 PyPI upload was upstream
commit `352d5f29d811e0b51807ca57ed5551dd08ece528`, committed
2026-06-16T20:20:40Z with the message identifying the 3.1.0 version bump.
The three wheel uploads began 2026-06-17T00:04:01Z. Temporal proximity is
not proof that these wheels were built from that source.

Neither `3.1.0` nor `v3.1.0` resolves as a tag; the retrieved GitHub release
list was empty. The exact commit archive has SHA256
`df444341ce94a116757b9f9a56c174a5a9f0d4285f07242451aa0acbe641262b`.
Its recipe and related files are retained with URL and hashes in SOURCES.md.

The deployment Dockerfile pins LLVM commit
`0982db188b661d6744b06244fda64d43dd80206e`, builds compiler-rt beneath
`/root/llvm-project`, and selects its libFuzzer archive. That path is
consistent with the published archive's previously inspected DWARF path.
However, its wheel commands cover cp311/cp312/cp313, whereas the actual
3.1.0 release contains cp312/cp313/cp314. The CI matrix also lacks 3.14.
The manylinux base image is unpinned. This recipe therefore cannot establish
the exact published binary build or account for all three release artifacts.

The complete LLVM and compiler-rt license files at that LLVM commit were
read and preserved. They contain Apache 2.0 with LLVM exceptions and legacy
NCSA terms; compiler-rt also includes legacy MIT terms. Both flag separately
licensed third-party code. Their presence at a source lead does not establish
which source files contributed to the wheel's four native objects.

Next evidence requirement: an authenticated release build record binding the
three exact wheel hashes to source/toolchain inputs, or reproduced matching
native bytes with complete input and component-notice provenance. Do not add
these lead hashes to the verifier's accepted wheel/native fixtures yet.

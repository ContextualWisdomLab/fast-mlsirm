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

## Build-path and archive follow-up

The complete pinned setup.py was read. `get_libfuzzer_lib` accepts an
environment-selected archive; `BuildExt.build_extensions` can upgrade it
before copying it into the wheel. The sanitizer siblings are combined with
the selected libFuzzer using a caller-selected C++ linker (otherwise Clang
or GCC). Those tool and archive choices are additional reproducibility inputs.

The exact cp312 wheel and native archive were rehashed on s1 before `ar t`
and `nm -g` inspection. They match the previously recorded wheel hash
`ec5e11f21a4c197fe91f7aea2b2de88e623c73a21fc07b105ac6329a1588457b`
and archive hash
`60d06f6748c007c46c772b0abee959053973ee4b607a88f9f3db3562b2bbecaf`.
The archive has 24 members (retained in wheel-libfuzzer-members.txt), no
wrapper-named or temporary-named member, and defines LLVMFuzzerRunDriver.
The pinned version-check script therefore classifies it as up-to-date;
this does not require executing untrusted setup.py or its shell scripts.
This symbol also exists in the pinned LLVM source, so it is consistent with
that source lead but does not distinguish it from later LLVM versions.
No native license acceptance follows from this compatibility check.

## Public CI receipt

The exact source commit has one public Builds push run,
[27645691616](https://github.com/google/atheris/actions/runs/27645691616).
Job 81756860359 failed at Run pytype checks; build job 81757708148 was
skipped. Its artifacts endpoint returned total_count 0 at inspection on
2026-09-27 UTC. The other two runs on this commit are dependency graph
automation, not wheel publication. This public CI execution is therefore
not the missing authenticated publisher record for the three PyPI wheels.

# Windows Fortran-free runtime candidate (#2136)

Status: **HOLD**. This is a DLL-only source and link plan, not a NumPy wheel or
an approved runtime. It applies to the Draft candidate workflow in #2155 and
the pinned OpenBLAS source commit `446c436e10450c348169808f1a6b3fae0925c9f7`.

## Source and link contract

1. Fetch that exact OpenBLAS commit and record its tree and license hashes.
   Configure x64 Visual Studio tools with `clang-cl`, `NOFORTRAN=1`,
   `C_LAPACK=ON`, `USE_OPENMP=OFF`, and `BUILD_WITHOUT_LAPACK=OFF`. This selects
   the C-translated LAPACK path without a Fortran compiler; it does not by
   itself prove the absence of GCC runtime code in the resulting DLL.
2. Keep `INTERFACE64=1`, `DYNAMIC_ARCH=ON`, `TARGET=PRESCOTT`, and the
   `scipy_`/`64_` symbol and library naming as candidate ABI settings. The
   exported symbols and NumPy compatibility still need native evidence.
3. On Windows, the pinned `CMakeLists.txt` forces a temporary static archive
   when symbol renaming is requested and runs `perl` plus **`lld-link`** in a
   post-build command to produce the DLL. That command reads
   `CMAKE_LINKER_FLAGS`, not `CMAKE_SHARED_LINKER_FLAGS` or `CMAKE_LINKER`.
   The candidate workflow therefore checks for Perl, records `lld-link`, and
   supplies `/MAP` and `/VERBOSE` as separate CMake list arguments through
   `CMAKE_LINKER_FLAGS`. A local Ninja generation probe confirmed that a
   space-separated value becomes one quoted argument, while a semicolon list
   becomes two linker arguments. A missing map or
   loaded-member trace fails the existing verifier; a successful configure
   alone is insufficient.

The source behavior above is from the [pinned OpenBLAS CMake file](https://github.com/OpenMathLib/OpenBLAS/blob/446c436e10450c348169808f1a6b3fae0925c9f7/CMakeLists.txt)
(Windows symbol-renaming condition and post-build command). The [OpenBLAS
Windows guide](https://github.com/OpenMathLib/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio)
documents the CMake/clang-cl DLL route. LLVM's [COFF linker options](https://github.com/llvm/llvm-project/blob/main/lld/COFF/Options.td)
include `/map` and `/verbose`; hosted output must still confirm the actual
archive-member format and completeness.

## Existing paths and remaining acceptance

The current default branch has no DLL-only Windows workflow. Its
`publish-pypi.yml` has Windows wheel jobs and publication credentials, while
`release-tag.yml` and `pypi-gap-guard.yml` can initiate publication. Reusing
those for this candidate would cross the no-publish boundary. #2155 is the
repository-owned candidate path, but its new `workflow_dispatch` file cannot
run until admitted to the default branch through ordinary review and checks.

One hosted Windows run bound to an exact reviewed head must retain the DLL,
configure/link logs, map, import/export tables, source/tree/toolchain hashes,
NOTICE, SBOM, and verified attestation. Check direct and delayed PE imports,
transitive DLL closure, every linked archive member, x64 machine type, ILP64
symbol exports, and NumPy ABI plus numerical parity before any wheel work.
Fail on any `gfortran`, `quadmath`, `gcc`, `winpthread`, or `stdc++` import or
linked member. The current verifier covers direct imports and named linker/map
evidence; it does not establish the entire transitive runtime closure or
NumPy parity.

License review must bind the bytes and terms of every retained MSVC/LLVM
runtime and archive component. The pinned OpenBLAS/LAPACK notices alone do not
clear those components. No hosted DLL, PE closure, runtime-license clearance,
or GPL/LGPL/AGPL-free NumPy wheel has been produced by this plan.

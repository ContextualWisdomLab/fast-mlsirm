# NumPy cp312 Linux build-backend archive evidence

The six existing hash-pinned backend wheels were copied read-only from the s1 candidate audit_A/build-deps directory. Each wheel hash matches both the original lock and the exact official PyPI version/file record. The original artifacts and lock were not changed. No fresh compilation was performed.

The receipt records every dist-info license file and actual ELF magic, including Ninja's extensionless executable (one ELF) and Cython's 19 extension modules. Pure-Python package names alone do not prove absence of embedded native tools.

Four primary grants match the existing repository standard-text recognizer. Meson-python has a complete permissive MIT-style grant with an additional retention parenthesis; it is recorded as a manually read variant, not promoted to canonical MIT. Ninja's complete Apache terms match the source-companion Apache body after only whitespace, HTTP/HTTPS and section-4 list labels are normalized; its additional appendix is preserved. These are primary-text observations, not complete archive rights verdicts.

Cython wheel has no license files. Its official same-version sdist SHA-256 is eed0d93fbca7087f143b42c34b05a825849bdf17f101572c2105acfa49aa88b8. COPYING.txt and LICENSE.txt were obtained from that exact archive. All 312 wheel Python/C/header/declaration source members match the sdist byte-for-byte. The 19 compiled ELF members are not thereby provenance-bound. The source grant explicitly addresses compiled output and embedded snippets; see the complete companion COPYING.txt.

Sources: https://pypi.org/pypi/Cython/3.3.0/json and corresponding versioned PyPI JSON records for the other five packages; exact publisher archive URLs and hashes are in backend-archive-receipt.json. Primary license files are reproduced verbatim with their copyright notices.

Remaining work: bind backend native build provenance and bundled component grants; perform a fresh hash-locked build in the already pinned manylinux image, with tool/environment/native provider receipts and numeric/ABI checks. This evidence does not clear the original Python3/Cargo11 HOLDs or prove the published12 matrix.

## Same-version pure Cython alternative

Official cython-3.3.0-py3-none-any.whl SHA-256 9b24b5c8cd536946b62086fcafee6d5509d3f549f72d553d2336af87ffbe0da1 contains no ELF members. Its 312 source members match the official sdist. A separate six-package hash lock changes only Cython artifact hash, retaining all versions. This is prepared evidence, not an installed/build-tested selection; original lock and prior builds are unchanged. A fresh build must use the alternative lock explicitly if selected and validate generated code and numerical results.

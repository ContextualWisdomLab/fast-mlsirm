# Automatic backend resolution must preserve one numerical owner

## Decision under test

`backend="auto"` is a convenience selector for the production Rust/PyO3 numerical implementation. It is not permission to switch to an independent NumPy implementation when the compiled extension is unavailable. A missing compiled core therefore fails closed before psychometric numerical work begins.

The explicit `backend="numpy"` surface is retained in this bounded migration as a reference/parity choice. It is never selected implicitly by automatic production resolution. Rust device selection remains a different axis: a Rust GPU request may fall back to the parity-verified Rust CPU implementation because the numerical owner and formula contract remain Rust-owned.

## Failure boundary

Python's import machinery provides a direct capability probe: `importlib.util.find_spec()` returns `None` when no module specification is found and importing the module is a separate operation. The package uses that boundary to distinguish an unavailable extension from an available compiled module. The PyO3 and maturin primary documentation describe the native extension as the Python-importable compiled module produced and distributed with the package. Those mechanics support a fail-closed contract when the package's required production extension is missing; they do not justify substituting a different numerical implementation.

The package exposes a stable, non-reflective error for an unavailable automatic production backend rather than reflecting local paths, ABI details, environment data, or import exception text into the fallback decision. Actual import/ABI failures remain diagnostic failures rather than being reclassified as permission to run NumPy arithmetic.

Backend and Rust-device names are control-plane values, not general string-convertible data. Calling `str()` or overridden string methods on an arbitrary object can execute caller-defined Python before the allowlist decision. The public normalizers therefore accept exact built-in `str` instances only, then perform package-owned whitespace/case normalization and positive allowlist validation. Non-strings and `str` subclasses fail before conversion, native-core discovery, or any caller callback. This is a type-validation boundary consistent with CWE-1287 and OWASP ASVS 5.0 input-validation guidance; it changes no numerical formula or backend ownership.

NIST SP 800-218 Version 1.1 remains the final SSDF authority. NIST SP 800-218 Rev. 1 Version 1.2 is an Initial Public Draft as of this doctoring update and is tracked as newer draft guidance rather than represented as final.

## Falsification and acceptance

This decision is falsified if any ordinary `backend="auto"` call can select NumPy because the Rust extension is missing or incompatible. It is also falsified if backend/device validation executes caller-defined string-conversion or normalization callbacks before rejecting an invalid control value. Acceptance requires tests proving that automatic resolution selects Rust when available, fails closed when absent, explicit NumPy resolution remains an explicit caller decision, and malformed control objects fail before native discovery. Installed-wheel/package evidence must continue to prove the Rust extension is present in supported production artifacts.

## References

MITRE. (2026, April 30). *CWE-1287: Improper validation of specified type of input (Version 4.20)*. Common Weakness Enumeration. https://cwe.mitre.org/data/definitions/1287.html

Open Worldwide Application Security Project. (n.d.). *Application Security Verification Standard 5.0: V2.2 input validation*. Retrieved August 14, 2026, from https://cornucopia.owasp.org/taxonomy/asvs-5.0/02-validation-and-business-logic/02-input-validation

Souppaya, M., Scarfone, K., & Dodson, D. (2022). *Secure Software Development Framework (SSDF) Version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218). National Institute of Standards and Technology. https://doi.org/10.6028/NIST.SP.800-218

Booth, H., Ogata, M., Kent, K., Souppaya, M., & Dodson, D. (2025). *Secure Software Development Framework (SSDF) Version 1.2: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218 Rev. 1, Initial Public Draft). National Institute of Standards and Technology. https://doi.org/10.6028/NIST.SP.800-218r1.ipd

Python Software Foundation. (n.d.). *importlib — The implementation of import*. Python 3.14.6 documentation. Retrieved August 10, 2026, from https://docs.python.org/3/library/importlib.html

PyO3 Project. (n.d.). *Building and distribution*. PyO3 user guide. Retrieved August 10, 2026, from https://pyo3.rs/main/building-and-distribution

Maturin Project. (n.d.). *Introduction*. Maturin user guide. Retrieved August 10, 2026, from https://www.maturin.rs/

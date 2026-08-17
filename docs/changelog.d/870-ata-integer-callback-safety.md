# ATA integer callback safety

## Fixed

- Automated test assembly now admits only exact built-in integers and explicitly supported genuine NumPy integer scalar identities for public length, seed, exposure, content-count, and exclusion controls before normalization.
- Caller-defined Python and NumPy integer subclasses fail closed before conversion callbacks or item-information work, while existing finite-domain validation and genuine NumPy scalar compatibility are preserved.
- Added focused public-boundary regressions for hostile scalar and container controls without changing ATA information, selection, or scoring arithmetic.

# ATA content-label validation trust boundary

## Fixed

- Validate automated-test-assembly content-label shape and string element types before item-information evaluation, rejecting arbitrary object labels without invoking caller-controlled `__str__`/`__repr__` callbacks while preserving accepted Python/NumPy string labels and existing assembly numerics.

## Security

- Keep invalid ATA content controls on a stable package-owned error surface rather than allowing arbitrary representation callbacks to execute during NumPy string coercion.

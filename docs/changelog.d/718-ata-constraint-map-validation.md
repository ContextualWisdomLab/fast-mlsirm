# ATA constraint-map validation trust boundary

## Fixed

- Validate ATA content-constraint keys/counts, exposure maps, seed, and exposure_max as admitted types before item-information evaluation, rejecting hostile string/integer conversion callbacks while preserving accepted Python/NumPy string keys and exact integers.

## Security

- Keep invalid ATA semantic controls on a stable package-owned error surface rather than allowing arbitrary `__str__`/`__int__`/`__index__` callbacks during constraint-map coercion.

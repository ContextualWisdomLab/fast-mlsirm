"""Exact integer seed admission shared by model and bootstrap callers."""
import numpy as np

_INTEGER_TYPES = (
    np.int8, np.int16, np.int32, np.int64, np.intp, np.longlong,
    np.uint8, np.uint16, np.uint32, np.uint64, np.uintp, np.ulonglong,
)


def _u64_seed(value: object, *, name: str = "seed") -> int:
    """Preserve trusted Python/NumPy integer seeds in the Rust u64 range.

    Python Software Foundation, Built-in Types, Numeric Types:
    https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex
    Python integers have unlimited precision; conversion through float can
    lose their bits. NumPy Developers, Scalars, Integer types:
    https://numpy.org/doc/stable/reference/arrays.scalars.html#integer-types
    NumPy integers are fixed-width scalars. Their exact scalar types are
    admitted before conversion to a Python integer. Exact type admission
    reuses the GRM wrapper's existing boundary and avoids subclass callbacks.
    Floats, bools, strings, arrays and caller-defined integer subclasses are
    rejected; the u64 bound is the existing Rust API representation contract,
    not a scientifically recommended seed or random-stream independence rule.
    """
    value_type = type(value)
    if value_type is int:
        seed = value
    elif any(value_type is scalar_type for scalar_type in _INTEGER_TYPES):
        seed = int(value)
    else:
        raise ValueError(f"{name} must be a non-negative integer")
    if not 0 <= seed < 2**64:
        raise ValueError(f"{name} must be in [0, 2**64)")
    return seed

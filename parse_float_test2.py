import json

def parse_float_valid(v):
    import math
    f = float(v)
    if not math.isfinite(f):
        raise ValueError("non-finite")
    return f

print(json.loads('{"a": 1e999}', parse_float=parse_float_valid))

import json
import math

def parse_float_valid(v):
    f = float(v)
    if not math.isfinite(f):
        raise ValueError("non-finite")
    return f

print(json.loads('{"a": 1.0}', parse_float=parse_float_valid))

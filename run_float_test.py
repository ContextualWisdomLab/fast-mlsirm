import json
import math

content = '{"key": 1e999}'

def test_no_float_hook():
    try:
        data = json.loads(content, parse_constant=lambda x: x)
        print("Without parse_float:", data)
    except Exception as e:
        print("Error:", e)

def _reject_float_nonfinite(value):
    f_val = float(value)
    if not math.isfinite(f_val):
        raise ValueError("contract_json contains non-finite numbers")
    return f_val

def test_with_float_hook():
    try:
        data = json.loads(content, parse_constant=lambda x: x, parse_float=_reject_float_nonfinite)
        print("With parse_float:", data)
    except Exception as e:
        print("Error:", e)

test_no_float_hook()
test_with_float_hook()

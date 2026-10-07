import re

def process(file, old, new):
    with open(file, 'r') as f:
        content = f.read()
    content = content.replace(old, new)
    with open(file, 'w') as f:
        f.write(content)

io_old = """    kwargs = {
        "parse_constant": (
            reject_nonfinite_constant if parse_constant is None else parse_constant
        ),
        "parse_float": lambda v: float(v) if __import__('math').isfinite(float(v)) else (_ for _ in ()).throw(ValueError(f"{source} contains a non-finite JSON numeric value")),
        "object_pairs_hook": reject_duplicate_members,
    }
    return json.loads(content, **kwargs)"""
io_new = """    def reject_nonfinite_float(value: str) -> float:
        import math
        f_val = float(value)
        if not math.isfinite(f_val):
            raise ValueError(f"{source} contains a non-finite JSON numeric value")
        return f_val

    kwargs = {
        "parse_constant": (
            reject_nonfinite_constant if parse_constant is None else parse_constant
        ),
        "parse_float": reject_nonfinite_float,
        "object_pairs_hook": reject_duplicate_members,
    }
    return json.loads(content, **kwargs)"""
process("python/fast_mlsirm/io.py", io_old, io_new)

ce_old1 = """def _reject_json_constant(value: str) -> object:
    \"\"\"Reject non-finite JSON extensions unsupported by the manifest contract.\"\"\"
    raise ValueError(f"manifest JSON contains unsupported constant: {value}")"""
ce_new1 = """def _reject_json_constant(value: str) -> object:
    \"\"\"Reject non-finite JSON extensions unsupported by the manifest contract.\"\"\"
    raise ValueError(f"manifest JSON contains unsupported constant: {value}")


def _reject_json_float(value: str) -> float:
    import math
    f_val = float(value)
    if not math.isfinite(f_val):
        raise ValueError("manifest JSON contains non-finite float")
    return f_val"""
process("python/fast_mlsirm/cross_engine_conformance.py", ce_old1, ce_new1)

ce_old2 = """            parsed = json.loads(
                value,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_json_constant,
                parse_float=lambda v: float(v) if __import__('math').isfinite(float(v)) else (_ for _ in ()).throw(ValueError("manifest JSON contains non-finite float")),
            )"""
ce_new2 = """            parsed = json.loads(
                value,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_json_constant,
                parse_float=_reject_json_float,
            )"""
process("python/fast_mlsirm/cross_engine_conformance.py", ce_old2, ce_new2)

rc_old1 = """def _reject_nonfinite(_value: str) -> None:
    \"\"\"Reject NaN and infinity tokens accepted by Python's permissive decoder.\"\"\"
    raise _NonFiniteJsonNumber"""
rc_new1 = """def _reject_nonfinite(_value: str) -> None:
    \"\"\"Reject NaN and infinity tokens accepted by Python's permissive decoder.\"\"\"
    raise _NonFiniteJsonNumber


def _reject_float_nonfinite(value: str) -> float:
    import math
    f_val = float(value)
    if not math.isfinite(f_val):
        raise _NonFiniteJsonNumber
    return f_val"""
process("python/fast_mlsirm/rubric/candidates.py", rc_old1, rc_new1)

rc_old2 = """        decoded = json.loads(
            raw_json,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
            parse_float=lambda v: float(v) if __import__('math').isfinite(float(v)) else (_ for _ in ()).throw(_NonFiniteJsonNumber()),
        )"""
rc_new2 = """        decoded = json.loads(
            raw_json,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
            parse_float=_reject_float_nonfinite,
        )"""
process("python/fast_mlsirm/rubric/candidates.py", rc_old2, rc_new2)

lj_old1 = """def _duplicate_free_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, member in pairs:
        if key in value:
            raise _DuplicateJsonKeyError(key)
        value[key] = member
    return value"""
lj_new1 = """def _duplicate_free_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, member in pairs:
        if key in value:
            raise _DuplicateJsonKeyError(key)
        value[key] = member
    return value


def _reject_float_nonfinite(value: str) -> float:
    import math
    f_val = float(value)
    if not math.isfinite(f_val):
        raise JudgeFormatError("judge response JSON contains non-finite float")
    return f_val


def _reject_constant(_value: str) -> float:
    raise JudgeFormatError("judge response JSON contains unsupported constant")"""
process("python/fast_mlsirm/llm_judge.py", lj_old1, lj_new1)

lj_old2 = """        value = json.loads(
            text,
            object_pairs_hook=_duplicate_free_object,
            parse_constant=lambda v: (_ for _ in ()).throw(JudgeFormatError("judge response JSON contains unsupported constant")),
            parse_float=lambda v: float(v) if __import__('math').isfinite(float(v)) else (_ for _ in ()).throw(JudgeFormatError("judge response JSON contains non-finite float")),
        )"""
lj_new2 = """        value = json.loads(
            text,
            object_pairs_hook=_duplicate_free_object,
            parse_constant=_reject_constant,
            parse_float=_reject_float_nonfinite,
        )"""
process("python/fast_mlsirm/llm_judge.py", lj_old2, lj_new2)

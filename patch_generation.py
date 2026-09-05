import re

with open("python/fast_mlsirm/rubric/generation.py", "r") as f:
    content = f.read()

search = """def _contract_object(contract_json: str) -> dict[str, Any]:
    \"\"\"Parse canonical contract JSON and require a top-level object.\"\"\"
    if type(contract_json) is not str or not contract_json:
        raise ValueError("contract_json must be non-empty JSON text")
    _validate_contract_depth(contract_json)
    try:
        contract = json.loads(contract_json)
    except (TypeError, ValueError) as exc:
        raise ValueError("contract_json must be valid JSON text") from exc"""

replace = """def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    \"\"\"Reject duplicate JSON object keys to prevent JSON smuggling.\"\"\"
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(_literal: str) -> Any:
    \"\"\"Reject non-finite JSON constants.\"\"\"
    raise ValueError("contract_json contains a non-finite numeric value")


def _contract_object(contract_json: str) -> dict[str, Any]:
    \"\"\"Parse canonical contract JSON and require a top-level object.\"\"\"
    if type(contract_json) is not str or not contract_json:
        raise ValueError("contract_json must be non-empty JSON text")
    _validate_contract_depth(contract_json)
    try:
        contract = json.loads(
            contract_json,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("contract_json must be valid JSON text") from exc"""

if search in content:
    content = content.replace(search, replace)
    with open("python/fast_mlsirm/rubric/generation.py", "w") as f:
        f.write(content)
    print("Patched generation successfully")
else:
    print("Could not find search block in generation")

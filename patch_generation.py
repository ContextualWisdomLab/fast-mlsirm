import re

with open("python/fast_mlsirm/rubric/generation.py", "r") as f:
    content = f.read()

search = """def _reject_nonfinite_json(_literal: str) -> Any:
    \"\"\"Reject non-finite JSON constants.\"\"\"
    raise ValueError("contract_json contains a non-finite numeric value")"""

replace = """def _reject_nonfinite_json(literal: str) -> Any:
    \"\"\"Reject non-finite JSON constants.\"\"\"
    raise ValueError("contract_json contains a non-finite numeric value")"""

if search in content:
    content = content.replace(search, replace)
    with open("python/fast_mlsirm/rubric/generation.py", "w") as f:
        f.write(content)
    print("Patched generation successfully")
else:
    print("Could not find search block in generation")

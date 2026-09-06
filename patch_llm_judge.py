import re

with open("python/fast_mlsirm/llm_judge.py", "r") as f:
    content = f.read()

search = """def _reject_nonfinite(_literal: str) -> float:
    raise JudgeFormatError("judge response contains non-finite numeric value")"""

replace = """def _reject_nonfinite(literal: str) -> float:
    raise JudgeFormatError("judge response contains non-finite numeric value")"""

if search in content:
    content = content.replace(search, replace)
    with open("python/fast_mlsirm/llm_judge.py", "w") as f:
        f.write(content)
    print("Patched llm_judge successfully")
else:
    print("Could not find search block in llm_judge")

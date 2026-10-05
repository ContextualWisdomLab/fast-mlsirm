import os
import glob

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # We want to replace
    #         elif char in "]}":
    #             depth -= 1
    # with
    #         elif char in "]}":
    #             if depth > 0:
    #                 depth -= 1

    if '        elif char in "]}":\n            depth -= 1' in content:
        content = content.replace(
            '        elif char in "]}":\n            depth -= 1',
            '        elif char in "]}":\n            if depth > 0:\n                depth -= 1'
        )
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Fixed {filepath}")

for filepath in [
    "python/fast_mlsirm/rubric/candidates.py",
    "python/fast_mlsirm/rubric/generation.py",
    "python/fast_mlsirm/cross_engine_conformance.py",
    "python/fast_mlsirm/llm_judge.py",
    "python/fast_mlsirm/io.py",
]:
    fix_file(filepath)

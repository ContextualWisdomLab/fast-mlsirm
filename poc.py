import json
import sys

sys.setrecursionlimit(100000)
data = "{}"
for _ in range(50000):
    data = f'{{"a": {data}}}'

with open("deep2.json", "w") as f:
    f.write(data)

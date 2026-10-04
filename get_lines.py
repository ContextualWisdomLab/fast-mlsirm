with open("python/fast_mlsirm/report.py") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "<style>" in line:
            print(f"Found style tag at line {i+1}")

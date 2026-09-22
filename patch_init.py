with open("python/fast_mlsirm/__init__.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if line.strip() == '"detect_dif_logistic_purified",':
        new_lines.append(line)
        new_lines.append('    "detect_dif_mantel_haenszel",\n')
        new_lines.append('    "detect_dif_mantel_smd",\n')
        new_lines.append('    "detect_dif_gmh",\n')
        new_lines.append('    "detect_dif_breslow_day",\n')
        skip = True
        continue
    if skip and line.strip() == '):':
        skip = False
        new_lines.append(line)
        new_lines.append('    if hasattr(_legacy_init, _dif_name):\n')
        new_lines.append('        setattr(_legacy_init, _dif_name, getattr(_dif, _dif_name))\n')
        continue
    if skip:
        continue

    new_lines.append(line)

with open("python/fast_mlsirm/__init__.py", "w") as f:
    f.writelines(new_lines)

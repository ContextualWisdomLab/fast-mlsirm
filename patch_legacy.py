with open("python/fast_mlsirm/_legacy_init.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if line.strip() == '"detect_dif_mantel_haenszel_purified",':
        skip = True
        new_lines.append(line)
        new_lines.append('    "detect_dif_logistic_purified",\n')
        new_lines.append('    "detect_dif_mantel_haenszel",\n')
        new_lines.append('    "detect_dif_mantel_smd",\n')
        new_lines.append('    "detect_dif_gmh",\n')
        new_lines.append('    "detect_dif_breslow_day",\n')
        continue

    if skip and line.strip() == '"detect_dif_logistic_purified",':
        continue
    if skip and line.strip() == '"score_wle",':
        skip = False

    new_lines.append(line)

with open("python/fast_mlsirm/_legacy_init.py", "w") as f:
    f.writelines(new_lines)

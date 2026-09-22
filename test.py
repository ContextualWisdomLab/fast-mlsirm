import re
with open("python/fast_mlsirm/__init__.py", "r") as f:
    text = f.read()

# Replace any occurrence of the loop that has missing closing bracket with the correct one
correct_loop = """for _dif_name in (
    "detect_dif_logistic",
    "detect_dif_mantel_haenszel_purified",
    "detect_dif_logistic_purified",
    "detect_dif_mantel_haenszel",
    "detect_dif_mantel_smd",
    "detect_dif_gmh",
    "detect_dif_breslow_day",
):"""
text = re.sub(r'for _dif_name in \([\s\S]*?"detect_dif_breslow_day",\n(?!:\n)', correct_loop + '\n', text)
with open("python/fast_mlsirm/__init__.py", "w") as f:
    f.write(text)

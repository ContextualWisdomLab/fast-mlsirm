with open("tests/test_report_security.py", "r") as f:
    content = f.read()

# Replace the overly strict regex with one that handles HTML escaping (&#x27;)
content = content.replace(
    r"csp_match = re.search(r\"style-src 'sha256-([^']+)'\", html)",
    r"csp_match = re.search(r\"style-src (?:'|&#x27;)sha256-([^'&#]+)(?:'|&#x27;)\", html)"
)

with open("tests/test_report_security.py", "w") as f:
    f.write(content)

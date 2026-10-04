import re
from fast_mlsirm.report import render_diagnostics_report
import json

diag_path = "diag.json"
with open(diag_path, "w") as f:
    f.write(json.dumps({
        "metadata": {
            "version": "1",
            "run_id": "test",
            "engine": "test"
        },
        "model_fit": {
            "loglik": -1.0,
            "aic": 1.0,
            "bic": 1.0,
            "aicc": 1.0,
            "deviance": 1.0,
            "df": 1
        },
        "summary": {
            "person_count": 1,
            "item_count": 1,
            "response_count": 1
        },
        "metrics": {},
        "exact_values": {}
    }))

render_diagnostics_report(diag_path, "out.html")

with open("out.html", "r") as f:
    html = f.read()

print("HTML CSP tag:")
csp_tag_match = re.search(r'<meta http-equiv="Content-Security-Policy".*?>', html)
if csp_tag_match:
    print(csp_tag_match.group(0))

print("Is sha256 in tag?", "sha256" in csp_tag_match.group(0) if csp_tag_match else "No tag")

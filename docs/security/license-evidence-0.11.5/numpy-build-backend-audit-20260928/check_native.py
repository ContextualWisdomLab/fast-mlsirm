import importlib.metadata,json,subprocess,re
from pathlib import Path
rows=[]
for member in importlib.metadata.distribution("numpy").files:
 p=Path(importlib.metadata.distribution("numpy").locate_file(member))
 if not p.is_file(): continue
 with p.open("rb") as f: magic=f.read(4)
 if magic != b"\x7fELF": continue
 result=subprocess.run(["readelf","-d",str(p)],check=True,text=True,capture_output=True,timeout=30)
 needed=re.findall(r"\(NEEDED\).*?\[(.*?)\]",result.stdout)
 assert not any("gfortran" in n or "quadmath" in n for n in needed),str(member)
 rows.append({"member":str(member),"needed":needed})
assert len(rows)==20
print(json.dumps({"status":"static DT_NEEDED audit only","elf_count":len(rows),"no_gfortran_or_quadmath_needed":True,"members":rows},indent=2))

"""Repair this hash-bound local candidate; never modify or publish its input."""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parent
source = next((root / "candidate-input").glob("*.whl"))
notice = (root / "OPENBLAS-0.3.34-NOTICES.txt").read_bytes()
assert hashlib.sha256(source.read_bytes()).hexdigest() == "c6fbea224227ebfcb80d694be15f74c72a06c901833f3f073df3405ade4e07a9"
assert hashlib.sha256(notice).hexdigest() == "9f21f7061f26cdc6f173c29a5a2754c68326d7b397c6bc75bfb4f7543ed21ba4"


def verify_record(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        assert len(names) == len(set(names))
        assert not any(n.endswith(("RECORD.jws", "RECORD.p7s")) for n in names)
        record = next(n for n in names if n.endswith(".dist-info/RECORD"))
        rows = list(csv.reader(io.StringIO(z.read(record).decode())))
        assert len(rows) == len({row[0] for row in rows})
        assert {row[0] for row in rows} == {n for n in names if not n.endswith("/")}
        for name, digest, size in rows:
            if name == record:
                assert (digest, size) == ("", "")
            else:
                data = z.read(name)
                actual = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                assert (digest, size) == ("sha256=" + actual, str(len(data))), name


verify_record(source)
with zipfile.ZipFile(source) as z:
    entries = {i.filename: (i, z.read(i)) for i in z.infolist()}
metadata = next(n for n in entries if n.endswith(".dist-info/METADATA"))
record = next(n for n in entries if n.endswith(".dist-info/RECORD"))
notice_name = metadata.rsplit("/", 1)[0] + "/licenses/OPENBLAS-0.3.34-NOTICES.txt"
assert notice_name not in entries
header, separator, body = entries[metadata][1].partition(b"\n\n")
assert separator
entries[metadata] = (entries[metadata][0], header + b"\nLicense-File: OPENBLAS-0.3.34-NOTICES.txt\n\n" + body)
info = zipfile.ZipInfo(notice_name, entries[metadata][0].date_time)
info.compress_type = zipfile.ZIP_DEFLATED
info.external_attr = 0o100644 << 16
entries[notice_name] = (info, notice)
rows = []
for name, (_, data) in sorted(entries.items()):
    if not name.endswith("/") and name != record:
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append((name, "sha256=" + digest, str(len(data))))
rows.append((record, "", ""))
output = io.StringIO(newline="")
csv.writer(output, lineterminator="\n").writerows(rows)
entries[record] = (entries[record][0], output.getvalue().encode())
paths = []
for label in ("A", "B"):
    folder = root / label
    folder.mkdir(exist_ok=True)
    path = folder / source.name
    with zipfile.ZipFile(path, "x") as z:
        for _, (info, data) in sorted(entries.items()):
            z.writestr(info, data)
    verify_record(path)
    with zipfile.ZipFile(source) as before, zipfile.ZipFile(path) as after:
        assert set(after.namelist()) - set(before.namelist()) == {notice_name}
        assert all(before.read(n) == after.read(n) for n in before.namelist() if n not in (metadata, record))
    paths.append(path)
assert paths[0].read_bytes() == paths[1].read_bytes()
receipt = {"status": "local notice repair only; release HOLD", "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
           "output_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(), "notice_sha256": hashlib.sha256(notice).hexdigest(),
           "record_verified": True, "repeat_repack_identical": True, "unchanged_native_and_source_members": True,
           "changed_existing_members": [metadata, record], "added_member": notice_name}
(root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))

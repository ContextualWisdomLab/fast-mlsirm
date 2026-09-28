"""Make a local notice-complete pilot from the exact PyPI wheel."""

import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent
NAME = "sortedcontainers-2.4.0-py2.py3-none-any.whl"
SOURCE_SHA = "a163dcaede0f1c021485e957a39245190e74249897e2ae4b2aa38595db237ee0"
TERMS_SHA = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
ADDED = "sortedcontainers-2.4.0.dist-info/LICENSE-APACHE-2.0.txt"
RECORD = "sortedcontainers-2.4.0.dist-info/RECORD"


def digest(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def verify_record(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)) and archive.testzip() is None
        rows = list(csv.reader(io.StringIO(archive.read(RECORD).decode())))
        assert {row[0] for row in rows} == set(names)
        for name, hash_value, size in rows:
            if name == RECORD:
                assert (hash_value, size) == ("", "")
            else:
                data = archive.read(name)
                assert (hash_value, size) == (f"sha256={digest(data)}", str(len(data)))


source = ROOT / NAME
terms = (ROOT / "LICENSE-2.0.txt").read_bytes()
assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA
assert hashlib.sha256(terms).hexdigest() == TERMS_SHA
verify_record(source)

with zipfile.ZipFile(source) as archive:
    entries = {info.filename: (info, archive.read(info)) for info in archive.infolist()}
assert ADDED not in entries
info = zipfile.ZipInfo(ADDED, entries[RECORD][0].date_time)
info.compress_type = zipfile.ZIP_DEFLATED
info.external_attr = 0o100644 << 16
entries[ADDED] = (info, terms)

rows = [(name, f"sha256={digest(data)}", str(len(data)))
        for name, (_, data) in sorted(entries.items()) if name != RECORD]
rows.append((RECORD, "", ""))
output = io.StringIO(newline="")
csv.writer(output, lineterminator="\n").writerows(rows)
entries[RECORD] = (entries[RECORD][0], output.getvalue().encode())

paths = []
for label in ("A", "B"):
    folder = ROOT / label
    folder.mkdir(exist_ok=True)
    path = folder / NAME
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for _, (member_info, data) in sorted(entries.items()):
            archive.writestr(member_info, data)
    if path.exists():
        assert path.read_bytes() == buffer.getvalue()
    else:
        path.write_bytes(buffer.getvalue())
    verify_record(path)
    with zipfile.ZipFile(source) as before, zipfile.ZipFile(path) as after:
        assert set(after.namelist()) - set(before.namelist()) == {ADDED}
        assert all(before.read(name) == after.read(name)
                   for name in before.namelist() if name != RECORD)
    paths.append(path)

assert paths[0].read_bytes() == paths[1].read_bytes()
receipt = {
    "status": "local notice pilot only; original Python HOLD remains",
    "source_sha256": SOURCE_SHA,
    "apache_terms_sha256": TERMS_SHA,
    "output_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
    "added_member": ADDED,
    "changed_existing_members": [RECORD],
    "all_original_package_and_metadata_members_identical": True,
    "record_verified": True,
    "repeat_repack_identical": True,
}
(ROOT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))

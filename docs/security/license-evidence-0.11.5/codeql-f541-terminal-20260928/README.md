# Exact f541 main CodeQL canary

Run `36346059984`, attempt 1, was dispatched from fast-mlsirm main
`f54143b968c0f332fa8d49ed3c116fc7d51891b0` and completed successfully.
The Actions job `108695318060` ran on `cwlab-s1-01` from 03:54:46 to
03:56:02 UTC; the Python job `108695318220` ran on `cwlab-s1-03` from
03:56:02 to 04:02:35 UTC, on 2026-09-28. Both checked out that exact SHA.

The two files here are the unmodified GitHub artifact ZIP responses. Their
SHA256 values equal the live artifact API digests:

| Language | Artifact ID | ZIP SHA256 | SARIF member SHA256 | Results |
| --- | ---: | --- | --- | ---: |
| Actions | `10950643261` | `59792942e78b1192ee289b823dcfef1352f14ce6ef16ac2cfa62880e205f8b13` | `c2d5ee2dd0a7be450d57cc44f1687ed506cc6e8db068f23e9857dcd06a9b3cbc` | 0 |
| Python | `10950343881` | `cee6e72a21233ceee463035b4d2d37c1cd6a9f556a612171b4667b62ed86e23b` | `71a7c24c74e302c86d7fcbf5d4d33fffd7d33a73384a37c476493e6387d18796` | 0 |

From this directory, verify the retained bytes with Python's standard library:

```bash
python3 - <<'PY'
import hashlib
import json
import zipfile
from pathlib import Path

expected = {
    'actions': ('59792942e78b1192ee289b823dcfef1352f14ce6ef16ac2cfa62880e205f8b13',
                'c2d5ee2dd0a7be450d57cc44f1687ed506cc6e8db068f23e9857dcd06a9b3cbc'),
    'python': ('cee6e72a21233ceee463035b4d2d37c1cd6a9f556a612171b4667b62ed86e23b',
               '71a7c24c74e302c86d7fcbf5d4d33fffd7d33a73384a37c476493e6387d18796'),
}
for language, (zip_sha, sarif_sha) in expected.items():
    path = Path(f'{language}-artifact.zip')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == zip_sha
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == [f'{language}.sarif']
        assert archive.testzip() is None
        raw = archive.read(f'{language}.sarif')
    assert hashlib.sha256(raw).hexdigest() == sarif_sha
    sarif = json.loads(raw)
    assert len(sarif['runs']) == 1
    assert sarif['runs'][0]['tool']['driver']['name'] == 'CodeQL'
    assert sarif['runs'][0]['results'] == []
    print(language, 'zero CodeQL results, hashes verified')
PY
```

This is evidence for the named pre-merge main SHA only. It does not prove
CodeQL on the subsequent `3a05e8a0` merge, clear license HOLDs, or establish
published 0.11.5 wheel acceptance. No new scan was dispatched for this receipt.

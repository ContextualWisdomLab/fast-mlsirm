#!/usr/bin/env bash
set -euo pipefail

target=${1:?target triple required}
case "$target" in
  x86_64-unknown-linux-gnu)
    source_file=LICENSE-THIRD-PARTY
    snapshot_file=tools/third_party_licenses.snapshot.json
    ;;
  aarch64-unknown-linux-gnu|x86_64-pc-windows-msvc)
    source_file="docs/security/license-evidence-0.11.5/target-notices/LICENSE-THIRD-PARTY-$target"
    snapshot_file="docs/security/license-evidence-0.11.5/target-notices/$target.snapshot.json"
    ;;
  *)
    echo "no reviewed third-party license bundle for $target" >&2
    exit 1
    ;;
esac

test -f "$source_file"
test -f "$snapshot_file"
head -n 2 "$source_file" | grep -Fq "Target: $target. Entries:"
python - "$source_file" "$snapshot_file" "$target" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

notice = Path(sys.argv[1]).read_text()
raw = Path(sys.argv[2]).read_bytes()
snapshot = json.loads(raw)
header = notice[:1000]
if (snapshot["binding_target"] != sys.argv[3]
        or f"snapshot sha256 {hashlib.sha256(raw).hexdigest()}" not in header
        or f"Entries: {len(snapshot['rows'])}." not in header):
    raise SystemExit("reviewed notice and snapshot do not match target")
PY
if [[ "$source_file" != LICENSE-THIRD-PARTY ]]; then
  cp "$snapshot_file" tools/third_party_licenses.snapshot.json
  cp "$source_file" LICENSE-THIRD-PARTY
fi

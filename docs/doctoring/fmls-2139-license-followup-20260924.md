# PR 2139 license-inventory follow-up

Base commit: `6e57aec9ad06a0c847320154dc9cf933d401f78a`.

The Cargo registry evidence path now reads each `.crate` once through a stable
regular-file descriptor. The descriptor identity, size, and modification time
must agree with the source path before and after the read. The same captured
bytes supply both the Cargo.lock checksum comparison and license extraction.
Each extracted license row records the enclosing artifact SHA-256.

The inventory stays on hold when a hash-bound registry archive contains no
license file, when the source path changes during the read, or when artifact
text grants GPL, LGPL, or AGPL terms absent from the declared SPDX expression.
Package metadata remains separate evidence and cannot replace or erase the
artifact text.

Focused verification used the repository's existing Python environment and
performed no package installation, download, Cargo invocation, or release:

```text
python3 -m py_compile tools/license_inventory.py tests/test_license_inventory_generator.py
exit 0

python3 -m pytest -q tests/test_license_inventory_generator.py
27 passed in 0.39s
exit 0
```

The focused tests cover missing Cargo license text, a declared MIT license with
separate LGPL grant text, one-byte-source reuse for hash and extraction, source
path replacement during a read, and the unchanged matching-byte permissive
case. This evidence does not establish the complete release inventory, hosted
checks, independent review, or release acceptance.

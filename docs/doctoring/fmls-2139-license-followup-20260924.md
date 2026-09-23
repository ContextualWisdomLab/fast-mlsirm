# PR 2139 license-inventory follow-up

Base commit: `6e57aec9ad06a0c847320154dc9cf933d401f78a`.

The Cargo registry evidence path now reads each `.crate` once through a stable
regular-file descriptor. The descriptor identity, size, and modification time
must agree with the source path before and after the read. The same captured
bytes supply both the Cargo.lock checksum comparison and license extraction.
Each extracted license row records the enclosing artifact SHA-256.

The inventory stays on hold when a hash-bound registry archive contains no
license file, when the source path changes during the read, when a license
candidate is unrecognized or conditional, or when artifact text grants GPL,
LGPL, or AGPL terms absent from the declared SPDX expression. Candidate files
are collected at every archive depth, and Cargo metadata's `license_file` path
is included even when its basename is not license-like. MIT acceptance requires
independent grant, warranty, and liability clauses from the standard text, and
the normalized body must equal the canonical template after removing only an
optional `MIT License` title. Copyright-looking headers, added conditions, and
other trailing prose are not accepted without exact pre-reviewed bytes: a
holder-name grammar cannot distinguish a name from an appended ASCII-only
condition. Other recognized license phrases remain unverified and HOLD until
their own complete canonical validators exist. Every
collected candidate file must independently have canonical verification, so a
valid MIT file cannot mask a restricted COPYING file or a title-only NOTICE.
Detection is case-insensitive. A Cargo metadata `license_file` path must exist
in the hash-bound archive. Package metadata remains separate evidence and
cannot replace or erase the artifact text.

Focused verification used the repository's existing Python environment and
performed no package installation, download, Cargo invocation, or release:

```text
python3 -m py_compile tools/license_inventory.py tests/test_license_inventory_generator.py
exit 0

PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -p no:cacheprovider --noconftest -q tests/test_license_inventory_generator.py
43 passed in 0.29s
exit 0
```

The focused tests cover missing or unrecognized Cargo license text, commercial
restrictions, negated MIT phrases, a declared MIT license with nested or
uppercase LGPL grant text, metadata-declared nonstandard license paths,
one-byte-source reuse for hash and extraction, source path replacement during a
read, and a full standard MIT-text matching-byte permissive case. The six
preserved independent-review counterexamples all return HOLD under this commit.
The later academic-use-only and missing-declared-license-file counterexamples
also return HOLD, while their preserved canonical-MIT positive control remains
PERMISSIVE.
The former copyright-header positive fixture intentionally changes from
PERMISSIVE to HOLD; both academic-only and commercial-use-prohibited ASCII
headers return HOLD. The headerless canonical MIT body remains PERMISSIVE. The
two multi-file masking counterexamples also return HOLD.
This evidence does not establish the complete release inventory, hosted checks,
independent review, or release acceptance.

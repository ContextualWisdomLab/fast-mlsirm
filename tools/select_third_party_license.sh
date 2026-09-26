#!/usr/bin/env bash
set -euo pipefail

target=${1:?target triple required}
case "$target" in
  x86_64-unknown-linux-gnu)
    source_file=LICENSE-THIRD-PARTY
    ;;
  aarch64-unknown-linux-gnu|x86_64-pc-windows-msvc)
    source_file="docs/security/license-evidence-0.11.5/target-notices/LICENSE-THIRD-PARTY-$target"
    ;;
  *)
    echo "no reviewed third-party license bundle for $target" >&2
    exit 1
    ;;
esac

test -f "$source_file"
head -n 2 "$source_file" | grep -Fq "Target: $target. Entries:"
if [[ "$source_file" != LICENSE-THIRD-PARTY ]]; then
  cp "$source_file" LICENSE-THIRD-PARTY
fi

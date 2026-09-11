#!/usr/bin/env bash
set -euo pipefail

# GitHub's Ubuntu 24.04 image already ships Google Chrome, including a software
# Vulkan implementation. Reuse that immutable-for-the-job image payload rather
# than placing the required GPU gate behind a live apt mirror.
chrome_root="${FAST_MLSIRM_SWIFTSHADER_ROOT:-/opt/google/chrome}"
icd_manifest="${chrome_root}/vk_swiftshader_icd.json"
swiftshader_driver="${chrome_root}/libvk_swiftshader.so"
vulkan_loader="${chrome_root}/libvulkan.so.1"

for required_file in "$icd_manifest" "$swiftshader_driver" "$vulkan_loader"; do
  if [[ ! -f "$required_file" || ! -r "$required_file" ]]; then
    echo "software Vulkan image contract missing required file: $required_file" >&2
    exit 1
  fi
done

python3 - "$icd_manifest" "$swiftshader_driver" <<'PY'
import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1]).resolve(strict=True)
expected_driver = Path(sys.argv[2]).resolve(strict=True)
try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    library_path = manifest["ICD"]["library_path"]
except (KeyError, TypeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid SwiftShader Vulkan manifest: {exc}") from exc

if not isinstance(library_path, str) or not library_path:
    raise SystemExit("invalid SwiftShader Vulkan manifest library_path")
resolved_driver = Path(library_path)
if not resolved_driver.is_absolute():
    resolved_driver = manifest_path.parent / resolved_driver
try:
    resolved_driver = resolved_driver.resolve(strict=True)
except OSError as exc:
    raise SystemExit(f"SwiftShader manifest driver is unavailable: {exc}") from exc
if resolved_driver != expected_driver:
    raise SystemExit(
        "SwiftShader manifest driver mismatch: "
        f"expected {expected_driver}, got {resolved_driver}"
    )
PY

runtime_dir="${XDG_RUNTIME_DIR:-/tmp/fast-mlsirm-xdg-runtime}"
mkdir -p "$runtime_dir"
chmod 700 "$runtime_dir"

github_env="${GITHUB_ENV:?GITHUB_ENV must be provided by GitHub Actions}"
printf 'VK_DRIVER_FILES=%s\n' "$icd_manifest" >> "$github_env"
printf 'XDG_RUNTIME_DIR=%s\n' "$runtime_dir" >> "$github_env"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  printf 'LD_LIBRARY_PATH=%s:%s\n' "$chrome_root" "$LD_LIBRARY_PATH" >> "$github_env"
else
  printf 'LD_LIBRARY_PATH=%s\n' "$chrome_root" >> "$github_env"
fi

# Preserve exact image-provided binary identity in the immutable job log.
sha256sum "$icd_manifest" "$swiftshader_driver" "$vulkan_loader"
if [[ -x "${chrome_root}/google-chrome" ]]; then
  "${chrome_root}/google-chrome" --version
fi

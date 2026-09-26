"""Build LICENSE-THIRD-PARTY from a hash-verified Cargo license inventory (stdlib only).

Two steps, both deterministic:

  snapshot  --inventory <license_inventory.py output> --reviewed-fixture <reviewed texts JSON>
            --binding-target <triple> --out <snapshot.json>
      Keeps the non-own crates in the target binding graph with the fields needed to
      re-verify and render them, plus each row's acceptance basis.

  render    --snapshot <snapshot.json> --cargo-registry-cache <$CARGO_HOME/registry/cache>
            [--upstream-evidence <upstream JSON>] --out LICENSE-THIRD-PARTY
      Re-reads every license file from the .crate whose sha256 is recorded (or from the
      sha-pinned upstream file), checks every sha256, adds the crate's own NOTICE file for
      Apache-2.0 crates (Apache-2.0 section 4(d)), and fails if any crate lacks a text.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
from pathlib import Path

MARKER = "coordinator verdict (user-delegated, 2026-09-26), not legal review"
BASIS_ORDER = ["documented exception", "pointer rule", "upstream-vcs", "SPDX template",
               "byte-identical reviewed", "canonical"]
NOTICE_NAME = re.compile(r"^NOTICE(\.[A-Za-z0-9]+)?$", re.I)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def row_basis(row: dict, categories: dict[str, str]) -> str:
    """Strongest special basis for a PERMISSIVE row (see BASIS_ORDER)."""
    found = set()
    if row.get("license_text_origin") == "upstream-vcs":
        found.add("upstream-vcs")
    for f in row["license_files_in_artifact"]:
        if f.get("documented_exception"):
            found.add("documented exception")
        elif (f.get("pointer_notice") or {}).get("satisfied"):
            found.add("pointer rule")
        elif f.get("verified_standard_text"):
            found.add(categories.get(f["sha256"], "canonical"))
    return min(found, key=BASIS_ORDER.index) if found else "none"


def snapshot(inventory: dict, fixture: list[dict], target: str) -> dict:
    first_reviewed = {"foldhash@0.2.0", "atheris@3.1.0", "allocator-api2-0.2.21", "glow-0.17.0"}
    categories = {e["raw_sha256"]: "SPDX template" if "spdx_template_spans" in e
                  else "canonical" if e["package"] in first_reviewed else "byte-identical reviewed"
                  for e in fixture}
    rows = []
    for r in inventory["cargo"]:
        if not r.get("in_binding_target_graph") or r["source"].startswith("path"):
            continue
        if r["license_class"] != "PERMISSIVE":
            raise SystemExit(f"{r['name']}@{r['version']} is {r['license_class']}, not PERMISSIVE")
        rows.append({
            "name": r["name"], "version": r["version"], "spdx": r["spdx_normalized"],
            "elected": r["elected_license"], "crate_sha256": r["source_hash"]["measured"],
            "basis": row_basis(r, categories),
            "files": sorted(({"path": f["path"], "sha256": f["sha256"],
                              "origin": f.get("origin", "crate"), "url": f.get("url"),
                              "vcs_sha1": f.get("vcs_sha1")} for f in r["license_files_in_artifact"]),
                            key=lambda f: f["path"]),
        })
    rows.sort(key=lambda r: (r["name"], r["version"]))
    return {"binding_target": target, "rows": rows}


def crate_members(cache: Path, name: str, version: str, expected: str) -> dict[str, bytes]:
    paths = sorted(cache.glob(f"*/{name}-{version}.crate"))
    if not paths:
        raise SystemExit(f"{name}@{version}: .crate not found under {cache}")
    data = paths[0].read_bytes()
    if sha256(data) != expected:
        raise SystemExit(f"{name}@{version}: .crate sha256 {sha256(data)} != recorded {expected}")
    prefix = f"{name}-{version}/"
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        return {m.name[len(prefix):]: tf.extractfile(m).read()
                for m in tf.getmembers() if m.isfile() and m.name.startswith(prefix)}


def render(snap: dict, cache: Path, upstream: dict, upstream_base: Path) -> tuple[str, int]:
    out, notices = [], 0
    for row in snap["rows"]:
        ident = f"{row['name']}@{row['version']}"
        if not row["files"]:
            raise SystemExit(f"{ident}: no license text in the inventory")
        members = crate_members(cache, row["name"], row["version"], row["crate_sha256"])
        texts = []
        for f in row["files"]:
            if f["origin"] == "upstream-vcs":
                rec = next((u for u in upstream.get(ident, {}).get("files", []) if u["path"] == f["path"]), None)
                if rec is None:
                    raise SystemExit(f"{ident}: upstream file {f['path']} not in --upstream-evidence")
                data = (upstream_base / rec["local_path"]).read_bytes()
            elif f["path"] in members:
                data = members[f["path"]]
            else:
                raise SystemExit(f"{ident}: {f['path']} missing from the .crate")
            if sha256(data) != f["sha256"]:
                raise SystemExit(f"{ident}: {f['path']} sha256 {sha256(data)} != recorded {f['sha256']}")
            texts.append((f, data))
        apache_notices = sorted((p, d) for p, d in members.items()
                                if "Apache-2.0" in (row["spdx"] or "") and NOTICE_NAME.match(p)
                                and all(p != f["path"] for f in row["files"]))
        notices += len(apache_notices)
        out += ["=" * 78, f"{row['name']} {row['version']}",
                f"SPDX: {row['spdx']}    Elected: {row['elected']}",
                f"Crate sha256: {row['crate_sha256']}", f"Acceptance basis: {row['basis']}", "Files:"]
        out += [f"  {f['path']}  sha256 {f['sha256']}" + (f"  (upstream {f['url']} @ {f['vcs_sha1']})"
                if f["origin"] == "upstream-vcs" else "") for f, _ in texts]
        out += [f"  {p}  sha256 {sha256(d)}  (Apache-2.0 NOTICE)" for p, d in apache_notices]
        for f, data in texts:
            out += ["", f"----- {row['name']} {row['version']}: {f['path']} -----",
                    data.decode("utf-8", "replace").rstrip("\n")]
        for p, d in apache_notices:
            out += ["", f"----- {row['name']} {row['version']}: {p} (NOTICE) -----", d.decode("utf-8", "replace").rstrip("\n")]
        out.append("")
    return "\n".join(out) + "\n", notices


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--inventory", required=True)
    s.add_argument("--reviewed-fixture", required=True)
    s.add_argument("--binding-target", required=True)
    s.add_argument("--out", required=True)
    r = sub.add_parser("render")
    r.add_argument("--snapshot", required=True)
    r.add_argument("--cargo-registry-cache", required=True)
    r.add_argument("--upstream-evidence")
    r.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "snapshot":
        inventory = json.loads(Path(args.inventory).read_text())
        if inventory["summary"].get("cargo_binding_target") != args.binding_target:
            raise SystemExit("inventory was not produced for --binding-target")
        snap = snapshot(inventory, json.loads(Path(args.reviewed_fixture).read_text()), args.binding_target)
        snap["source_inventory_sha256"] = sha256(Path(args.inventory).read_bytes())
        Path(args.out).write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")
        return 0
    snap_bytes = Path(args.snapshot).read_bytes()
    snap = json.loads(snap_bytes)
    upstream_path = Path(args.upstream_evidence) if args.upstream_evidence else None
    upstream = json.loads(upstream_path.read_text()) if upstream_path else {}
    body, notices = render(snap, Path(args.cargo_registry_cache), upstream, upstream_path.parent if upstream_path else Path("."))
    header = [
        "THIRD-PARTY LICENSES for the Rust crates statically linked into fast_mlsirm/_core",
        f"Target: {snap['binding_target']}. Entries: {len(snap['rows'])}. Apache-2.0 NOTICE files included: {notices}.",
        f"Acceptance: {MARKER}.",
        f"Generated by tools/third_party_licenses.py from snapshot sha256 {sha256(snap_bytes)}",
        f"(source inventory sha256 {snap.get('source_inventory_sha256', 'unknown')}).",
        "This repository's own crates (mlsirm-core, fast-mlsirm-py) are covered by LICENSE.",
        "",
    ]
    Path(args.out).write_text("\n".join(header) + body)
    print(f"entries={len(snap['rows'])} apache_notices={notices}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

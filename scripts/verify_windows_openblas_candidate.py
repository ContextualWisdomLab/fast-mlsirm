"""Fail closed on the first, DLL-only Windows OpenBLAS candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


OPENBLAS_COMMIT = "446c436e10450c348169808f1a6b3fae0925c9f7"
LICENSE_HASHES = {
    "LICENSE": "190b5a9c8d9723fe958ad33916bd7346d96fab3c5ea90832bb02d854f620fcff",
    "lapack-netlib/LICENSE": "978539245c405775bf347b1256c36df4daf8b69a64b099e94b1d53742201621e",
}
FORBIDDEN = re.compile(r"(?i)(?:lib)?(?:gcc|gfortran|quadmath|winpthread|stdc\+\+)")
ALLOWED_DLL = re.compile(
    r"(?i)^(?:kernel32|vcruntime140(?:_1)?|ucrtbase|"
    r"api-ms-win-(?:crt|core)-[a-z0-9-]+)\.dll$"
)
MEMBER = re.compile(r"(?im)^\s*Loaded\s+(.+?\.(?:lib|a)\([^)]+\))\s*$")
IMPORT = re.compile(r"(?im)^\s*([\w.+-]+\.dll)\s*$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(imports: str, link_log: str, map_text: str) -> tuple[list[str], list[str]]:
    if "Image has the following dependencies:" not in imports:
        raise ValueError("dumpbin /DEPENDENTS section missing")
    dlls = sorted(set(IMPORT.findall(imports)), key=str.lower)
    if not dlls or any(not ALLOWED_DLL.fullmatch(dll) for dll in dlls):
        raise ValueError(f"unknown or prohibited PE import: {dlls}")
    if any(
        "loaded" in line.lower() and FORBIDDEN.search(line)
        for line in link_log.splitlines()
    ):
        raise ValueError("prohibited runtime in linker load trace")
    members = MEMBER.findall(link_log)
    if not members:
        raise ValueError("/VERBOSE:LIB did not identify loaded archive members")
    if any(FORBIDDEN.search(member) for member in members):
        raise ValueError("prohibited archive member loaded")
    if not map_text.strip() or "openblas" not in map_text.lower():
        raise ValueError("OpenBLAS /MAP evidence missing")
    if FORBIDDEN.search(map_text):
        raise ValueError("prohibited runtime appears in /MAP evidence")
    return dlls, members


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--imports", type=Path, required=True)
    parser.add_argument("--link-log", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.dll.is_file() or not args.dll.stat().st_size:
        raise ValueError("candidate DLL missing or empty")
    for relpath, expected in LICENSE_HASHES.items():
        if sha256(args.source / relpath) != expected:
            raise ValueError(f"pinned {relpath} changed")
    dlls, members = inspect(
        args.imports.read_text(encoding="utf-8", errors="replace"),
        args.link_log.read_text(encoding="utf-8", errors="replace"),
        args.map.read_text(encoding="utf-8", errors="replace"),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    digest = sha256(args.dll)
    evidence = {
        "source_commit": OPENBLAS_COMMIT,
        "dll": args.dll.name,
        "dll_sha256": digest,
        "imports": dlls,
        "loaded_archive_members": members,
        "map_sha256": sha256(args.map),
        "link_log_sha256": sha256(args.link_log),
        "license_sha256": LICENSE_HASHES,
    }
    (args.out / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": {
                "type": "library",
                "name": args.dll.name,
                "hashes": [{"alg": "SHA-256", "content": digest}],
            }
        },
        "components": [
            {"type": "library", "name": "OpenBLAS", "version": OPENBLAS_COMMIT},
            {
                "type": "library",
                "name": "LAPACK C translation",
                "version": OPENBLAS_COMMIT,
            },
            *({"type": "library", "name": name} for name in dlls),
            *({"type": "library", "name": name} for name in members),
        ],
    }
    (args.out / "sbom.cdx.json").write_text(
        json.dumps(sbom, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    notice = "OpenBLAS\n" + (args.source / "LICENSE").read_text(encoding="utf-8")
    notice += "\nLAPACK C translation\n" + (
        args.source / "lapack-netlib/LICENSE"
    ).read_text(encoding="utf-8")
    (args.out / "NOTICE.txt").write_text(notice, encoding="utf-8")


if __name__ == "__main__":
    main()

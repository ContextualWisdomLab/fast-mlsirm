"""Record the target Cargo graph and tools in the wheel's actual build environment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib


def _run(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def capture(source: Path, environ: dict[str, str]) -> dict:
    leg = environ["CARGO_RELEASE_LEG"]
    source_sha = environ["CARGO_RELEASE_SHA"]
    build_pass = environ["CARGO_BUILD_PASS"]
    image = environ.get("CARGO_BUILD_IMAGE", "")
    interpreter = environ["CARGO_BUILD_PYTHON"]
    if (not re.fullmatch(r"(?:x86_64-unknown-linux-gnu|aarch64-unknown-linux-gnu|universal2-apple-darwin|x86_64-pc-windows-msvc)-py3\.1[234]", leg)
            or not re.fullmatch(r"[0-9a-f]{40}", source_sha)
            or build_pass not in ("first", "second")
            or _run("git", "-C", str(source), "rev-parse", "HEAD") != source_sha):
        raise ValueError("build scope is not bound to the release source and leg")
    target, python = leg.rsplit("-py", 1)
    if (target.endswith("linux-gnu") and not re.fullmatch(r"[a-z0-9./_-]+@sha256:[0-9a-f]{64}", image)
            or not target.endswith("linux-gnu") and image):
        raise ValueError("build scope container identity differs from target")
    if not _run(interpreter, "--version").startswith(f"Python {python}."):
        raise ValueError("build interpreter differs from wheel target")
    pyproject = source / "pyproject.toml"
    lock = source / "crates/fast-mlsirm-py/Cargo.lock"
    locked = {(item["name"], item["version"], item.get("source")): item.get("checksum")
              for item in tomllib.loads(lock.read_text())["package"]}
    if tomllib.loads(pyproject.read_text())["tool"]["maturin"]["features"] != ["pyo3/extension-module"]:
        raise ValueError("unrecognized maturin feature selection")
    targets = (["aarch64-apple-darwin", "x86_64-apple-darwin"]
               if target == "universal2-apple-darwin" else [target])
    graphs = {}
    for triple in targets:
        metadata = json.loads(_run(
            "cargo", "metadata", "--locked", "--format-version", "1",
            "--filter-platform", triple, "--manifest-path",
            str(source / "crates/fast-mlsirm-py/Cargo.toml"),
            "--features", "pyo3/extension-module",
        ))
        packages = {item["id"]: item for item in metadata["packages"]}
        nodes = metadata["resolve"]["nodes"]
        if metadata["resolve"]["root"] not in {node["id"] for node in nodes}:
            raise ValueError("Cargo graph has no selected wheel crate")
        graphs[triple] = sorted(({
            "name": packages[node["id"]]["name"],
            "version": packages[node["id"]]["version"],
            "source": packages[node["id"]]["source"],
            "checksum": locked[(packages[node["id"]]["name"],
                                packages[node["id"]]["version"],
                                packages[node["id"]]["source"])],
            "features": sorted(node["features"]),
        } for node in nodes), key=lambda row: (row["name"], row["version"], row["source"] or ""))
    return {
        "schema_version": 1, "source_sha": source_sha, "leg": leg, "pass": build_pass,
        "build_env": "container:" + image if image else "runner:" + "/".join(
            environ[key] for key in ("ImageOS", "ImageVersion", "RUNNER_OS", "RUNNER_ARCH")),
        "cargo_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "pyproject_sha256": hashlib.sha256(pyproject.read_bytes()).hexdigest(),
        "cargo_version": _run("cargo", "--version"),
        "rustc_version": _run("rustc", "--version"),
        "maturin_version": _run("maturin", "--version"),
        "python_version": _run(interpreter, "--version"),
        "cargo_features": ["pyo3/extension-module"], "cargo_targets": graphs,
    }


if __name__ == "__main__":
    result = capture(Path.cwd(), dict(os.environ))
    output = Path("repro-digest") / f"{result['leg']}.build-{result['pass']}.json"
    output.parent.mkdir(exist_ok=True)
    if output.exists() or output.is_symlink():
        raise SystemExit("build scope output already exists")
    output.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")

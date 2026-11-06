#!/usr/bin/env python3
"""Run remote_worker without requiring a built ``_core`` extension.

Used for multi-host Valkey/SubprocessExecutor path validation when the host
has NumPy + package sources but no maturin-built extension. Loads leaf modules
via a stub package so ``mc_replicate`` → ``simulate`` works.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "python"
PKG_DIR = ROOT / "fast_mlsirm"
sys.path.insert(0, str(ROOT))

pkg = types.ModuleType("fast_mlsirm")
pkg.__path__ = [str(PKG_DIR)]
pkg.__file__ = str(PKG_DIR / "__init__.py")
sys.modules["fast_mlsirm"] = pkg

# Import leaf modules that mc_replicate needs (no package __init__).
for name in (
    "fast_mlsirm.types",
    "fast_mlsirm.math",
    "fast_mlsirm.config",
    "fast_mlsirm.simulation",
    "fast_mlsirm.remote_exec",
    "fast_mlsirm.remote_worker",
):
    importlib.import_module(name)

from fast_mlsirm.remote_worker import main

import os
os.environ.setdefault("FAST_MLSIRM_LIBRARY_VERSION", "0.11.4")

if __name__ == "__main__":
    raise SystemExit(main())

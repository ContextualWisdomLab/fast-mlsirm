"""Guard one-shot integration against queued stale workflow definitions."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


if (
    Path(sys.argv[0]).name == "apply_rotation_core_patch.py"
    and os.environ.get("FAST_MLSIRM_ROTATION_MATH_FIXED") != "1"
):
    environment = dict(os.environ)
    environment["FAST_MLSIRM_ROTATION_MATH_FIXED"] = "1"
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("fix_rotation_math_contract.py"))],
        check=True,
        env=environment,
    )

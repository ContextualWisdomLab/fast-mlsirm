"""Defense-in-depth contract for the raw Python polytomous prediction helper."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_raw_polytomous_predictions_reject_category_count_above_fitter_domain() -> None:
    """The raw helper must replay the 64-category fitter ceiling before core discovery."""

    program = textwrap.dedent(
        """
        import importlib
        import numpy as np
        import fast_mlsirm.polytomous as module

        module = importlib.reload(module)

        def unexpected_core_discovery():
            raise AssertionError("out-of-domain raw prediction reached compiled-core discovery")

        module._core_module = unexpected_core_discovery
        fit = module.PolytomousFit(
            "gpcm",
            np.array([1.0]),
            np.zeros((1, 64), dtype=np.float64),
            0.0,
            0,
        )

        try:
            module._polytomous_predictions(fit, np.array([0.0]))
        except ValueError as error:
            assert str(error) == "n_cat must be in 2..=64"
        else:
            raise AssertionError("raw prediction accepted 65 categories")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr

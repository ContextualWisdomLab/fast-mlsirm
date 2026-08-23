"""Defense-in-depth contract for the internal polytomous prediction adapter."""

from importlib import reload

import numpy as np
import pytest

import fast_mlsirm.polytomous as polytomous_module


def test_raw_prediction_replays_fitter_category_ceiling_before_native(monkeypatch):
    """The uninstalled raw adapter must reject 65 categories before Rust discovery."""

    raw_module = reload(polytomous_module)
    fit = raw_module.PolytomousFit(
        "gpcm",
        np.array([1.0]),
        np.zeros((1, 64)),
        0.0,
        0,
    )

    def unexpected_core_discovery():
        raise AssertionError("out-of-domain category count reached compiled-core discovery")

    monkeypatch.setattr(raw_module, "_core_module", unexpected_core_discovery)
    with pytest.raises(ValueError, match=r"n_cat must be in 2\.\.=64"):
        raw_module._polytomous_predictions(fit, np.array([0.0]))

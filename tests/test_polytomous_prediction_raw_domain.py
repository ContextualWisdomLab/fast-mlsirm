"""Defense-in-depth contract for the internal polytomous prediction adapter."""

import numpy as np
import pytest

import fast_mlsirm.polytomous as polytomous_module


def test_raw_prediction_replays_fitter_category_ceiling_before_native(monkeypatch):
    """The raw adapter must reject 65 categories before compiled-core discovery."""

    fit = polytomous_module.PolytomousFit(
        "gpcm",
        np.array([1.0]),
        np.zeros((1, 64)),
        0.0,
        0,
    )

    def unexpected_core_discovery():
        raise AssertionError("out-of-domain category count reached compiled-core discovery")

    monkeypatch.setattr(polytomous_module, "_core_module", unexpected_core_discovery)
    with pytest.raises(ValueError, match=r"n_cat must be in 2\.\.=64"):
        polytomous_module._polytomous_predictions(fit, np.array([0.0]))

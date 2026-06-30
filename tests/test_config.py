import pytest
from fast_mlsirm.config import FitConfig, MLS2PLMConfig

def test_mls2plm_config_validation():
    # test n_persons < 1
    with pytest.raises(ValueError, match="n_persons must be >= 1"):
        MLS2PLMConfig(n_persons=0).validate()

    # test n_dims < 1
    with pytest.raises(ValueError, match="n_dims must be >= 1"):
        MLS2PLMConfig(n_dims=0).validate()

    # test items_per_dim < 1
    with pytest.raises(ValueError, match="items_per_dim must be >= 1"):
        MLS2PLMConfig(items_per_dim=0).validate()

    # test latent_dim < 1
    with pytest.raises(ValueError, match="latent_dim must be >= 1"):
        MLS2PLMConfig(latent_dim=0).validate()

    # test phi
    with pytest.raises(ValueError, match="phi must produce a positive-definite equicorrelation matrix"):
        MLS2PLMConfig(phi=1.5).validate()
    with pytest.raises(ValueError, match="phi must produce a positive-definite equicorrelation matrix"):
        MLS2PLMConfig(n_dims=2, phi=-1.1).validate()

    # test gamma
    with pytest.raises(ValueError, match="gamma must be >= 0"):
        MLS2PLMConfig(gamma=-0.1).validate()

    # test dtype
    with pytest.raises(ValueError, match="dtype must be float32 or float64"):
        MLS2PLMConfig(dtype="int32").validate()

def test_fit_config_validation():
    # test valid model
    with pytest.raises(ValueError, match="model must be one of"):
        FitConfig(model="INVALID").validate()

    # test latent_dim
    with pytest.raises(ValueError, match="latent_dim must be >= 1"):
        FitConfig(latent_dim=0).validate()

    # test optimizer
    with pytest.raises(ValueError, match="optimizer must be one of"):
        FitConfig(optimizer="invalid_opt").validate()

    # test max_iter
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        FitConfig(max_iter=0).validate()

    # test n_restarts
    with pytest.raises(ValueError, match="n_restarts must be >= 1"):
        FitConfig(n_restarts=0).validate()

    # test learning_rate
    with pytest.raises(ValueError, match="learning_rate must be > 0"):
        FitConfig(learning_rate=-0.1).validate()

    # test init_gamma
    with pytest.raises(ValueError, match="init_gamma must be > 0"):
        FitConfig(init_gamma=0).validate()

    # test eps_distance
    with pytest.raises(ValueError, match="eps_distance must be > 0"):
        FitConfig(eps_distance=0).validate()

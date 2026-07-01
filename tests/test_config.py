import pytest
from fast_mlsirm.config import MLS2PLMConfig, FitConfig

def test_mls2plmconfig_valid():
    """Test valid MLS2PLMConfig passes validation."""
    config = MLS2PLMConfig()
    config.validate()  # Should not raise any exception

def test_fitconfig_valid():
    """Test valid FitConfig passes validation."""
    config = FitConfig()
    config.validate()  # Should not raise any exception
    assert config.backend == "numpy"

def test_mls2plmconfig_invalid_n_persons():
    with pytest.raises(ValueError, match="n_persons must be >= 1"):
        MLS2PLMConfig(n_persons=0).validate()

def test_mls2plmconfig_invalid_n_dims():
    with pytest.raises(ValueError, match="n_dims must be >= 1"):
        MLS2PLMConfig(n_dims=0).validate()

def test_mls2plmconfig_invalid_items_per_dim():
    with pytest.raises(ValueError, match="items_per_dim must be >= 1"):
        MLS2PLMConfig(items_per_dim=0).validate()

def test_mls2plmconfig_invalid_latent_dim():
    with pytest.raises(ValueError, match="latent_dim must be >= 1"):
        MLS2PLMConfig(latent_dim=0).validate()

def test_mls2plmconfig_invalid_phi_lower_bound():
    config = MLS2PLMConfig(n_dims=2, phi=-1.0)
    with pytest.raises(ValueError, match="phi must produce a positive-definite equicorrelation matrix"):
        config.validate()

def test_mls2plmconfig_invalid_phi_upper_bound():
    with pytest.raises(ValueError, match="phi must produce a positive-definite equicorrelation matrix"):
        MLS2PLMConfig(phi=1.0).validate()

def test_mls2plmconfig_invalid_gamma():
    with pytest.raises(ValueError, match="gamma must be >= 0"):
        MLS2PLMConfig(gamma=-0.1).validate()

def test_mls2plmconfig_invalid_dtype():
    with pytest.raises(ValueError, match="dtype must be float32 or float64"):
        MLS2PLMConfig(dtype="float16").validate()

def test_fitconfig_invalid_model():
    with pytest.raises(ValueError, match="model must be one of"):
        FitConfig(model="INVALID").validate()

def test_fitconfig_invalid_latent_dim():
    with pytest.raises(ValueError, match="latent_dim must be >= 1"):
        FitConfig(latent_dim=0).validate()

def test_fitconfig_invalid_optimizer():
    with pytest.raises(ValueError, match="optimizer must be one of"):
        FitConfig(optimizer="sgd").validate()

def test_fitconfig_invalid_max_iter():
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        FitConfig(max_iter=0).validate()

def test_fitconfig_invalid_n_restarts():
    with pytest.raises(ValueError, match="n_restarts must be >= 1"):
        FitConfig(n_restarts=0).validate()

def test_fitconfig_invalid_learning_rate():
    with pytest.raises(ValueError, match="learning_rate must be > 0"):
        FitConfig(learning_rate=0.0).validate()

def test_fitconfig_invalid_init_gamma():
    with pytest.raises(ValueError, match="init_gamma must be > 0"):
        FitConfig(init_gamma=0.0).validate()

def test_fitconfig_invalid_eps_distance():
    with pytest.raises(ValueError, match="eps_distance must be > 0"):
        FitConfig(eps_distance=0.0).validate()


def test_fitconfig_invalid_backend():
    with pytest.raises(ValueError, match="backend must be one of"):
        FitConfig(backend="cuda").validate()

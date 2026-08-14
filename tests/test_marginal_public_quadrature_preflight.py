"""Public-entry regressions for bounded marginal quadrature preflight."""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

import fast_mlsirm.estimators.marginal as marginal


class _HostileInteger:
    """Caller-controlled object whose integer conversion leaks hostile text."""

    def __int__(self) -> int:
        """Raise if production code attempts caller-controlled conversion."""
        raise RuntimeError("caller-secret-q-xi")


@pytest.mark.parametrize(
    "q_xi",
    (True, 7.5, np.int64(7), _HostileInteger()),
)
def test_spatial_q_xi_fails_before_array_coercion(
    monkeypatch: pytest.MonkeyPatch,
    q_xi: object,
) -> None:
    """Malformed tensor quadrature cannot reach arrays or caller callbacks."""

    def forbidden_asarray(*_args: object, **_kwargs: object) -> np.ndarray:
        pytest.fail("q_xi validation must precede NumPy array coercion")

    monkeypatch.setattr(marginal.np, "asarray", forbidden_asarray)

    with pytest.raises(
        ValueError,
        match="q_xi must be an exact built-in integer",
    ) as caught:
        marginal.fit_marginal_numpy(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            model="MLSRM",
            latent_dim=1,
            q_xi=q_xi,  # type: ignore[arg-type]
        )

    assert "caller-secret-q-xi" not in str(caught.value)


def test_unsupported_exact_q_xi_fails_before_array_coercion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An astronomical exact integer is rejected without constructing its power."""

    def forbidden_asarray(*_args: object, **_kwargs: object) -> np.ndarray:
        pytest.fail("quadrature validation must precede NumPy array coercion")

    monkeypatch.setattr(marginal.np, "asarray", forbidden_asarray)

    with pytest.raises(ValueError, match="unsupported quadrature size"):
        marginal.fit_marginal_numpy(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            model="MLSRM",
            latent_dim=3,
            q_xi=10**200,
        )


def test_tensor_node_count_caps_before_multiplication() -> None:
    """The tensor-node count uses division checks and contains no power operator."""
    assert marginal._bounded_tensor_node_count(  # type: ignore[attr-defined]
        10**200,
        3,
        limit=1_000_000,
    ) == 1_000_001

    tree = ast.parse(inspect.getsource(marginal._bounded_tensor_node_count))  # type: ignore[attr-defined]
    assert not any(isinstance(node, ast.Pow) for node in ast.walk(tree))


def test_nonspatial_preflight_does_not_inspect_unused_q_xi() -> None:
    """MIRT owns one latent-space placeholder node and ignores tensor settings."""
    assert marginal._preflight_xi_node_count(  # type: ignore[attr-defined]
        xi_rule="gh",
        q_xi=_HostileInteger(),
        xi_points=256,
        latent_dim=3,
        uses_space=False,
    ) == 1


def test_qmc_preflight_uses_exact_point_count_without_inspecting_q_xi() -> None:
    """QMC preflight is governed by exact xi_points rather than tensor q_xi."""
    assert marginal._preflight_xi_node_count(  # type: ignore[attr-defined]
        xi_rule="qmc",
        q_xi=_HostileInteger(),
        xi_points=17,
        latent_dim=3,
        uses_space=True,
    ) == 17


@pytest.mark.parametrize(
    ("q_xi", "latent_dim", "limit", "message"),
    (
        ("3", 1, 1_000_000, "q_xi, latent_dim, and limit must be exact built-in integers"),
        (3, 0, 1_000_000, "q_xi, latent_dim, and limit must be positive"),
    ),
)
def test_tensor_node_count_rejects_invalid_contract_types(
    q_xi: object,
    latent_dim: object,
    limit: object,
    message: str,
) -> None:
    """Tensor node counting refuses non-built-in and non-positive controls."""
    with pytest.raises(ValueError, match=message):
        marginal._bounded_tensor_node_count(q_xi, latent_dim, limit=limit)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"uses_space": 1, "q_xi": 3, "xi_points": 3, "latent_dim": 1}, "uses_space"),
        ({"uses_space": True, "q_xi": 7, "xi_points": "3", "latent_dim": 1, "xi_rule": "qmc"}, "xi_points"),
        ({"uses_space": True, "q_xi": 7, "xi_points": 0, "latent_dim": 1, "xi_rule": "qmc"}, "xi_points"),
        ({"uses_space": True, "q_xi": 7, "xi_points": 3, "latent_dim": "1"}, "latent_dim"),
        ({"uses_space": True, "q_xi": 7, "xi_points": 3, "latent_dim": 0}, "latent_dim"),
    ),
)
def test_preflight_rejects_invalid_qmc_and_tensor_controls(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """QMC and tensor rules validate each control before allocation."""
    values = {
        "xi_rule": "gh",
        "q_xi": 3,
        "xi_points": 3,
        "latent_dim": 1,
        "uses_space": True,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        marginal._preflight_xi_node_count(**values)  # type: ignore[arg-type]


def test_tensor_preflight_rejects_a_product_grid_over_the_bound() -> None:
    """A valid quadrature size is still rejected when its tensor product is huge."""
    with pytest.raises(ValueError, match="tensor-grid limit"):
        marginal._preflight_xi_node_count(
            xi_rule="gh",
            q_xi=21,
            xi_points=3,
            latent_dim=5,
            uses_space=True,
        )


def test_marginal_fit_rejects_invalid_theta_quadrature_before_arrays() -> None:
    """The theta quadrature count must be a positive built-in integer."""
    with pytest.raises(ValueError, match="q_theta"):
        marginal.fit_marginal_numpy(
            np.array([[0, 1], [1, 0]], dtype=np.float64),
            np.ones((2, 2), dtype=bool),
            np.zeros(2, dtype=np.int64),
            model="MLSRM",
            latent_dim=1,
            q_xi=7,
            q_theta=0,
        )

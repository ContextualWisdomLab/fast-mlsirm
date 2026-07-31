"""Automated test assembly (ATA) tests.

These exercise the target-information assembler in ``fast_mlsirm.ata`` on a
calibrated bank. They assert real optimal-test-design properties (van der
Linden, 2005): the assembled form's test information function beats a naive /
random selection at the target trait points, the length and content constraints
are honored, exposure caps are respected, and assembly is deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.ata import AssembledForm, assemble_to_target, item_information_matrix
from fast_mlsirm.types import MLSIRMParams

MODEL = "MIRT"  # simple-structure 2PL with no latent-space distance term


def _bank(n_items: int = 30) -> tuple[MLSIRMParams, np.ndarray]:
    """Build a clean unidimensional 2PL item bank and its factor map."""
    a = np.linspace(0.6, 2.4, n_items)
    b = np.linspace(-3.0, 3.0, n_items)
    rng = np.random.default_rng(0)
    a = rng.permutation(a)
    b = rng.permutation(b)
    bank = MLSIRMParams(
        theta=np.array([[0.0]]),
        alpha=np.log(a),
        b=b,
        xi=np.zeros((1, 1)),
        zeta=np.zeros((n_items, 1)),
        tau=-30.0,
    )
    return bank, np.zeros(n_items, dtype=int)


def _random_form_info(bank, fid, thetas, length, seed):
    """Mean test information of a random length-``length`` selection."""
    matrix = item_information_matrix(bank, fid, thetas, model=MODEL)
    rng = np.random.default_rng(seed)
    acc = np.zeros(matrix.shape[0])
    reps = 25
    for _ in range(reps):
        pick = rng.choice(matrix.shape[1], size=length, replace=False)
        acc += matrix[:, pick].sum(axis=1)
    return acc / reps


def test_assembled_tif_beats_random_at_target_points():
    bank, fid = _bank(n_items=30)
    thetas = np.array([-1.0, 0.0, 1.0])
    target = np.array([100.0, 100.0, 100.0])  # unreachable -> pure maximisation
    length = 10
    form = assemble_to_target(bank, fid, thetas, target, length, model=MODEL, seed=0)
    assert isinstance(form, AssembledForm)
    assert form.items.size == length
    assert len(set(form.items.tolist())) == length

    random_info = _random_form_info(bank, fid, thetas, length, seed=123)
    # The greedy target-information form dominates the average random form at
    # every target trait point.
    assert np.all(form.achieved_info >= random_info - 1e-9)
    assert form.achieved_info.sum() > random_info.sum()


def test_assembly_meets_reachable_target_with_zero_shortfall():
    bank, fid = _bank(n_items=30)
    thetas = np.array([-0.5, 0.5])
    target = np.array([3.0, 3.0])  # modest, reachable
    form = assemble_to_target(bank, fid, thetas, target, length=12, model=MODEL, seed=0)
    assert form.shortfall == pytest.approx(0.0, abs=1e-9)
    assert np.all(form.achieved_info >= target - 1e-9)


def test_content_constraints_are_honored():
    bank, fid = _bank(n_items=30)
    thetas = np.array([0.0])
    target = np.array([100.0])
    content = np.array(["A"] * 15 + ["B"] * 15)
    form = assemble_to_target(
        bank, fid, thetas, target, length=10, model=MODEL,
        content=content, min_per_content={"B": 4}, max_per_content={"A": 5}, seed=0,
    )
    labels = content[form.items]
    assert int(np.sum(labels == "A")) <= 5
    assert int(np.sum(labels == "B")) >= 4
    assert form.content_counts.get("B", 0) >= 4


def test_exposure_cap_excludes_overexposed_items():
    bank, fid = _bank(n_items=20)
    thetas = np.array([0.0])
    target = np.array([100.0])
    # The globally most informative item at theta=0 is barred by exposure.
    info = item_information_matrix(bank, fid, thetas, model=MODEL)[0]
    hot = int(np.argmax(info))
    form = assemble_to_target(
        bank, fid, thetas, target, length=6, model=MODEL,
        exposure_counts={hot: 3}, exposure_max=3, seed=0,
    )
    assert hot not in form.items.tolist()


def test_exclude_removes_items():
    bank, fid = _bank(n_items=20)
    thetas = np.array([0.0])
    target = np.array([100.0])
    banned = np.array([0, 1, 2, 3, 4])
    form = assemble_to_target(
        bank, fid, thetas, target, length=6, model=MODEL, exclude=banned, seed=0,
    )
    assert not set(form.items.tolist()) & set(banned.tolist())


def test_assembly_is_deterministic_under_seed():
    bank, fid = _bank(n_items=30)
    thetas = np.array([-1.0, 0.0, 1.0])
    target = np.array([4.0, 5.0, 4.0])
    a = assemble_to_target(bank, fid, thetas, target, length=10, model=MODEL, seed=42)
    b = assemble_to_target(bank, fid, thetas, target, length=10, model=MODEL, seed=42)
    assert a.items.tolist() == b.items.tolist()


def test_invalid_arguments_raise():
    bank, fid = _bank(n_items=10)
    thetas = np.array([0.0])
    with pytest.raises(ValueError):
        assemble_to_target(bank, fid, thetas, np.array([1.0, 2.0]), length=3, model=MODEL)
    with pytest.raises(ValueError):
        assemble_to_target(bank, fid, thetas, np.array([1.0]), length=99, model=MODEL)
    with pytest.raises(ValueError):
        # min-content requirement impossible to satisfy within the length.
        assemble_to_target(
            bank, fid, thetas, np.array([1.0]), length=2, model=MODEL,
            content=np.array(["A"] * 10), min_per_content={"B": 1},
        )

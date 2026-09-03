"""Machine-readable GA fit capability-contract regressions."""

from __future__ import annotations

import json
from pathlib import Path
import runpy

import pytest

from fast_mlsirm.capabilities import (
    FIT_CAPABILITY_SCHEMA_VERSION,
    PRODUCTION_NUMERIC_OWNER,
    FitCapability,
    fit_capabilities,
    fit_capability_manifest,
    main as capability_main,
)
from fast_mlsirm.config import FitConfig, VALID_ESTIMATORS, VALID_MODELS


_REPOSITORY_ROOT = Path(__file__).parents[1]
_CAPABILITY_CONTRACT = _REPOSITORY_ROOT / "contracts" / "fit-capabilities-v1.json"
_CAPABILITY_MODULE = _REPOSITORY_ROOT / "python" / "fast_mlsirm" / "capabilities.py"
_GIT_ATTRIBUTES = _REPOSITORY_ROOT / ".gitattributes"


def test_fit_capabilities_cover_the_public_model_vocabulary_once() -> None:
    capabilities = fit_capabilities()
    assert FIT_CAPABILITY_SCHEMA_VERSION == "1.0"
    assert PRODUCTION_NUMERIC_OWNER == "rust"
    assert tuple(capability.model for capability in capabilities) == tuple(
        sorted(VALID_MODELS)
    )
    assert len({capability.model for capability in capabilities}) == len(capabilities)


def test_fit_capabilities_match_public_config_validation() -> None:
    by_model = {capability.model: capability.estimators for capability in fit_capabilities()}
    for model in sorted(VALID_MODELS):
        for estimator in sorted(VALID_ESTIMATORS):
            try:
                FitConfig(model=model, estimator=estimator)
            except ValueError:
                accepted = False
            else:
                accepted = True
            assert (estimator in by_model[model]) is accepted


def test_bifactor_capability_is_marginal_only() -> None:
    by_model = {capability.model: capability.estimators for capability in fit_capabilities()}
    assert by_model["BIFAC2PLM"] == ("mmle",)


@pytest.mark.parametrize("estimator", ["em", "bayes"])
def test_reserved_estimators_are_not_advertised(estimator: str) -> None:
    advertised = {
        accepted
        for capability in fit_capabilities()
        for accepted in capability.estimators
    }
    assert estimator not in advertised
    with pytest.raises(ValueError, match="estimator must be one of"):
        FitConfig(estimator=estimator)


def test_fit_capability_value_object_rejects_forged_support() -> None:
    with pytest.raises(ValueError, match="canonical model-estimator capability"):
        FitCapability(model="BIFAC2PLM", estimators=("jmle", "mmle"))


def test_fit_capabilities_do_not_expose_shared_mutable_authority() -> None:
    first = fit_capabilities()
    original_model = first[0].model
    try:
        object.__setattr__(first[0], "model", "caller-mutation")
        second = fit_capabilities()
        assert second[0].model == original_model
        assert fit_capability_manifest()["models"][0]["model"] == original_model
    finally:
        object.__setattr__(first[0], "model", original_model)


def test_fit_capability_manifest_is_json_shaped_and_fresh() -> None:
    first = fit_capability_manifest()
    assert first == {
        "schema_version": "1.0",
        "production_numeric_owner": "rust",
        "models": [
            {"model": capability.model, "estimators": list(capability.estimators)}
            for capability in fit_capabilities()
        ],
    }
    first["models"][0]["estimators"].append("caller-mutation")
    assert "caller-mutation" not in fit_capability_manifest()["models"][0]["estimators"]


def test_fit_capability_manifest_has_a_machine_readable_module_cli(capsys) -> None:
    assert capability_main([]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == fit_capability_manifest()


def test_fit_capability_module_entrypoint_is_coverage_owned(capsys) -> None:
    source = _CAPABILITY_MODULE.read_text(encoding="utf-8")
    assert "pragma: no cover" not in source

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("fast_mlsirm.capabilities", run_name="__main__")
    assert exit_info.value.code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == fit_capability_manifest()


def test_fit_capability_contract_artifact_is_exact_cli_wire_payload(capsys) -> None:
    expected = json.dumps(
        fit_capability_manifest(), sort_keys=True, separators=(",", ":")
    )
    expected_bytes = (expected + "\n").encode("utf-8")
    assert _CAPABILITY_CONTRACT.read_bytes() == expected_bytes

    assert capability_main([]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.encode("utf-8") == expected_bytes


def test_fit_capability_contract_forces_lf_checkout_bytes() -> None:
    attributes = _GIT_ATTRIBUTES.read_text(encoding="utf-8").splitlines()
    assert "contracts/fit-capabilities-v1.json text eol=lf" in attributes

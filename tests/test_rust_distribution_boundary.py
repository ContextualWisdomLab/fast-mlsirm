from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
CORE_MANIFEST = ROOT / "crates" / "mlsirm-core" / "Cargo.toml"
PYPROJECT = ROOT / "pyproject.toml"
PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "publish-pypi.yml"


def _toml(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def test_mlsirm_core_is_not_an_independent_registry_product() -> None:
    core_package = _toml(CORE_MANIFEST)["package"]
    pyproject = _toml(PYPROJECT)
    workflow = PUBLISH_WORKFLOW.read_text(encoding="utf-8")

    # The Rust core is the numerical owner behind the fast-mlsirm package, not
    # a separately released crates.io product. Keep the registry boundary
    # explicit so repository-wide Cargo credentials cannot widen it by default.
    assert core_package.get("publish") is False
    assert pyproject["project"]["name"] == "fast-mlsirm"
    assert pyproject["tool"]["maturin"]["manifest-path"] == "crates/fast-mlsirm-py/Cargo.toml"
    assert "pypa/gh-action-pypi-publish" in workflow
    assert "cargo publish" not in workflow

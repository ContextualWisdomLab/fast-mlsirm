from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
CORE_MANIFEST = ROOT / "crates" / "mlsirm-core" / "Cargo.toml"
BINDING_MANIFEST = ROOT / "crates" / "fast-mlsirm-py" / "Cargo.toml"
PYPROJECT = ROOT / "pyproject.toml"
PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "publish-pypi.yml"


def _toml(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def test_rust_crates_are_not_independent_registry_products() -> None:
    core_package = _toml(CORE_MANIFEST)["package"]
    binding_package = _toml(BINDING_MANIFEST)["package"]
    pyproject = _toml(PYPROJECT)
    workflow = PUBLISH_WORKFLOW.read_text(encoding="utf-8")

    # Both Rust crates are implementation owners behind the fast-mlsirm PyPI
    # product. Keep registry admission explicit so a Cargo credential cannot
    # accidentally turn either implementation crate into a second product.
    assert core_package.get("publish") is False
    assert binding_package.get("publish") is False
    assert pyproject["project"]["name"] == "fast-mlsirm"
    assert pyproject["tool"]["maturin"]["manifest-path"] == "crates/fast-mlsirm-py/Cargo.toml"
    assert "pypa/gh-action-pypi-publish" in workflow
    assert "cargo publish" not in workflow

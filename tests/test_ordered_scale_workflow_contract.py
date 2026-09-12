"""The installed-artifact lane must execute the test family it admits."""
from pathlib import Path


def test_ordered_scale_copies_its_complete_admitted_test_family():
    """Prevent a new admitted regression file from being omitted during isolated execution."""
    workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/ordered-scale.yml").read_text()
    assert "- tests/test_scale_codebook*.py" in workflow
    assert "cp tests/test_scale_codebook*.py /tmp/ordered-scale-tests/" in workflow

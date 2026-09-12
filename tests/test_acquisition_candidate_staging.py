from pathlib import Path

import pytest

from scripts import build_acquisition_release


def test_candidate_staging_is_byte_bound_and_origin_mutation_fails(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / "fast_mlsirm-0.9.1-py3-none-any.whl"
    sdist = dist / "fast_mlsirm-0.9.1.tar.gz"
    wheel.write_bytes(b"wheel-v1")
    sdist.write_bytes(b"sdist-v1")
    staging = tmp_path / "evidence" / "release-acceptance" / "candidate-distribution"

    staged_wheel, staged_sdist, source_states = (
        build_acquisition_release._stage_candidate_artifacts(
            wheel=wheel,
            sdist=sdist,
            staging_dir=staging,
        )
    )

    assert staged_wheel != wheel
    assert staged_sdist != sdist
    assert staged_wheel.read_bytes() == b"wheel-v1"
    assert staged_sdist.read_bytes() == b"sdist-v1"
    build_acquisition_release._require_candidate_sources_unchanged(source_states)

    wheel.write_bytes(b"wheel-v2")
    assert staged_wheel.read_bytes() == b"wheel-v1"
    with pytest.raises(RuntimeError, match="candidate artifact changed after staging"):
        build_acquisition_release._require_candidate_sources_unchanged(source_states)


def test_candidate_staging_rejects_preexisting_symlink_before_cleanup(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / "fast_mlsirm-0.9.1-py3-none-any.whl"
    sdist = dist / "fast_mlsirm-0.9.1.tar.gz"
    wheel.write_bytes(b"wheel-v1")
    sdist.write_bytes(b"sdist-v1")

    output = tmp_path / "evidence"
    output.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "must-survive.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    staging = output / "candidate-distribution"
    staging.symlink_to(external, target_is_directory=True)

    with pytest.raises(RuntimeError, match="candidate staging path"):
        build_acquisition_release._stage_candidate_artifacts(
            wheel=wheel,
            sdist=sdist,
            staging_dir=staging,
        )

    assert staging.is_symlink()
    assert sentinel.read_text(encoding="utf-8") == "preserve"

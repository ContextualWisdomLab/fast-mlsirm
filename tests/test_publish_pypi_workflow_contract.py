from __future__ import annotations

from pathlib import Path
import gzip
import hashlib
import io
import os
import re
import subprocess
import sys
import tarfile
import textwrap
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-pypi.yml"
RELEASE_TAG_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-tag.yml"
PYPI_PUBLISH_SHA = "dc37677b2e1c63e2034f94d8a5b11f265b73ba33"


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _release_tag_workflow_text() -> str:
    return RELEASE_TAG_WORKFLOW.read_text(encoding="utf-8")


def _job_block(text: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        text,
    )
    assert match is not None, f"missing {name!r} job"
    return match.group(0)


def test_release_builds_are_bound_to_the_reviewed_source_commit() -> None:
    text = _workflow_text()
    verify = _job_block(text, "verify-release")
    sdist = _job_block(text, "sdist")
    wheels = _job_block(text, "wheels")

    assert "      release_commit:\n" in text
    assert re.search(
        r"(?m)^      release_commit:\n(?:        .*\n)*?        required: true$",
        text,
    )

    for job in (verify, sdist, wheels):
        assert "ref: ${{ inputs.release_commit }}" in job
        assert "persist-credentials: false" in job
        assert "ref: ${{ inputs.release_tag }}" not in job

    assert "fetch-depth: 0" in verify
    assert 'RELEASE_TAG: ${{ inputs.release_tag }}' in verify
    assert 'RELEASE_COMMIT: ${{ inputs.release_commit }}' in verify
    assert "release_commit must be a canonical 40-character lowercase SHA-1" in verify
    assert 'git rev-parse HEAD' in verify
    assert 'git rev-parse "$RELEASE_TAG^{commit}"' in verify
    assert "checked-out release source does not match release_commit" in verify
    assert "release tag does not target release_commit" in verify
    assert 'tomllib.load' in verify or 'tomllib.loads' in verify
    assert 'f"v{project[\'version\']}"' in verify
    assert text.count("maturin-version: v1.14.1") == 4


def test_wheels_cover_supported_cpython_versions_on_every_platform() -> None:
    wheels = _job_block(_workflow_text(), "wheels")

    # The extension is not built with PyO3 abi3, so a CPython 3.12 wheel cannot
    # satisfy 3.13/3.14 callers. Each release platform must build all currently
    # evidenced supported CPython versions instead of forcing newer callers to
    # compile the sdist with a local Rust toolchain.
    for version in ("3.12", "3.13", "3.14"):
        assert wheels.count(f'python-version: "{version}"') == 4
        assert wheels.count(f"interpreter: python{version}") == 2

    assert wheels.count("interpreter: python\n") == 6
    assert "python-version: ${{ matrix.python-version }}" in wheels
    assert "args: --release --out dist -i ${{ matrix.interpreter }}" in wheels
    assert "name: dist-wheel-${{ matrix.target }}-py${{ matrix.python-version }}" in wheels


def test_release_tag_workflow_explicitly_dispatches_package_publish() -> None:
    publish_text = _workflow_text()
    release_text = _release_tag_workflow_text()
    release_job = _job_block(release_text, "publish-release-tag")
    verify = _job_block(publish_text, "verify-release")

    # GITHUB_TOKEN-created release events do not recursively start ordinary
    # event-triggered workflows. Package publication therefore uses the one
    # supported recursive trigger. The control-plane workflow comes from the
    # protected default branch while artifact source identity is an immutable
    # explicit commit, so an older release tag cannot select an outdated
    # publication workflow definition. The dispatch also carries the exact
    # release-tag run commit so a moving default branch cannot silently select a
    # different publication control plane between verification and dispatch.
    assert "  workflow_dispatch:\n" in publish_text
    assert "      release_tag:\n" in publish_text
    assert "      release_commit:\n" in publish_text
    assert "      control_plane_commit:\n" in publish_text
    assert publish_text.count("        required: true\n") >= 3
    assert "types: [published]" not in publish_text

    assert 'DEFAULT_BRANCH: ${{ github.event.repository.default_branch }}' in verify
    assert 'PUBLISH_REF: ${{ github.ref }}' in verify
    assert 'CONTROL_PLANE_COMMIT: ${{ inputs.control_plane_commit }}' in verify
    assert 'CONTROL_PLANE_SHA: ${{ github.sha }}' in verify
    assert 'expected_ref="refs/heads/$DEFAULT_BRANCH"' in verify
    assert 'if [ "$PUBLISH_REF" != "$expected_ref" ]' in verify
    assert 'if [ "$CONTROL_PLANE_SHA" != "$CONTROL_PLANE_COMMIT" ]' in verify
    assert "publication control plane moved after release verification" in verify

    assert "permissions:\n      contents: write\n      actions: write" in release_job
    assert "gh workflow run publish-pypi.yml" in release_job
    assert 'DEFAULT_BRANCH: ${{ github.event.repository.default_branch }}' in release_job
    assert 'git rev-parse "origin/$DEFAULT_BRANCH"' in release_job
    assert "release commit must be an ancestor of the current default branch" in release_job
    assert '--ref "$DEFAULT_BRANCH"' in release_job
    assert '--ref "v$RELEASE_VERSION"' not in release_job
    assert '-f release_tag="v$RELEASE_VERSION"' in release_job
    assert '-f release_commit="$RELEASE_COMMIT"' in release_job
    assert '-f control_plane_commit="$CONTROL_PLANE_COMMIT"' in release_job
    assert 'RELEASE_COMMIT: ${{ inputs.release_commit }}' in release_job
    assert release_job.index('gh release create "v$RELEASE_VERSION"') < release_job.index(
        "gh workflow run publish-pypi.yml"
    )


def test_release_asset_write_is_isolated_from_pypi_credentials() -> None:
    text = _workflow_text()
    assets = _job_block(text, "release-assets")
    publish = _job_block(text, "publish-pypi")

    assert "permissions:\n      contents: write" in assets
    assert "GH_TOKEN: ${{ github.token }}" in assets
    assert "RELEASE_TAG: ${{ inputs.release_tag }}" in assets
    assert 'gh release upload "$RELEASE_TAG"' in assets
    assert "--clobber" not in assets
    assert "secrets.PIPY_TOKEN" not in assets
    assert "release $RELEASE_TAG is immutable; skipping GitHub asset upload" in assets
    assert ".immutable // false" in assets

    assert "environment: pypi" in publish
    assert "permissions:\n      contents: read" in publish
    assert "gh release upload" not in publish
    assert "contents: write" not in publish


def test_pypi_publish_uses_a_pinned_package_owned_uploader() -> None:
    text = _workflow_text()
    publish = _job_block(text, "publish-pypi")
    sdist = _job_block(text, "sdist")

    assert "python -m pip install" not in text
    assert "python -m twine upload" not in text
    assert "TWINE_USERNAME" not in text
    assert f"uses: pypa/gh-action-pypi-publish@{PYPI_PUBLISH_SHA}" in publish
    assert "password: ${{ secrets.PIPY_TOKEN }}" in publish
    assert "attestations: false" in publish
    assert "skip-existing: true" in publish
    assert "Ensure LICENSE is present in the sdist" in sdist
    assert 'license_member = f"{root}/LICENSE"' in sdist


def test_pypi_publish_can_recover_independently_of_immutable_asset_upload() -> None:
    text = _workflow_text()
    assets = _job_block(text, "release-assets")
    publish = _job_block(text, "publish-pypi")

    # Release assets and PyPI are two independent publication sinks fed by the
    # same verified build artifacts. Immutable releases skip asset upload
    # without failing the job so a previously failed PyPI publication can be
    # retried to a green overall run.
    assert "needs: [sdist, wheels, reproducibility-record]" in assets
    assert "needs: [sdist, wheels, reproducibility-record]" in publish
    assert "release-assets" not in publish.split("needs:", 1)[1].split("\n", 1)[0]
    assert "skipping GitHub asset upload" in assets
    assert "skip-existing: true" in publish


def _step_python(job: str, step_name: str) -> str:
    step = job.split(f"- name: {step_name}\n", 1)[1]
    body = re.split(r"python3? - <<'PY'\n", step, maxsplit=1)[1].split("\n          PY\n", 1)[0]
    return textwrap.dedent(body)


def test_every_release_build_is_reproducible_from_the_release_commit_clock() -> None:
    text = _workflow_text()
    verify = _job_block(text, "verify-release")
    sdist = _job_block(text, "sdist")
    wheels = _job_block(text, "wheels")
    record = _job_block(text, "reproducibility-record")
    assets = _job_block(text, "release-assets")
    publish = _job_block(text, "publish-pypi")

    # SOURCE_DATE_EPOCH is the committer time of the exact release commit,
    # derived once and fanned out, never the wall clock of the build runner.
    assert 'epoch="$(git log -1 --format=%ct "$RELEASE_COMMIT")"' in verify
    assert "source_date_epoch: ${{ steps.source-date-epoch.outputs.value }}" in verify
    for job in (verify, sdist, wheels, record):
        assert "date +%s" not in job
    epoch_env = "SOURCE_DATE_EPOCH: ${{ needs.verify-release.outputs.source_date_epoch }}"
    assert f"    env:\n      {epoch_env}\n" in sdist
    assert f"    env:\n      {epoch_env}\n" in wheels

    # maturin-action does not forward SOURCE_DATE_EPOCH into the manylinux
    # container on its own, so every wheel build step passes it explicitly.
    builds = re.findall(r"(?ms)^      - name: (?:Build|Rebuild) wheel.*?(?=^      - )", wheels)
    assert len(builds) == 2
    for build in builds:
        assert "docker-options: -e SOURCE_DATE_EPOCH\n" in build
        assert "if:" not in build

    # Every publishable artifact (12 wheels + sdist) is rebuilt from a clean
    # target and compared; no leg may opt out of the double build.
    assert "verify-reproducible" not in wheels
    assert "--out dist-rebuild --target-dir target-rebuild" in builds[1]
    assert "- name: Rebuild sdist for byte-reproducibility check" in sdist
    assert "args: --out dist-rebuild" in sdist
    compare_wheel = _step_python(wheels, "Compare double-build wheel digests and record them")
    compare_sdist = _step_python(sdist, "Compare double-build sdist digests and record them")
    assert compare_wheel == compare_sdist
    assert 'second = digest("dist-rebuild")' in compare_wheel
    assert "if first != second:" in compare_wheel
    assert "is not byte-reproducible" in compare_wheel
    assert "clean-target-repeat-same-env" in compare_wheel
    assert "could not record the manylinux container digest" in compare_wheel
    assert '"--digests"' in compare_wheel
    assert "name: repro-rebuild-${{ matrix.target }}-py${{ matrix.python-version }}" in wheels
    assert "name: repro-rebuild-sdist" in sdist

    # The record fails closed: every downloaded publishable artifact needs a
    # byte-verified row whose digests match the published bytes, and both
    # publication sinks depend on it.
    gate = _step_python(record, "Require a byte-verified row for every publishable artifact")
    assert "pattern: dist-*" in record
    assert "pattern: repro-digest-*" in record
    assert 'failures.append(f"{name}: no byte-verification row")' in gate
    assert 'row["sha256"] != sha or row["rebuild_sha256"] != sha' in gate
    assert "if failures:" in gate
    assert "len(published) != 13" in gate
    assert "NOT independent-environment or" in gate
    assert "name: reproducibility-record" in record
    for sink in (assets, publish):
        assert "needs: [sdist, wheels, reproducibility-record]" in sink
        assert "always()" not in sink
        assert "|| true" not in sink
    assert "|| true" not in record

    # SOURCE_DATE_EPOCH alone is not enough: with the default 16 codegen units
    # fresh builds of the binding crate differ in `.llvm.<hash>` symbol
    # suffixes, so the shipped release profile pins a single codegen unit.
    binding = (REPO_ROOT / "crates" / "fast-mlsirm-py" / "Cargo.toml").read_text(encoding="utf-8")
    assert re.search(r"(?m)^\[profile\.release\]\n(?:(?!\[).*\n)*?codegen-units = 1$", binding)


def _sdist_without_license(path: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as writer:
        for name, payload in (("fast_mlsirm-0.0.0/PKG-INFO", b"Metadata-Version: 2.4\n"),
                              ("fast_mlsirm-0.0.0/pyproject.toml", b"[project]\n")):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 1_700_000_000
            writer.addfile(info, io.BytesIO(payload))
    path.write_bytes(gzip.compress(buffer.getvalue(), mtime=1_700_000_000))


def test_historical_sdist_license_injection_is_byte_reproducible(tmp_path: Path) -> None:
    script = _step_python(_job_block(_workflow_text(), "sdist"), "Ensure LICENSE is present in the sdist")
    env = {**os.environ, "SOURCE_DATE_EPOCH": "1790068752"}
    outputs = []
    for attempt in range(2):
        work = tmp_path / f"run{attempt}"
        for dist in ("dist", "dist-rebuild"):
            (work / dist).mkdir(parents=True)
            _sdist_without_license(work / dist / "fast_mlsirm-0.0.0.tar.gz")
        (work / "LICENSE").write_bytes((REPO_ROOT / "LICENSE").read_bytes())
        if attempt:
            time.sleep(1.1)  # a wall-clock gzip mtime would now differ
        subprocess.run([sys.executable, "-c", script], cwd=work, env=env, check=True)
        for dist in ("dist", "dist-rebuild"):
            archive = work / dist / "fast_mlsirm-0.0.0.tar.gz"
            with tarfile.open(archive, "r:gz") as reader:
                assert reader.extractfile("fast_mlsirm-0.0.0/LICENSE").read() == (REPO_ROOT / "LICENSE").read_bytes()
            outputs.append(hashlib.sha256(archive.read_bytes()).hexdigest())
    assert len(set(outputs)) == 1, outputs

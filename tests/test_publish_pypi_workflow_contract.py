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
    assert 'git rev-parse -q --verify "$RELEASE_TAG^{commit}"' in verify
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


def test_direct_publish_ancestry_guard_against_real_git_history(tmp_path: Path) -> None:
    verify = _job_block(_workflow_text(), "verify-release")
    name = "Verify release source ancestry for direct publication"
    step = verify.split(f"- name: {name}\n", 1)[1].split("\n      - ", 1)[0]
    script = textwrap.dedent(step.split("        run: |\n", 1)[1])
    assert verify.index("fetch-depth: 0") < verify.index(name)
    assert verify.index(name) < verify.index("Require release tag and source commit")
    assert 'CONTROL_PLANE_COMMIT: ${{ inputs.control_plane_commit }}' in step
    assert 'shell: bash --noprofile --norc -e -o pipefail {0}' in step
    assert "fetch-tags: true" in verify
    # Execute only the actual guard; no workflow tag/sign/publish command runs.
    repo = tmp_path / "git-history"
    repo.mkdir()
    env = {**os.environ, "GIT_AUTHOR_NAME": "fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
           "GIT_COMMITTER_NAME": "fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=repo, env=env, text=True).strip()

    git("init", "-q")
    git("commit", "--allow-empty", "-qm", "root")
    root = git("rev-parse", "HEAD")
    git("commit", "--allow-empty", "-qm", "control")
    control = git("rev-parse", "HEAD")
    git("checkout", "--detach", "-q", root)
    git("commit", "--allow-empty", "-qm", "sibling")
    sibling = git("rev-parse", "HEAD")
    absent = "0" * 40
    cases = [(control, control, control, True), (root, root, control, True),
             (sibling, sibling, control, False), (control, control, root, False),
             (root, root, absent, False), (root, absent, control, False),
             (root, control, control, False)]
    for index, (checkout, release, caller, allowed) in enumerate(cases):
        git("checkout", "--detach", "-q", checkout)
        marker = tmp_path / f"downstream-{index}"
        result = subprocess.run(
            ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c",
             script + '\nprintf reached > "$DOWNSTREAM_MARKER"\n'],
            cwd=repo, env={**env, "RELEASE_COMMIT": release, "CONTROL_PLANE_COMMIT": caller,
                           "DOWNSTREAM_MARKER": str(marker)}, capture_output=True, text=True,
        )
        assert (result.returncode == 0) is allowed, result.stderr
        assert marker.exists() is allowed
        assert git("tag", "--list") == ""


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

    assert "permissions:\n      contents: read\n      actions: write" in release_job
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
    # R5a: release-tag.yml verifies and dispatches; it never creates the tag or release.
    assert "gh release create" not in release_job
    assert "/git/refs" not in release_job


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
    assert "needs: [sdist, wheels, reproducibility-record, release-admission, create-tag-and-release]" in assets
    assert "needs: [sdist, wheels, reproducibility-record, release-admission, create-tag-and-release]" in publish
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
    assert '"docker", "images"' not in compare_wheel  # never scan unrelated cached images
    assert "container: ${{ matrix.container }}" in builds[0]
    assert "container: ${{ matrix.container }}" in builds[1]
    containers = re.findall(r"target: (\S+)\n(?:            .*\n)*?            container: \"([^\"]*)\"", wheels)
    assert len(containers) == 12
    for target, image in containers:
        if "linux" in target:
            assert re.fullmatch(r"quay\.io/pypa/manylinux2014_(?:x86_64|aarch64)@sha256:[0-9a-f]{64}", image), image
        else:
            assert image == "", (target, image)
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
        assert "needs: [sdist, wheels, reproducibility-record, release-admission, create-tag-and-release]" in sink
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


_PINNED = "quay.io/pypa/manylinux2014_x86_64@sha256:" + "a" * 64
_UNRELATED = "quay.io/pypa/manylinux2014_x86_64@sha256:" + "b" * 64


def _run_compare(tmp_path: Path, name: str, *, leg: str, runner_os: str, container: str,
                 docker_digests: list[str] | None, runner_identity: bool = True) -> subprocess.CompletedProcess:
    script = _step_python(_job_block(_workflow_text(), "wheels"), "Compare double-build wheel digests and record them")
    work = tmp_path / name
    for dist in ("dist", "dist-rebuild"):
        (work / dist).mkdir(parents=True)
        (work / dist / "pkg-0-py3-none-any.whl").write_bytes(b"same bytes")
    stub = work / "bin"
    stub.mkdir()
    docker = stub / "docker"
    # Stub runner docker: `image inspect <ref>` succeeds only for images in docker_digests.
    docker.write_text(
        "#!/bin/sh\n"
        f"KNOWN='{' '.join(docker_digests or [])}'\n"
        'for ref; do :; done\n'  # the image reference is the last argument
        'for d in $KNOWN; do [ "$d" = "$ref" ] && { printf \'["%s"]\\n\' "$ref"; exit 0; }; done\n'
        'echo "Error: No such image: $ref" >&2; exit 1\n'
    )
    docker.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in {"ImageOS", "ImageVersion", "RUNNER_ARCH"}}
    env.update({"PATH": f"{stub}{os.pathsep}{env['PATH']}", "SOURCE_DATE_EPOCH": "1790068752", "LEG": leg,
                "ARTIFACT_GLOB": "*.whl", "RUNNER_OS": runner_os, "CONTAINER_IMAGE": container})
    if runner_identity:
        env.update({"ImageOS": "ubuntu24", "ImageVersion": "20260920.1", "RUNNER_ARCH": "X64"})
    return subprocess.run([sys.executable, "-c", script], cwd=work, env=env, capture_output=True, text=True)


def test_build_env_provenance_is_bound_to_the_invocation_and_fails_closed(tmp_path: Path) -> None:
    linux = "x86_64-unknown-linux-gnu-py3.12"
    ok = _run_compare(tmp_path, "linux-ok", leg=linux, runner_os="Linux", container=_PINNED,
                      docker_digests=[_PINNED, _UNRELATED])
    assert ok.returncode == 0, ok.stderr
    row = (tmp_path / "linux-ok" / "repro-digest" / f"{linux}.tsv").read_text().rstrip("\n").split("\t")
    assert row[-1] == f"container:{_PINNED}"

    # Negative shape 1: a Linux leg with only an unrelated cached image, or no bound digest.
    cached_only = _run_compare(tmp_path, "linux-cached-only", leg=linux, runner_os="Linux", container=_PINNED,
                               docker_digests=[_UNRELATED])
    assert cached_only.returncode != 0 and "is not present on this runner" in cached_only.stderr
    unbound = _run_compare(tmp_path, "linux-unbound", leg=linux, runner_os="Linux", container="",
                           docker_digests=[_UNRELATED])
    assert unbound.returncode != 0 and "no digest-pinned build container" in unbound.stderr
    tag_only = _run_compare(tmp_path, "linux-tag", leg=linux, runner_os="Linux",
                            container="quay.io/pypa/manylinux2014_x86_64:latest", docker_digests=[_UNRELATED])
    assert tag_only.returncode != 0 and "no digest-pinned build container" in tag_only.stderr

    # Negative shape 2: a container-less row (macOS/Windows/sdist) with a container digest or no runner identity.
    for leg, runner_os in (("universal2-apple-darwin-py3.12", "macOS"), ("x86_64-pc-windows-msvc-py3.12", "Windows"),
                           ("sdist", "Linux")):
        carries = _run_compare(tmp_path, f"{runner_os}-{leg}-carries", leg=leg, runner_os=runner_os,
                               container=_PINNED, docker_digests=[_PINNED])
        assert carries.returncode != 0 and "carries container provenance" in carries.stderr, (leg, carries.stderr)
        anonymous = _run_compare(tmp_path, f"{runner_os}-{leg}-anon", leg=leg, runner_os=runner_os, container="",
                                 docker_digests=[_PINNED], runner_identity=False)
        assert anonymous.returncode != 0 and "missing runner identity" in anonymous.stderr, (leg, anonymous.stderr)
        good = _run_compare(tmp_path, f"{runner_os}-{leg}-ok", leg=leg, runner_os=runner_os, container="",
                            docker_digests=[_PINNED])
        assert good.returncode == 0, (leg, good.stderr)
        row = (tmp_path / f"{runner_os}-{leg}-ok" / "repro-digest" / f"{leg}.tsv").read_text().rstrip("\n").split("\t")
        assert row[-1] == f"runner:ubuntu24/20260920.1/{runner_os}/X64"


_RELEASE_COMMIT = "c" * 40
_RELEASE_TAG = "v1.2.3"
_RUN_ID = 424242


def _needs(job: str) -> set[str]:
    match = re.search(r"(?m)^    needs: (?:\[(?P<many>[^\]]*)\]|(?P<one>\S+))$", job)
    if match is None:
        return set()
    if match.group("one"):
        return {match.group("one")}
    return {name.strip() for name in match.group("many").split(",")}


def _requires(text: str, job: str, dependency: str) -> bool:
    pending, seen = [job], set()
    while pending:
        current = pending.pop()
        for parent in _needs(_job_block(text, current)):
            if parent == dependency:
                return True
            if parent not in seen:
                seen.add(parent)
                pending.append(parent)
    return False


def _expected_legs() -> list[str]:
    admission = _job_block(_workflow_text(), "release-admission")
    match = re.search(r'EXPECTED_WHEEL_LEGS: "([^"]+)"', admission)
    assert match is not None
    return match.group(1).split()


def test_tag_and_release_are_created_only_after_release_admission() -> None:
    text = _workflow_text()
    release_job = _job_block(_release_tag_workflow_text(), "publish-release-tag")

    # R5a: only the post-admission job creates the immutable tag and release.
    creator = _job_block(text, "create-tag-and-release")
    assert '"$GITHUB_API_URL/repos/$GITHUB_REPOSITORY/git/refs"' in creator
    assert 'gh release create "v$RELEASE_VERSION"' in creator
    for name in ("verify-release", "sdist", "wheels", "reproducibility-record", "release-admission",
                 "release-assets", "publish-pypi"):
        job = _job_block(text, name)
        assert "gh release create" not in job and "/git/refs\"" not in job, name
    assert "gh release create" not in release_job and "/git/refs" not in release_job
    assert "contents: write" not in release_job

    assert _requires(text, "create-tag-and-release", "release-admission")
    assert _requires(text, "release-admission", "reproducibility-record")
    assert _requires(text, "release-admission", "verify-release")
    for sink in ("release-assets", "publish-pypi"):
        assert _requires(text, sink, "create-tag-and-release"), sink
        assert _requires(text, sink, "release-admission"), sink
    for name in ("release-admission", "create-tag-and-release", "release-assets", "publish-pypi"):
        job = _job_block(text, name)
        assert "always()" not in job and "continue-on-error" not in job and "|| true" not in job, name
        assert not re.search(r"(?m)^    if:", job), name

    # Condition 4: the admitted wheel legs are exactly the build matrix.
    wheels = _job_block(text, "wheels")
    matrix = re.findall(r"target: (\S+)\n(?:            .*\n)*?            python-version: \"([^\"]+)\"", wheels)
    assert sorted(f"{target}-py{version}" for target, version in matrix) == sorted(_expected_legs())


def _admission_fixture(root: Path) -> dict:
    import zipfile
    legs = _expected_legs()
    platforms = {"x86_64-unknown-linux-gnu": "manylinux2014_x86_64",
                 "aarch64-unknown-linux-gnu": "manylinux2014_aarch64",
                 "universal2-apple-darwin": "macosx_11_0_universal2",
                 "x86_64-pc-windows-msvc": "win_amd64"}
    files = {}
    for leg in legs:
        target, version = leg.rsplit("-py", 1)
        cp = "cp" + version.replace(".", "")
        files[leg] = f"pkg-1.2.3-{cp}-{cp}-{platforms[target]}.whl"
    files["sdist"] = "pkg-1.2.3.tar.gz"
    payload = {leg: f"bytes of {name}".encode() for leg, name in files.items()}
    for leg in legs:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            tag = "-".join(files[leg][:-4].rsplit("-", 3)[1:])
            archive.writestr("pkg-1.2.3.dist-info/WHEEL", f"Wheel-Version: 1.0\nTag: {tag}\n")
        payload[leg] = buffer.getvalue()
    sha = {leg: hashlib.sha256(data).hexdigest() for leg, data in payload.items()}
    (root / "dist").mkdir(parents=True)
    for leg, name in files.items():
        (root / "dist" / name).write_bytes(payload[leg])
    (root / "record").mkdir()
    rows = "".join(
        f"{leg}\ttrue\tclean-target-repeat-same-env\t{sha[leg]}\t{sha[leg]}\t{files[leg]}\trunner:x\n"
        for leg in sorted(files)
    )
    (root / "record" / "reproducibility-record.tsv").write_text(
        f"# release {_RELEASE_TAG} @ {_RELEASE_COMMIT}, SOURCE_DATE_EPOCH=1\n"
        "target\tbyte_verified\tverification\tsha256\trebuild_sha256\tfile\tbuild_env\n" + rows
    )
    artifacts = [f"dist-wheel-{leg}" for leg in legs] + ["dist-sdist", "reproducibility-record"]
    for leg in legs:
        name = f"license-evidence-{leg}"
        artifacts.append(name)
        bundle = root / "evidence" / name
        bundle.mkdir(parents=True)
        members = {files[leg]: payload[leg], files["sdist"]: payload["sdist"],
                   f"{files[leg]}.cdx.json": b"{}", f"{files['sdist']}.cdx.json": b"{}"}
        identity = {"source_repository": "owner/repo", "source_sha": _RELEASE_COMMIT, "evidence_artifact_name": name,
                    "artifacts": {"wheel": {"filename": files[leg], "sha256": sha[leg]},
                                  "sdist": {"filename": files["sdist"], "sha256": sha["sdist"]}}}
        members["source-identity.json"] = (__import__("json").dumps(identity) + "\n").encode()
        for member, data in members.items():
            (bundle / member).write_bytes(data)
        (bundle / "checksums.sha256").write_text(
            "".join(f"{hashlib.sha256(members[m]).hexdigest()}  {m}\n" for m in sorted(members))
        )
    listing = [{"id": index, "name": name, "workflow_run": {"id": _RUN_ID}, "expired": False, "digest": "sha256:" + "d" * 64}
               for index, name in enumerate(artifacts, 1)]
    return {"legs": legs, "files": files, "listing": listing}


def _run_admission(root: Path, step: str, listing: list[dict] | None = None) -> subprocess.CompletedProcess:
    import json
    import runpy
    import zipfile

    if listing is not None:
        (root / "run-artifacts.jsonl").write_text("".join(json.dumps(item) + "\n" for item in listing))
    if step == _BYTES_STEP and not (root / "downloaded").exists():
        # Exercise the actual transport with synthetic ZIPs, not a fake receipt.
        artifacts = [("reproducibility-record", list((root / "record").iterdir()))]
        record = (root / "record/reproducibility-record.tsv").read_text().splitlines()
        for line in record[2:]:
            fields = line.split("\t")
            name = "dist-sdist" if fields[0] == "sdist" else "dist-wheel-" + fields[0]
            path = root / "dist" / fields[5]
            # Missing files still reach the admission check as an empty mismatch.
            artifacts.append((name, [path] if path.exists() else []))
        artifacts += [(p.name, list(p.iterdir())) for p in (root / "evidence").iterdir()]
        recorded_names = {line.split("\t")[5] for line in record[2:]}
        extras = [p for p in (root / "dist").iterdir() if p.name not in recorded_names]
        for name, paths in artifacts:
            if name == "dist-sdist":
                paths.extend(extras)
        selected, archives = [], {}
        # A duplicated target is diagnosed by record admission, not this test's
        # archive constructor; no metadata or record content is repaired.
        seen = set()
        for name, paths in artifacts:
            if name in seen:
                continue
            seen.add(name)
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as z:
                for path in paths:
                    if path.is_file():
                        z.writestr(path.name, path.read_bytes())
                if not paths:
                    z.writestr("missing-member", b"missing")
            index = len(selected) + 1
            archives[index] = buffer.getvalue()
            selected.append({"id": index, "name": name, "digest": "sha256:" + hashlib.sha256(buffer.getvalue()).hexdigest()})
        module = runpy.run_path(str(REPO_ROOT / "scripts/ci/release_artifact_transport.py"))
        receipt = module["materialize"](selected, "owner/repo", root / "downloaded", lambda repo, index, output: output.write(archives[index]))
        (root / "selected-artifacts.json").write_text(json.dumps(selected))
        (root / "transport-receipt.json").write_text(json.dumps(receipt))
        (root / "trusted-control").symlink_to(REPO_ROOT, target_is_directory=True)
        if os.environ.get("CWL_GATE_FIXTURE_ROOT"):
            (root / "trusted-gate").symlink_to(os.environ["CWL_GATE_FIXTURE_ROOT"], target_is_directory=True)
    script = _step_python(_job_block(_workflow_text(), "release-admission"), step)
    env = {**os.environ, "EXPECTED_WHEEL_LEGS": " ".join(_expected_legs()), "RUN_ID": str(_RUN_ID),
           "RELEASE_COMMIT": _RELEASE_COMMIT, "RELEASE_TAG": _RELEASE_TAG, "REPOSITORY": "owner/repo"}
    return subprocess.run([sys.executable, "-c", script], cwd=root, env=env, capture_output=True, text=True)


_SET_STEP = "Require the exact same-run artifact set"
_BYTES_STEP = "Admit exactly the verified bytes of release_commit"


def test_release_admission_admits_only_verified_same_run_bytes(tmp_path: Path) -> None:
    import json

    fixture = _admission_fixture(tmp_path / "ok")
    ok_set = _run_admission(tmp_path / "ok", _SET_STEP, fixture["listing"])
    assert ok_set.returncode == 0, ok_set.stderr
    ok_bytes = _run_admission(tmp_path / "ok", _BYTES_STEP)
    # This historical fixture has empty SBOMs and no full gate authorization.
    # It must never be counted as successful release admission.
    assert ok_bytes.returncode != 0 and "central sealed handoff rejected" in ok_bytes.stderr
    assert not (tmp_path / "ok" / "admitted-manifest.tsv").exists()

    def refuse_set(name: str, mutate, expected: str) -> None:
        root = tmp_path / name
        listing = _admission_fixture(root)["listing"]
        result = _run_admission(root, _SET_STEP, mutate(listing))
        assert result.returncode != 0 and expected in result.stderr, (name, result.stderr)

    # Today no licence gate is wired: zero evidence bundles must refuse (fail-closed B1/B2).
    refuse_set("no-evidence", lambda l: [a for a in l if not a["name"].startswith("license-evidence-")],
               "licence evidence missing for 12 of 12 wheel legs")
    refuse_set("eleven-evidence", lambda l: l[:-1], "licence evidence missing for 1 of 12 wheel legs")
    refuse_set("other-run", lambda l: [dict(a, workflow_run={"id": 1}) if a["name"] == "dist-sdist" else a for a in l],
               "dist-sdist: not produced by run")
    refuse_set("expired", lambda l: [dict(a, expired=True) if a["name"] == "reproducibility-record" else a for a in l],
               "reproducibility-record: expired")
    refuse_set("no-digest", lambda l: [dict(a, digest=None) if a["name"].startswith("dist-wheel-") else a for a in l],
               "no sha256 artifact digest")
    refuse_set("duplicate", lambda l: l + [l[0]], "duplicate artifact name")
    refuse_set("extra-dist", lambda l: l + [dict(l[0], name="dist-wheel-extra")], "unexpected publishable")
    refuse_set("missing-dist", lambda l: [a for a in l if a["name"] != "dist-sdist"], "dist-sdist: not uploaded")
    refuse_set("missing-ids", lambda l: [{k: v for k, v in a.items() if k != "id"} for a in l], "immutable artifact ID")
    refuse_set("duplicate-ids", lambda l: [dict(a, id=1) for a in l], "immutable artifact ID")

    def refuse_bytes(name: str, mutate, expected: str) -> None:
        root = tmp_path / name
        fixture = _admission_fixture(root)
        mutate(root, fixture)
        result = _run_admission(root, _BYTES_STEP)
        assert result.returncode != 0 and expected in result.stderr, (name, result.stderr)

    first = _expected_legs()[0]

    def rewrite_identity(root: Path, leg: str, change) -> None:
        bundle = root / "evidence" / f"license-evidence-{leg}"
        identity = json.loads((bundle / "source-identity.json").read_text())
        change(identity)
        (bundle / "source-identity.json").write_text(json.dumps(identity) + "\n")
        members = {p.name: p.read_bytes() for p in bundle.iterdir() if p.name != "checksums.sha256"}
        (bundle / "checksums.sha256").write_text(
            "".join(f"{hashlib.sha256(members[m]).hexdigest()}  {m}\n" for m in sorted(members))
        )

    refuse_bytes("source-sha", lambda r, f: rewrite_identity(r, first, lambda i: i.update(source_sha="e" * 40)),
                 "source_sha=")
    refuse_bytes("wheel-sha", lambda r, f: rewrite_identity(
        r, first, lambda i: i["artifacts"]["wheel"].update(sha256="0" * 64)), "sealed wheel is not the recorded")
    refuse_bytes("sdist-seal", lambda r, f: rewrite_identity(
        r, first, lambda i: i["artifacts"]["sdist"].update(sha256="1" * 64)), "the 12 sdist seals diverge")
    refuse_bytes("tampered-member", lambda r, f: (r / "evidence" / f"license-evidence-{first}" / "source-identity.json")
                 .write_text("{}\n"), "checksums.sha256 does not match")
    refuse_bytes("extra-dist-file", lambda r, f: (r / "dist" / "extra.whl").write_bytes(b"x"),
                 "distribution files differ from the record")
    refuse_bytes("changed-dist-bytes", lambda r, f: (r / "dist" / f["files"][first]).write_bytes(b"other"),
                 "distribution files differ from the record")

    def record_mutation(root: Path, old: str, new: str) -> None:
        path = root / "record" / "reproducibility-record.tsv"
        path.write_text(path.read_text().replace(old, new, 1))

    refuse_bytes("record-commit", lambda r, f: record_mutation(r, _RELEASE_COMMIT, "f" * 40),
                 "reproducibility record is not bound")
    refuse_bytes("record-unverified", lambda r, f: record_mutation(r, "\ttrue\t", "\tfalse\t"),
                 "record row is not byte-verified")

    def identity_only(root: Path, fixture: dict) -> None:
        for bundle in (root / "evidence").iterdir():
            for path in bundle.iterdir():
                if path.name not in ("source-identity.json", "checksums.sha256"):
                    path.unlink()  # only synthetic members made by this test
            identity_path = bundle / "source-identity.json"
            (bundle / "checksums.sha256").write_text(
                f"{hashlib.sha256(identity_path.read_bytes()).hexdigest()}  source-identity.json\n")

    refuse_bytes("identity-only", identity_only, "central sealed handoff rejected")

    # An untrusted JSON verdict is not a trusted gate-success receipt.
    refuse_bytes("self-pass", lambda r, f: rewrite_identity(
        r, first, lambda i: i.update(result="PASS", stage="full")), "central sealed handoff rejected")


def test_publication_sinks_consume_only_the_admitted_bytes(tmp_path: Path) -> None:
    text = _workflow_text()
    for sink in ("release-assets", "publish-pypi"):
        job = _job_block(text, sink)
        step = job.split("- name: Require the admitted bytes\n", 1)[1].split("run: |\n", 1)[1]
        script = textwrap.dedent(step.split("\n      - ", 1)[0])
        for name, tamper in (("ok", None), ("changed", b"tampered"), ("extra", "extra")):
            root = tmp_path / f"{sink}-{name}"
            (root / "dist").mkdir(parents=True)
            (root / "admission").mkdir()
            (root / "dist" / "a.whl").write_bytes(b"a")
            (root / "admission" / "admitted-manifest.tsv").write_text(
                f"{hashlib.sha256(b'a').hexdigest()}  a.whl\n")
            if tamper == b"tampered":
                (root / "dist" / "a.whl").write_bytes(tamper)
            elif tamper == "extra":
                (root / "dist" / "b.whl").write_bytes(b"b")
            result = subprocess.run(["bash", "-c", script], cwd=root, capture_output=True, text=True)
            if tamper is None:
                assert result.returncode == 0, (sink, result.stderr)
            else:
                assert result.returncode != 0 and "not the admitted bytes" in result.stderr, (sink, name)


def test_admission_rejects_record_collapse_before_dict_coalescing(tmp_path: Path) -> None:
    for case in ("target", "filename", "abi", "platform"):
        root = tmp_path / case
        fixture = _admission_fixture(root)
        path = root / "record/reproducibility-record.tsv"
        lines = path.read_text().splitlines()
        wheel_rows = [i for i, line in enumerate(lines[2:], 2) if "\t" in line and not line.startswith("sdist\t")]
        first, second = wheel_rows[:2]
        row = lines[second].split("\t")
        if case == "target":
            row[0] = lines[first].split("\t")[0]
        elif case == "filename":
            row[5] = lines[first].split("\t")[5]
        elif case == "abi":
            fields = row[5].split("-")
            fields[-2] = "abi3"
            row[5] = "-".join(fields)
        else:
            row[5] = row[5].rsplit("-", 1)[0] + "-win_arm64.whl"
        lines[second] = "\t".join(row)
        path.write_text("\n".join(lines) + "\n")
        result = _run_admission(root, _BYTES_STEP)
        assert result.returncode != 0, case
        assert ("duplicate record" if case in ("target", "filename") else "platform/ABI mismatch") in result.stderr


def test_release_admission_ignores_legitimate_non_distribution_artifacts(tmp_path: Path) -> None:
    # Non-distribution artifacts a full run also uploads must not be confused
    # with the distribution set: central gate diagnostic reports (names per
    # ContextualWisdomLab/.github 00c6551183cca101cfc97c43656a17cc2491c1b4, gate
    # L567/L575), the per-leg licence pair inputs, and this workflow's own
    # reproducibility digest/rebuild artifacts.
    legs = _expected_legs()
    diagnostics = [
        f"release-dependency-{kind}-report--license-evidence-{leg}" for kind in ("license", "gate") for leg in legs
    ] + [f"license-pair-{leg}" for leg in legs] + [f"repro-digest-{leg}" for leg in legs] + [
        "repro-digest-sdist"] + [f"repro-rebuild-{leg}" for leg in legs] + ["repro-rebuild-sdist"]
    assert len(diagnostics) == 24 + 12 + 13 + 13

    def listing_with(root: Path, extra: list[str]) -> list[dict]:
        listing = _admission_fixture(root)["listing"]
        template = dict(listing[0])
        return listing + [dict(template, name=name) for name in extra]

    ok = _run_admission(tmp_path / "ok", _SET_STEP, listing_with(tmp_path / "ok", diagnostics))
    assert ok.returncode == 0, ok.stderr

    # Rejections that must survive the relaxation-free check above.
    for name, extra, expected in (
        ("unknown-evidence", diagnostics + ["license-evidence-unknown-leg"], "unexpected publishable or evidence"),
        ("unknown-dist", diagnostics + ["dist-wheel-unknown-leg"], "unexpected publishable or evidence"),
        ("duplicate-leg", diagnostics + [f"dist-wheel-{legs[0]}"], "duplicate artifact name"),
        ("duplicate-evidence", diagnostics + [f"license-evidence-{legs[0]}"], "duplicate artifact name"),
    ):
        result = _run_admission(tmp_path / name, _SET_STEP, listing_with(tmp_path / name, extra))
        assert result.returncode != 0 and expected in result.stderr, (name, result.stderr)

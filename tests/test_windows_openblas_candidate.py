"""The candidate gate must reject missing and forbidden native evidence."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/verify_windows_openblas_candidate.py"
)
WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github/workflows/openblas-win-nofortran-candidate.yml"
)
SPEC = importlib.util.spec_from_file_location("candidate_gate", SCRIPT)
assert SPEC and SPEC.loader
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


class CandidateGateTest(unittest.TestCase):
    def test_symbol_renaming_linker_receives_evidence_flags(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("'-DCMAKE_LINKER_FLAGS=/MAP;/VERBOSE'", workflow)
        self.assertIn("(Get-Command lld-link.exe -ErrorAction Stop).Source", workflow)
        self.assertNotIn("CMAKE_SHARED_LINKER_FLAGS", workflow)
        self.assertNotIn("-DCMAKE_LINKER=link.exe", workflow)

    def test_download_requires_attested_source_and_signer(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        verify = workflow.split("- name: Verify downloaded evidence", 1)[1]
        verify = verify.split("- name: Retain failed-build diagnostics", 1)[0]
        self.assertIn("gh attestation verify downloaded/manifest.json", verify)
        self.assertIn("--repo $env:GITHUB_REPOSITORY", verify)
        self.assertIn("--signer-workflow", verify)
        self.assertIn("--source-digest $env:GITHUB_SHA", verify)
        self.assertIn("if ($LASTEXITCODE -ne 0)", verify)

    def test_fail_closed_native_evidence(self):
        imports = "Image has the following dependencies:\n    KERNEL32.dll\n    VCRUNTIME140.dll\n"
        link = "  Loaded C:\\MSVC\\libcmt.lib(init.obj)\n"
        mapping = "libscipy_openblas64_.dll\n  0001:0000 openblas.obj\n"
        self.assertEqual(
            GATE.inspect(imports, link, mapping)[0],
            ["KERNEL32.dll", "VCRUNTIME140.dll"],
        )
        for bad in (
            (imports + "    libgfortran.dll\n", link, mapping),
            (imports + "    mystery.dll\n", link, mapping),
            (imports, "  Loaded C:\\GCC\\libgcc.a(start.obj)\n", mapping),
            (imports, "", mapping),
            (imports, link, ""),
            (imports, link, mapping + "libquadmath.a(q.obj)\n"),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                GATE.inspect(*bad)

    def test_evidence_manifest_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name in GATE.REQUIRED_EVIDENCE:
                (directory / name).write_text(name, encoding="utf-8")
            manifest = {
                "source_commit": GATE.OPENBLAS_COMMIT,
                "files": [
                    {"name": name, "sha256": GATE.sha256(directory / name)}
                    for name in sorted(GATE.REQUIRED_EVIDENCE)
                ],
            }
            (directory / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            GATE.verify_bundle(directory)
            (directory / "imports.txt").write_text("altered", encoding="utf-8")
            with self.assertRaises(ValueError):
                GATE.verify_bundle(directory)
            (directory / "imports.txt").write_text("imports.txt", encoding="utf-8")
            (directory / "unlisted.txt").write_text("extra", encoding="utf-8")
            with self.assertRaises(ValueError):
                GATE.verify_bundle(directory)


if __name__ == "__main__":
    unittest.main()

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HELPER_PATH = Path(__file__).with_name("apply_patch.py")
MODULE_SPEC = importlib.util.spec_from_file_location("graphify_apply_patch", HELPER_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
APPLY_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(APPLY_MODULE)


class ApplyPatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.helper_directory = Path(self.temporary_directory.name) / "graphify"
        self.virtual_environment = self.helper_directory / ".venv"
        self.site_packages = self.virtual_environment / "lib" / "python" / "site-packages"
        self.engine_path = self.site_packages / "graphify" / "extractors" / "engine.py"
        self.engine_path.parent.mkdir(parents=True)
        self.engine_path.write_text("old\n", encoding="utf-8")
        self.patch_path = self.helper_directory / "engine.patch"
        self.patch_path.parent.mkdir(parents=True, exist_ok=True)
        self.patch_path.write_text(
            "--- engine.py\n+++ engine.py\n@@ -1 +1 @@\n-old\n+new\n",
            encoding="utf-8",
        )
        self.manifest_path = self.helper_directory / "manifest.json"
        self.write_manifest()
        self.module_file = APPLY_MODULE.__file__
        APPLY_MODULE.__file__ = str(self.helper_directory / "apply_patch.py")
        self.source_patch = mock.patch.object(
            APPLY_MODULE, "EXPECTED_SOURCE_SHA256", hashlib.sha256(b"old\n").hexdigest()
        )
        self.prefix_patch = mock.patch.object(sys, "prefix", str(self.virtual_environment))
        self.site_patch = mock.patch.object(
            APPLY_MODULE.sysconfig, "get_path", return_value=str(self.site_packages)
        )
        self.prefix_patch.start()
        self.site_patch.start()
        self.source_patch.start()

    def tearDown(self):
        self.site_patch.stop()
        self.source_patch.stop()
        self.prefix_patch.stop()
        APPLY_MODULE.__file__ = self.module_file
        self.temporary_directory.cleanup()

    def write_manifest(self, **updates):
        manifest_value = {
            "source_sha256": hashlib.sha256(b"old\n").hexdigest(),
            "patch_sha256": hashlib.sha256(self.patch_path.read_bytes()).hexdigest(),
            "final_sha256": hashlib.sha256(b"new\n").hexdigest(),
            "tool_version": "graphifyy-test",
            "wheel_sha256": "0" * 64,
        }
        manifest_value.update(updates)
        self.manifest_path.write_text(json.dumps(manifest_value), encoding="utf-8")

    def test_valid_apply_and_idempotence(self):
        self.assertEqual(APPLY_MODULE.main(), 0)
        self.assertEqual(self.engine_path.read_text(encoding="utf-8"), "new\n")
        self.assertEqual(APPLY_MODULE.main(), 0)

    def test_wrong_prefix_rejected(self):
        with mock.patch.object(sys, "prefix", "/tmp/global-python"), self.assertRaises(SystemExit):
            APPLY_MODULE.main()

    def test_site_packages_outside_venv_rejected(self):
        with mock.patch.object(
            APPLY_MODULE.sysconfig, "get_path", return_value="/tmp/site-packages"
        ), self.assertRaises(SystemExit):
            APPLY_MODULE.main()

    def test_symlinked_venv_rejected(self):
        symlink_helper = Path(self.temporary_directory.name) / "symlink-helper"
        symlink_helper.mkdir()
        real_environment = Path(self.temporary_directory.name) / "real-venv"
        real_environment.mkdir()
        (symlink_helper / ".venv").symlink_to(real_environment)
        with mock.patch.object(APPLY_MODULE, "__file__", str(symlink_helper / "apply_patch.py")):
            with self.assertRaises(SystemExit):
                APPLY_MODULE.main()

    def test_engine_symlink_outside_rejected(self):
        self.engine_path.unlink()
        outside_path = Path(self.temporary_directory.name) / "outside.py"
        outside_path.write_text("old\n", encoding="utf-8")
        self.engine_path.symlink_to(outside_path)
        with self.assertRaises(SystemExit):
            APPLY_MODULE.main()

    def test_patch_tamper_rejected_before_engine_change(self):
        original_bytes = self.engine_path.read_bytes()
        self.patch_path.write_text(self.patch_path.read_text() + "tamper", encoding="utf-8")
        with self.assertRaises(SystemExit):
            APPLY_MODULE.main()
        self.assertEqual(self.engine_path.read_bytes(), original_bytes)

    def test_unknown_source_rejected_before_engine_change(self):
        original_bytes = self.engine_path.read_bytes()
        self.write_manifest(source_sha256="1" * 64)
        with self.assertRaises(SystemExit):
            APPLY_MODULE.main()
        self.assertEqual(self.engine_path.read_bytes(), original_bytes)

    def test_final_digest_mismatch_leaves_original(self):
        original_bytes = self.engine_path.read_bytes()
        self.write_manifest(final_sha256="2" * 64)
        with self.assertRaises(SystemExit):
            APPLY_MODULE.main()
        self.assertEqual(self.engine_path.read_bytes(), original_bytes)

    def test_failed_patch_leaves_original(self):
        original_bytes = self.engine_path.read_bytes()
        self.patch_path.write_text("not a patch\n", encoding="utf-8")
        self.write_manifest(patch_sha256=hashlib.sha256(self.patch_path.read_bytes()).hexdigest())
        with self.assertRaises(subprocess.CalledProcessError):
            APPLY_MODULE.main()
        self.assertEqual(self.engine_path.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()

"""The candidate gate must reject missing and forbidden native evidence."""

import importlib.util
from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/verify_windows_openblas_candidate.py"
)
SPEC = importlib.util.spec_from_file_location("candidate_gate", SCRIPT)
assert SPEC and SPEC.loader
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


class CandidateGateTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

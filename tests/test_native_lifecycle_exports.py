"""Source-composition RED; native execution is verified separately."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class NativeComposition(unittest.TestCase):
    def test_recovered_native_is_enrolled(self):
        self.assertTrue((ROOT / 'crates/mlsirm-core/src/mask_grm.rs').is_file())
        lib = (ROOT / 'crates/mlsirm-core/src/lib.rs').read_text()
        self.assertIn('pub mod mask_grm;', lib)
    def test_real_exports_are_enrolled(self):
        lib = (ROOT / 'crates/fast-mlsirm-py/src/lib.rs').read_text()
        self.assertTrue('wrap_pyfunction!(sequential_dif_round_gpu' in lib)
        self.assertIn('wrap_pyfunction!(sequential_dif_capabilities', lib)

if __name__ == '__main__':
    unittest.main()

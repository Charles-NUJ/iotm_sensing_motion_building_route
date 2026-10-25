import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_missions import generate

class MissionTest(unittest.TestCase):
    def test_formation_fixture(self):
        generated = np.fromstring(generate("formation", 24, 4501).decode("ascii"), sep=" ")
        original = np.fromstring((ROOT / "inputs/formation_n24_s4501.dat").read_text(encoding="ascii"), sep=" ")
        self.assertEqual(generated.shape, original.shape)
        np.testing.assert_allclose(generated, original, rtol=0, atol=1e-10)

if __name__ == "__main__":
    unittest.main()

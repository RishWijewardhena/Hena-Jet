import unittest
from pathlib import Path

import numpy as np

from scripts.mano_smoke_test import generate_neutral_hand


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "Mano" / "mano_v1_2" / "models"


class ManoSmokeTest(unittest.TestCase):
    def test_generates_finite_left_and_right_hand_meshes(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                vertices, faces = generate_neutral_hand(MODEL_DIR, side)

                self.assertEqual(vertices.shape, (778, 3))
                self.assertEqual(faces.ndim, 2)
                self.assertEqual(faces.shape[1], 3)
                self.assertTrue(np.isfinite(vertices).all())
                self.assertGreater(faces.shape[0], 1_000)


if __name__ == "__main__":
    unittest.main()

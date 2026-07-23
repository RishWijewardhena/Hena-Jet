import math
import unittest
from pathlib import Path

import numpy as np

from scripts.capture_zed_multilevel import (
    lift_pose_metrics,
    normalize_height_offsets_m,
    pass_directory,
    transition_is_ready,
)


class CaptureZedMultilevelTests(unittest.TestCase):
    def test_accepts_three_strictly_increasing_absolute_height_offsets(self):
        self.assertEqual(
            normalize_height_offsets_m([0.0, 0.02, 0.04]),
            (0.0, 0.02, 0.04),
        )

    def test_rejects_invalid_height_sequences(self):
        invalid_sequences = (
            [0.02, 0.04],
            [0.0],
            [0.0, 0.02, 0.02],
            [0.0, -0.02],
            [0.0, math.nan],
        )
        for offsets in invalid_sequences:
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                normalize_height_offsets_m(offsets)

    def test_builds_stable_pass_directory_names_in_millimetres(self):
        root = Path("captures/hand")
        self.assertEqual(
            pass_directory(root, 0, 0.0),
            root / "pass_00_height_000mm",
        )
        self.assertEqual(
            pass_directory(root, 2, 0.04),
            root / "pass_02_height_040mm",
        )

    def test_transition_requires_both_minimum_wait_and_operator_confirmation(self):
        self.assertFalse(transition_is_ready(30.0, 30.0, False))
        self.assertFalse(transition_is_ready(29.999, 30.0, True))
        self.assertTrue(transition_is_ready(30.0, 30.0, True))

    def test_accepts_exact_vertical_lift_with_unchanged_orientation(self):
        before = np.eye(4)
        after = np.eye(4)
        after[:3, 3] = [0.0, -0.02, 0.0]

        metrics = lift_pose_metrics(
            before,
            after,
            expected_height_delta_m=0.02,
            object_up=np.array([0.0, -1.0, 0.0]),
            translation_tolerance_m=0.002,
            rotation_tolerance_deg=1.0,
        )

        self.assertTrue(metrics["accepted"])
        self.assertAlmostEqual(metrics["vertical_translation_m"], 0.02)
        self.assertAlmostEqual(metrics["vertical_error_m"], 0.0)
        self.assertAlmostEqual(metrics["lateral_error_m"], 0.0)
        self.assertAlmostEqual(metrics["rotation_error_deg"], 0.0)

    def test_rejects_excess_vertical_lateral_or_rotation_error(self):
        before = np.eye(4)
        cases = []

        vertical = np.eye(4)
        vertical[:3, 3] = [0.0, -0.017, 0.0]
        cases.append(vertical)

        lateral = np.eye(4)
        lateral[:3, 3] = [0.003, -0.02, 0.0]
        cases.append(lateral)

        rotated = np.eye(4)
        angle = math.radians(1.1)
        rotated[:3, :3] = [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
        rotated[:3, 3] = [0.0, -0.02, 0.0]
        cases.append(rotated)

        for candidate in cases:
            with self.subTest(candidate=candidate):
                metrics = lift_pose_metrics(
                    before,
                    candidate,
                    expected_height_delta_m=0.02,
                    object_up=np.array([0.0, -1.0, 0.0]),
                    translation_tolerance_m=0.002,
                    rotation_tolerance_deg=1.0,
                )
                self.assertFalse(metrics["accepted"])


if __name__ == "__main__":
    unittest.main()

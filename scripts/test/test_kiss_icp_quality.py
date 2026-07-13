import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from kiss_icp_quality import (  # noqa: E402
    GlobalVoxelAccumulator,
    deterministic_voxel_sample,
    evaluate_frame_quality,
    filter_organized_cloud,
    pose_step,
)


def packed_bgra(red: int, green: int, blue: int, alpha: int = 255) -> np.float32:
    value = np.array(
        [blue | (green << 8) | (red << 16) | (alpha << 24)],
        dtype=np.uint32,
    )
    return value.view(np.float32)[0]


class OrganizedCloudFilterTests(unittest.TestCase):
    def test_rejects_invalid_range_confidence_and_depth_edges(self) -> None:
        cloud = np.zeros((7, 7, 4), dtype=np.float32)
        yy, xx = np.indices((7, 7))
        cloud[..., 0] = 0.15
        cloud[..., 1] = (xx - 3) * 0.001
        cloud[..., 2] = (yy - 3) * 0.001
        cloud[..., 3] = packed_bgra(10, 20, 30)
        confidence = np.zeros((7, 7), dtype=np.float32)

        cloud[3, 3, 0] = np.nan
        cloud[1, 1, 0] = 0.10
        confidence[1, 5] = 61
        cloud[4:, :, 0] = 0.17
        cloud[5, 5, 0] = 0.23

        result = filter_organized_cloud(
            cloud,
            confidence,
            min_depth_m=0.11,
            max_depth_m=0.22,
            coordinate_system="RIGHT_HANDED_Z_UP_X_FWD",
            confidence_threshold=60,
            edge_threshold_m=0.008,
            erode_invalid_boundary=True,
        )

        self.assertFalse(result.mask[3, 3])
        self.assertFalse(result.mask[1, 1])
        self.assertFalse(result.mask[5, 5])
        self.assertFalse(result.mask[1, 5])
        self.assertFalse(result.mask[3, 0])
        self.assertFalse(result.mask[4, 0])
        self.assertGreater(result.points.shape[0], 0)
        np.testing.assert_array_equal(result.colors[0], [10, 20, 30])
        self.assertGreater(result.metrics["base_valid_coverage"], result.metrics["filtered_valid_coverage"])

    def test_rejects_zero_inf_and_nan_points(self) -> None:
        cloud = np.zeros((5, 5, 4), dtype=np.float32)
        cloud[..., 0] = 0.15
        cloud[..., 3] = packed_bgra(1, 2, 3)
        confidence = np.zeros((5, 5), dtype=np.float32)
        cloud[0, 0, :3] = 0
        cloud[0, 1, 1] = np.inf
        cloud[0, 2, 2] = np.nan

        result = filter_organized_cloud(
            cloud,
            confidence,
            min_depth_m=0.11,
            max_depth_m=0.22,
            coordinate_system="RIGHT_HANDED_Z_UP_X_FWD",
            confidence_threshold=60,
            edge_threshold_m=0.008,
            erode_invalid_boundary=False,
        )

        self.assertEqual(result.points.shape[0], 22)
        self.assertTrue(np.isfinite(result.points).all())


class SamplingAndQualityTests(unittest.TestCase):
    def test_voxel_sampling_is_deterministic_and_bounded(self) -> None:
        points = np.array(
            [[0.001 * i, 0.002 * (i % 7), 0.15] for i in range(100)],
            dtype=np.float32,
        )
        first = deterministic_voxel_sample(points, voxel_m=0.003, max_points=17)
        second = deterministic_voxel_sample(points, voxel_m=0.003, max_points=17)

        self.assertLessEqual(first.shape[0], 17)
        np.testing.assert_array_equal(first, second)

    def test_frame_quality_rejects_sparse_low_coverage_and_planar_clouds(self) -> None:
        sparse = np.zeros((20, 3), dtype=np.float32)
        self.assertEqual(
            evaluate_frame_quality(sparse, 0.5, min_points=100).reason,
            "too_few_points",
        )

        points = np.random.default_rng(7).normal(size=(500, 3)).astype(np.float32) * 0.02
        self.assertEqual(
            evaluate_frame_quality(points, 0.01, min_points=100).reason,
            "low_valid_coverage",
        )

        planar = points.copy()
        planar[:, 2] = 0.15
        self.assertEqual(
            evaluate_frame_quality(planar, 0.5, min_points=100).reason,
            "degenerate_geometry",
        )

        valid = points + np.array([0.0, 0.0, 0.15], dtype=np.float32)
        self.assertTrue(evaluate_frame_quality(valid, 0.5, min_points=100).accepted)


class PoseAndFusionTests(unittest.TestCase):
    def test_pose_step_reports_translation_and_rotation(self) -> None:
        current = np.eye(4)
        angle = np.deg2rad(6.0)
        current[:3, :3] = [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
        current[0, 3] = 0.03

        translation_m, rotation_deg = pose_step(np.eye(4), current)

        self.assertAlmostEqual(translation_m, 0.03)
        self.assertAlmostEqual(rotation_deg, 6.0)

    def test_global_voxel_accumulator_averages_and_counts_frames(self) -> None:
        accumulator = GlobalVoxelAccumulator(voxel_m=0.01)
        first_points = np.array([[0.001, 0.0, 0.15], [0.002, 0.0, 0.15]])
        first_colors = np.array([[100, 0, 0], [200, 0, 0]], dtype=np.uint8)
        second_points = np.array([[0.003, 0.0, 0.15], [0.02, 0.0, 0.15]])
        second_colors = np.array([[0, 100, 0], [0, 0, 255]], dtype=np.uint8)

        accumulator.update(first_points, first_colors)
        accumulator.update(second_points, second_colors)

        points, colors, counts = accumulator.to_arrays(min_observations=2)
        self.assertEqual(points.shape, (1, 3))
        self.assertEqual(counts.tolist(), [2])
        np.testing.assert_allclose(points[0], [0.00225, 0.0, 0.15])
        np.testing.assert_array_equal(colors[0], [75, 50, 0])


if __name__ == "__main__":
    unittest.main()

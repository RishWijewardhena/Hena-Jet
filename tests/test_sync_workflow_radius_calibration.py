from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SYNC_WORKFLOW_DIR = (
    Path(__file__).resolve().parent.parent / "scripts" / "sync_workflow"
)
sys.path.insert(0, str(SYNC_WORKFLOW_DIR))

from radius_calibration import (  # noqa: E402
    build_profile_marker_map,
    convert_world_to_color_to_world_to_depth,
    fit_orbit_circle,
    has_sufficient_angular_coverage,
    marker_normal,
)


class ProfileMarkerMapTests(unittest.TestCase):
    def test_uses_requested_profile_carrier_and_axial_geometry(self):
        marker_map = build_profile_marker_map()

        self.assertEqual(marker_map["dictionary"], "DICT_5X5_100")
        self.assertEqual(marker_map["marker_size_m"], 0.018)
        self.assertEqual(marker_map["profile_cross_section_m"], [0.040, 0.020])
        self.assertEqual(marker_map["carrier_offset_m"], 0.001)
        self.assertEqual(
            [marker["center_m"][2] for marker in marker_map["markers"]],
            [-0.045, -0.015, 0.015, 0.045],
        )

        centers = {
            marker["face"]: np.asarray(marker["center_m"])
            for marker in marker_map["markers"]
        }
        np.testing.assert_allclose(centers["front"], [0.0, 0.011, -0.045])
        np.testing.assert_allclose(centers["right"], [0.021, 0.0, -0.015])
        np.testing.assert_allclose(centers["back"], [0.0, -0.011, 0.015])
        np.testing.assert_allclose(centers["left"], [-0.021, 0.0, 0.045])

    def test_marker_corners_are_18_mm_and_face_outward(self):
        marker_map = build_profile_marker_map()
        expected_normals = {
            "front": [0.0, 1.0, 0.0],
            "right": [1.0, 0.0, 0.0],
            "back": [0.0, -1.0, 0.0],
            "left": [-1.0, 0.0, 0.0],
        }

        for marker in marker_map["markers"]:
            corners = np.asarray(marker["corners_m"])
            side_lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
            np.testing.assert_allclose(side_lengths, 0.018, atol=1e-12)
            np.testing.assert_allclose(
                marker_normal(corners), expected_normals[marker["face"]], atol=1e-12
            )


class OrbitFitTests(unittest.TestCase):
    def test_recovers_a_tilted_three_dimensional_circle(self):
        radius = 0.1175
        center = np.array([0.035, -0.020, 0.080])
        axis = np.array([0.2, -0.3, 0.9327379053])
        axis /= np.linalg.norm(axis)
        basis_u = np.cross(axis, [0.0, 0.0, 1.0])
        basis_u /= np.linalg.norm(basis_u)
        basis_v = np.cross(axis, basis_u)
        angles = np.deg2rad(np.arange(0.0, 360.0, 30.0))
        points = np.array(
            [center + radius * (np.cos(a) * basis_u + np.sin(a) * basis_v) for a in angles]
        )

        result = fit_orbit_circle(points)

        self.assertAlmostEqual(result["radius_m"], radius, places=9)
        np.testing.assert_allclose(result["center_m"], center, atol=1e-9)
        self.assertAlmostEqual(result["rmse_m"], 0.0, places=9)

    def test_robust_refit_removes_a_large_trajectory_outlier(self):
        angles = np.deg2rad(np.arange(0.0, 360.0, 30.0))
        points = np.column_stack(
            (0.1175 * np.cos(angles), 0.1175 * np.sin(angles), np.zeros_like(angles))
        )
        points[4] += [0.030, -0.020, 0.015]

        result = fit_orbit_circle(points, mad_threshold=3.5)

        self.assertFalse(result["inlier_mask"][4])
        self.assertAlmostEqual(result["radius_m"], 0.1175, places=6)

    def test_checks_full_orbit_coverage_using_the_largest_circular_gap(self):
        self.assertTrue(
            has_sufficient_angular_coverage(
                [0, 45, 90, 135, 180, -135, -90, -45],
                min_unique_angles=6,
                max_gap_deg=90.0,
            )
        )
        self.assertFalse(
            has_sufficient_angular_coverage(
                [0, 15, 30, 45, 60, 75],
                min_unique_angles=6,
                max_gap_deg=90.0,
            )
        )


class CameraExtrinsicTests(unittest.TestCase):
    def test_converts_world_to_color_pose_to_world_to_depth_pose(self):
        world_to_color = np.eye(4)
        world_to_color[:3, 3] = [0.1, -0.2, 0.3]
        depth_to_color = np.eye(4)
        depth_to_color[:3, 3] = [0.025, 0.0, 0.0]

        world_to_depth = convert_world_to_color_to_world_to_depth(
            world_to_color, depth_to_color
        )

        np.testing.assert_allclose(world_to_depth[:3, 3], [0.075, -0.2, 0.3])
        color_center_world = np.linalg.inv(world_to_color)[:3, 3]
        depth_center_world = np.linalg.inv(world_to_depth)[:3, 3]
        np.testing.assert_allclose(depth_center_world - color_center_world, [0.025, 0, 0])


if __name__ == "__main__":
    unittest.main()

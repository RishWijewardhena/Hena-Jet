from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SYNC_WORKFLOW_DIR = (
    Path(__file__).resolve().parent.parent / "scripts" / "sync_workflow"
)
sys.path.insert(0, str(SYNC_WORKFLOW_DIR))

from calculating_radius.orbit_pose_map import (  # noqa: E402
    aggregate_world_to_camera_poses,
    build_orbit_pose_map,
    camera_poses_in_reference,
)


def camera_to_world(center, angle_deg=0.0):
    angle = np.deg2rad(angle_deg)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = center
    return transform


class PoseAggregationTests(unittest.TestCase):
    def test_robust_pose_aggregation_rejects_one_large_outlier(self):
        good = [
            np.linalg.inv(
                camera_to_world(
                    [0.100 + offset, -0.020, 0.030],
                    angle_deg=20.0 + rotation_offset,
                )
            )
            for offset, rotation_offset in (
                (-0.0002, -0.08),
                (-0.0001, -0.03),
                (0.0, 0.0),
                (0.0001, 0.03),
                (0.0002, 0.08),
            )
        ]
        outlier = np.linalg.inv(camera_to_world([0.145, 0.020, -0.010], 35.0))

        result = aggregate_world_to_camera_poses(good + [outlier], min_inliers=3)

        self.assertEqual(result["inlier_mask"], [True] * 5 + [False])
        recovered = np.asarray(result["world_to_camera"])
        np.testing.assert_allclose(
            np.linalg.inv(recovered)[:3, 3], [0.100, -0.020, 0.030], atol=1e-6
        )
        self.assertLess(result["rotation_spread_deg"], 0.1)


class OrbitPoseMapTests(unittest.TestCase):
    def test_relative_pose_maps_each_camera_directly_into_the_reference_camera(self):
        world_to_depth_0 = np.linalg.inv(camera_to_world([0.0, -0.1, 0.0], 0.0))
        world_to_depth_90 = np.linalg.inv(camera_to_world([0.1, 0.0, 0.0], 90.0))
        angle_records = [
            {"angle_deg": 0.0, "pose_valid": True, "world_to_depth": world_to_depth_0.tolist()},
            {"angle_deg": 90.0, "pose_valid": True, "world_to_depth": world_to_depth_90.tolist()},
        ]

        pose_map = build_orbit_pose_map(
            angle_records,
            reference_angle_deg=0.0,
            x_position_mm=150.0,
            marker_map_path="profile_marker_map.json",
        )
        poses = camera_poses_in_reference(pose_map, [0.0, 90.0])

        np.testing.assert_allclose(poses[0], np.eye(4), atol=1e-12)
        expected = world_to_depth_0 @ np.linalg.inv(world_to_depth_90)
        np.testing.assert_allclose(poses[1], expected, atol=1e-12)
        point_in_world = np.array([0.02, 0.03, 0.04, 1.0])
        point_in_camera_90 = world_to_depth_90 @ point_in_world
        np.testing.assert_allclose(
            poses[1] @ point_in_camera_90,
            world_to_depth_0 @ point_in_world,
            atol=1e-12,
        )

    def test_pose_map_rejects_an_angle_without_a_measured_pose(self):
        pose = np.eye(4).tolist()
        pose_map = build_orbit_pose_map(
            [{"angle_deg": 0.0, "pose_valid": True, "world_to_depth": pose}],
            reference_angle_deg=0.0,
            x_position_mm=150.0,
            marker_map_path="profile_marker_map.json",
        )

        with self.assertRaisesRegex(ValueError, "no measured pose"):
            camera_poses_in_reference(pose_map, [0.0, 10.0])


if __name__ == "__main__":
    unittest.main()

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuse_orbit_posegraph import (
    icp_result_is_acceptable,
    relative_camera_transform,
    statistical_outlier_filter,
    vslam_orbit_metrics,
    transform_delta,
)
from fuse_tsdf_scan import (
    camera_to_world_matrix,
    capture_pose,
    depth_confidence_mask,
    object_crop_mask,
)


class OrbitPoseGraphTests(unittest.TestCase):
    def test_statistical_outlier_filter_removes_isolated_point(self):
        grid = np.array(
            [[x * 0.001, y * 0.001, 0.0] for x in range(5) for y in range(5)],
            dtype=np.float64,
        )
        points = np.vstack([grid, np.array([[1.0, 1.0, 1.0]])])
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)

        filtered, removed = statistical_outlier_filter(
            cloud,
            neighbors=8,
            std_ratio=1.0,
        )

        filtered_points = np.asarray(filtered.points)
        self.assertGreaterEqual(removed, 1)
        self.assertGreaterEqual(len(filtered_points), 20)
        self.assertFalse(np.any(np.all(np.isclose(filtered_points, [1.0, 1.0, 1.0]), axis=1)))

    def test_camera_mount_yaw_changes_orientation_without_moving_optical_center(self):
        base = camera_to_world_matrix(35.0, 0.192, 0.05)
        yawed = camera_to_world_matrix(
            35.0,
            0.192,
            0.05,
            camera_yaw_deg=-6.0,
        )

        np.testing.assert_allclose(yawed[:3, 3], base[:3, 3], atol=1e-12)
        self.assertFalse(np.allclose(yawed[:3, :3], base[:3, :3]))
        np.testing.assert_allclose(
            yawed[:3, :3].T @ yawed[:3, :3],
            np.eye(3),
            atol=1e-12,
        )

    def test_relative_camera_transform_maps_source_camera_origin_to_target_camera(self):
        source_pose = camera_to_world_matrix(0.0, 0.192, 0.05)
        target_pose = camera_to_world_matrix(5.0, 0.192, 0.05)

        source_to_target = relative_camera_transform(source_pose, target_pose)
        source_origin_in_target = source_to_target @ np.array([0.0, 0.0, 0.0, 1.0])
        expected = np.linalg.inv(target_pose) @ source_pose[:, 3]

        np.testing.assert_allclose(source_origin_in_target, expected, atol=1e-12)
        np.testing.assert_allclose(
            target_pose @ source_to_target,
            source_pose,
            atol=1e-12,
        )

    def test_transform_delta_reports_candidate_correction_from_prior(self):
        prior = np.eye(4)
        candidate = np.eye(4)
        angle = math.radians(3.0)
        candidate[:3, :3] = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        candidate[:3, 3] = [0.004, 0.003, 0.0]

        translation_m, rotation_deg = transform_delta(prior, candidate)

        self.assertTrue(math.isclose(translation_m, 0.005, abs_tol=1e-12))
        self.assertTrue(math.isclose(rotation_deg, 3.0, abs_tol=1e-9))

    def test_icp_acceptance_rejects_large_departure_from_orbit_prior(self):
        prior = np.eye(4)
        candidate = np.eye(4)
        candidate[0, 3] = 0.03

        accepted, reason, correction_m, _ = icp_result_is_acceptable(
            fitness=0.8,
            rmse_m=0.003,
            prior=prior,
            candidate=candidate,
            min_fitness=0.3,
            max_rmse_m=0.01,
            max_correction_m=0.01,
            max_correction_deg=5.0,
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "translation correction exceeds prior guard")
        self.assertTrue(math.isclose(correction_m, 0.03))

    def test_icp_acceptance_accepts_good_small_refinement(self):
        prior = np.eye(4)
        candidate = np.eye(4)
        candidate[1, 3] = 0.002

        accepted, reason, _, _ = icp_result_is_acceptable(
            fitness=0.75,
            rmse_m=0.002,
            prior=prior,
            candidate=candidate,
            min_fitness=0.3,
            max_rmse_m=0.01,
            max_correction_m=0.01,
            max_correction_deg=5.0,
        )

        self.assertTrue(accepted)
        self.assertEqual(reason, "accepted")

    def test_vslam_pose_source_uses_saved_camera_to_world_matrix(self):
        from types import SimpleNamespace

        saved_pose = np.eye(4)
        saved_pose[0, 3] = 0.123
        meta = {
            "radius_m": 0.192,
            "height_m": 0.0,
            "angle_deg": 5.0,
            "camera_to_vslam_world": saved_pose.tolist(),
        }
        args = SimpleNamespace(
            pose_source="vslam",
            override_radius_m=None,
            override_height_m=None,
            invert_angles=False,
            angle_offset_deg=0.0,
            camera_yaw_deg=0.0,
            center_offset_x_m=0.0,
            center_offset_y_m=0.0,
        )

        radius, pose = capture_pose(meta, args)

        self.assertEqual(radius, 0.192)
        np.testing.assert_allclose(pose, saved_pose)

    def test_depth_confidence_uses_zero_best_and_100_worst_scale(self):
        confidence = np.array([[0, 49, 50, 51, 100]], dtype=np.uint8)
        np.testing.assert_array_equal(
            depth_confidence_mask(confidence, 50),
            [[True, True, True, False, False]],
        )

    def test_object_crop_uses_explicit_center_up_and_radial_distance(self):
        points = np.array(
            [
                [[0.0, 0.0, 0.0], [0.04, 0.0, 0.0]],
                [[0.0, -0.03, 0.0], [0.07, 0.0, 0.0]],
            ]
        )
        mask = object_crop_mask(
            points,
            center=np.zeros(3),
            up=np.array([0.0, -1.0, 0.0]),
            min_height_m=-0.01,
            max_height_m=0.04,
            max_radius_m=0.06,
        )
        np.testing.assert_array_equal(mask, [[True, True], [True, False]])

    def test_vslam_orbit_metrics_pass_for_exact_circle_and_closed_trajectory(self):
        radius = 0.192
        center = np.array([0.0, 0.0, radius])
        poses = []
        for angle_deg in range(5, 361, 5):
            angle = math.radians(angle_deg)
            pose = np.eye(4)
            pose[:3, 3] = center + [radius * math.sin(angle), 0.0, -radius * math.cos(angle)]
            poses.append(pose)
        metrics = vslam_orbit_metrics(
            np.stack(poses),
            center=center,
            up=np.array([0.0, -1.0, 0.0]),
            initial_pose=np.eye(4),
        )
        self.assertLess(metrics["radius_rmse_m"], 1e-12)
        self.assertLess(metrics["closure_translation_m"], 1e-12)
        self.assertLess(metrics["closure_rotation_deg"], 1e-12)


if __name__ == "__main__":
    unittest.main()

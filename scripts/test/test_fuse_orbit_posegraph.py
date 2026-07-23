import math
import sys
from tempfile import TemporaryDirectory
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuse_orbit_posegraph import (
    build_vslam_feasibility_report,
    discover_capture_paths,
    icp_result_is_acceptable,
    registration_pair_plan,
    relative_camera_transform,
    statistical_outlier_filter,
    vslam_orbit_metrics,
    transform_delta,
)
from fuse_tsdf_scan import (
    camera_to_world_matrix,
    capture_pose,
    depth_confidence_mask,
    discover_capture_paths as discover_tsdf_capture_paths,
    object_crop_mask,
)


class OrbitPoseGraphTests(unittest.TestCase):
    def test_discovers_and_orders_multilevel_captures_by_pass_and_capture_index(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for pass_index, height_mm in ((1, 20), (0, 0)):
                pass_dir = root / f"pass_{pass_index:02d}_height_{height_mm:03d}mm"
                pass_dir.mkdir()
                for capture_index, angle in ((1, 10), (0, 5)):
                    path = pass_dir / f"angle_{angle:04d}p00.json"
                    path.write_text(
                        '{'
                        f'"multilevel_pass_index": {pass_index}, '
                        f'"multilevel_capture_index": {pass_index * 72 + capture_index}, '
                        f'"timestamp_ns": {1000 + capture_index}'
                        '}',
                        encoding="utf-8",
                    )
                    paths.append(path)

            discovered = discover_capture_paths(root)
            tsdf_discovered = discover_tsdf_capture_paths(root)

            self.assertEqual(
                [(path.parent.name, path.name) for path in discovered],
                [
                    ("pass_00_height_000mm", "angle_0005p00.json"),
                    ("pass_00_height_000mm", "angle_0010p00.json"),
                    ("pass_01_height_020mm", "angle_0005p00.json"),
                    ("pass_01_height_020mm", "angle_0010p00.json"),
                ],
            )
            self.assertEqual(tsdf_discovered, discovered)

    def test_multilevel_feasibility_evaluates_each_pass_closure_separately(self):
        radius = 0.192
        center = np.array([0.0, 0.0, radius])
        up = np.array([0.0, -1.0, 0.0])
        frames = []
        pass_records = []

        for pass_index, height_m in enumerate((0.0, 0.02)):
            start_pose = np.eye(4)
            start_pose[:3, 3] = center + [0.0, -height_m, -radius]
            final_pose = start_pose.copy()
            pass_records.append(
                {
                    "pass_index": pass_index,
                    "capture_count": 72,
                    "start_camera_to_vslam_world": start_pose.tolist(),
                    "final_capture_camera_to_vslam_world": final_pose.tolist(),
                }
            )
            for angle_deg in range(5, 361, 5):
                angle = math.radians(angle_deg)
                pose = np.eye(4)
                pose[:3, 3] = center + [
                    radius * math.sin(angle),
                    -height_m,
                    -radius * math.cos(angle),
                ]
                frames.append(
                    SimpleNamespace(
                        pass_index=pass_index,
                        height_offset_m=height_m,
                        prior_pose=pose,
                        radius_m=radius,
                        meta={
                            "tracking_state": "OK",
                            "odometry_status": "OK",
                        },
                    )
                )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scan_session.json").write_text(
                __import__("json").dumps(
                    {
                        "height_offsets_m": [0.0, 0.02],
                        "captures_per_pass": 72,
                        "passes": pass_records,
                    }
                ),
                encoding="utf-8",
            )
            (root / "vslam_trajectory.jsonl").write_text(
                '{"tracking_state":"OK","odometry_status":"OK",'
                '"timestamp_ns":1,"camera_to_vslam_world":'
                + __import__("json").dumps(np.eye(4).tolist())
                + '}\n',
                encoding="utf-8",
            )
            args = SimpleNamespace(
                capture_dir=root,
                object_center_m=center,
                object_up=up,
            )

            report = build_vslam_feasibility_report(frames, args)

        self.assertEqual(report["required_capture_count"], 144)
        self.assertEqual(len(report["passes"]), 2)
        self.assertTrue(all(item["passed"] for item in report["passes"]))
        self.assertTrue(report["passed"])

    def test_registration_plan_keeps_passes_separate_and_pairs_nearest_views(self):
        def frame(pass_index, xyz):
            pose = np.eye(4)
            pose[:3, 3] = xyz
            return SimpleNamespace(pass_index=pass_index, prior_pose=pose)

        frames = [
            frame(0, [1.0, 0.0, 0.0]),
            frame(0, [0.0, 0.0, 1.0]),
            frame(0, [-1.0, 0.0, 0.0]),
            frame(1, [0.0, -0.02, 1.0]),
            frame(1, [-1.0, -0.02, 0.0]),
            frame(1, [1.0, -0.02, 0.0]),
        ]

        plan = registration_pair_plan(
            frames,
            center=np.zeros(3),
            up=np.array([0.0, -1.0, 0.0]),
            include_loop_closure=True,
        )

        self.assertEqual(
            [pair for pair in plan if pair[0] == "sequential"],
            [
                ("sequential", 0, 1),
                ("sequential", 1, 2),
                ("sequential", 3, 4),
                ("sequential", 4, 5),
            ],
        )
        self.assertEqual(
            [pair for pair in plan if pair[0] == "loop_closure"],
            [("loop_closure", 0, 2), ("loop_closure", 3, 5)],
        )
        self.assertEqual(
            [pair for pair in plan if pair[0] == "cross_height"],
            [
                ("cross_height", 1, 3),
                ("cross_height", 2, 4),
                ("cross_height", 0, 5),
            ],
        )

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

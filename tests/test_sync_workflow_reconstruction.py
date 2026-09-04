from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d


SYNC_WORKFLOW_DIR = (
    Path(__file__).resolve().parent.parent / "scripts" / "sync_workflow"
)
sys.path.insert(0, str(SYNC_WORKFLOW_DIR))

import reconstruct_pipeline


class ReconstructionCliTests(unittest.TestCase):
    def test_motor_registration_is_the_safe_default(self):
        args = reconstruct_pipeline.parse_args(["--input-dir", "scan"])

        self.assertEqual(args.registration_mode, "motor")

    def test_uses_the_radius_recorded_during_capture(self):
        args = reconstruct_pipeline.parse_args(["--input-dir", "scan"])

        radius = reconstruct_pipeline.resolve_orbit_radius(
            args.orbit_radius_m,
            {"orbit_radius_m": 0.1175},
        )

        self.assertEqual(radius, 0.1175)

    def test_accepts_measured_aruco_pose_mode_and_pose_map(self):
        args = reconstruct_pipeline.parse_args(
            [
                "--input-dir", "scan",
                "--registration-mode", "aruco",
                "--pose-map", "calibration/orbit_pose_map.json",
            ]
        )

        self.assertEqual(args.registration_mode, "aruco")
        self.assertEqual(args.pose_map, Path("calibration/orbit_pose_map.json"))

    def test_rejects_pose_map_for_a_different_pointcloud_camera_frame(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            (input_dir / "intrinsics.json").write_text(
                json.dumps({"coordinate_frame": "color"})
            )

            with self.assertRaisesRegex(ValueError, "coordinate-frame mismatch"):
                reconstruct_pipeline.validate_pose_map_coordinate_frame(
                    input_dir, {"pointcloud_coordinate_frame": "depth"}
                )

    def test_registration_crop_defaults_to_10cm_without_shrinking_final_crop(self):
        args = reconstruct_pipeline.parse_args(["--input-dir", "scan"])

        final_crop_m, registration_crop_m = reconstruct_pipeline.resolve_crop_radii(
            args.crop_radius_m,
            args.registration_crop_radius_m,
            {"crop_radius_m": 0.15},
        )

        self.assertEqual(final_crop_m, 0.15)
        self.assertEqual(registration_crop_m, 0.10)

    def test_registration_crop_respects_a_smaller_final_crop_and_explicit_override(self):
        self.assertEqual(
            reconstruct_pipeline.resolve_crop_radii(
                None,
                None,
                {"crop_radius_m": 0.075},
            ),
            (0.075, 0.075),
        )
        self.assertEqual(
            reconstruct_pipeline.resolve_crop_radii(
                0.12,
                0.085,
                {"crop_radius_m": 0.15, "registration_crop_radius_m": 0.09},
            ),
            (0.12, 0.085),
        )


class MultiStationPoseTests(unittest.TestCase):
    def test_manifest_captures_are_loaded_in_station_and_angle_order(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            filenames = [
                "frame_s01_x280.0_y+010.0.ply",
                "frame_s00_x200.0_y+010.0.ply",
                "frame_s01_x280.0_y+000.0.ply",
                "frame_s00_x200.0_y+000.0.ply",
            ]
            for filename in filenames:
                (input_dir / filename).touch()
            metadata = {
                "schema_version": 2,
                "captures": [
                    {
                        "filename": filename,
                        "station_index": int(filename[7:9]),
                        "x_position_mm": 280.0 if "x280" in filename else 200.0,
                        "x_offset_m": 0.08 if "x280" in filename else 0.0,
                        "angle_deg": 10.0 if "y+010" in filename else 0.0,
                    }
                    for filename in filenames
                ],
            }

            captures = reconstruct_pipeline.discover_captures(input_dir, metadata)

            self.assertEqual(
                [(item.station_index, item.angle_deg) for item in captures],
                [(0, 0.0), (0, 10.0), (1, 0.0), (1, 10.0)],
            )

    def test_second_station_pose_adds_80mm_along_the_orbit_axis(self):
        captures = [
            reconstruct_pipeline.CaptureRecord(
                path=Path("frame_s00_x200.0_y+030.0.ply"),
                angle_deg=30.0,
                station_index=0,
                x_position_mm=200.0,
                x_offset_m=0.0,
            ),
            reconstruct_pipeline.CaptureRecord(
                path=Path("frame_s01_x280.0_y+030.0.ply"),
                angle_deg=30.0,
                station_index=1,
                x_position_mm=280.0,
                x_offset_m=0.08,
            ),
        ]

        poses = reconstruct_pipeline.build_orbit_poses(
            captures,
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.1175]),
            reference_angle_deg=0.0,
            angle_sign=1.0,
        )

        np.testing.assert_allclose(
            poses[1][:3, 3] - poses[0][:3, 3],
            [0.08, 0.0, 0.0],
            atol=1e-12,
        )

    def test_registration_pairs_stay_within_stations_and_link_matching_angles(self):
        frames = [
            reconstruct_pipeline.RegistrationFrame(
                path=Path(f"frame_{station}_{angle}.ply"),
                angle_deg=angle,
                prior_pose=np.eye(4),
                station_index=station,
                x_position_mm=200.0 + station * 80.0,
                x_offset_m=station * 0.08,
            )
            for station in (0, 1)
            for angle in (-180.0, -170.0, 0.0, 180.0)
        ]

        pairs = reconstruct_pipeline.registration_pairs(frames)

        self.assertIn((0, 1, "sequential"), pairs)
        self.assertIn((0, 3, "loop"), pairs)
        self.assertIn((0, 4, "station"), pairs)
        self.assertNotIn((3, 4, "sequential"), pairs)

    def test_legacy_angle_only_files_remain_supported(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            for filename in ("frame_10.0.ply", "frame_0.0.ply"):
                (input_dir / filename).touch()

            captures = reconstruct_pipeline.discover_captures(input_dir, {})

            self.assertEqual([item.angle_deg for item in captures], [0.0, 10.0])
            self.assertTrue(all(item.station_index == 0 for item in captures))
            self.assertTrue(all(item.x_offset_m == 0.0 for item in captures))

    def test_measured_pose_priors_are_loaded_by_angle(self):
        angle_0_pose = np.eye(4)
        angle_10_pose = np.eye(4)
        angle_10_pose[:3, 3] = [0.002, -0.001, 0.020]
        pose_map = {
            "schema_version": 1,
            "x_position_mm": 150.0,
            "angles": [
                {
                    "angle_deg": 0.0,
                    "pose_valid": True,
                    "camera_to_reference": angle_0_pose.tolist(),
                },
                {
                    "angle_deg": 10.0,
                    "pose_valid": True,
                    "camera_to_reference": angle_10_pose.tolist(),
                },
            ],
        }
        captures = [
            reconstruct_pipeline.CaptureRecord(
                Path("frame_0.ply"), 0.0, x_position_mm=150.0
            ),
            reconstruct_pipeline.CaptureRecord(
                Path("frame_10.ply"), 10.0, x_position_mm=150.0
            ),
        ]

        poses = reconstruct_pipeline.build_measured_pose_priors(captures, pose_map)

        np.testing.assert_allclose(poses[0], angle_0_pose)
        np.testing.assert_allclose(poses[1], angle_10_pose)

    def test_measured_pose_map_rejects_a_different_x_station(self):
        pose_map = {
            "schema_version": 1,
            "x_position_mm": 150.0,
            "angles": [
                {
                    "angle_deg": 0.0,
                    "pose_valid": True,
                    "camera_to_reference": np.eye(4).tolist(),
                }
            ],
        }
        captures = [
            reconstruct_pipeline.CaptureRecord(
                Path("frame_0.ply"), 0.0, x_position_mm=200.0
            )
        ]

        with self.assertRaisesRegex(ValueError, "X=150.0 mm"):
            reconstruct_pipeline.build_measured_pose_priors(captures, pose_map)


class GuardedIcpTests(unittest.TestCase):
    def test_rejects_an_icp_result_that_undoes_one_motor_step(self):
        prior = reconstruct_pipeline.rotation_about_axis(
            10.0,
            np.array([0.0, 0.0, 0.1175]),
            np.array([1.0, 0.0, 0.0]),
        )

        accepted, reason, correction_m, correction_deg = (
            reconstruct_pipeline.registration_result_is_acceptable(
                fitness=0.50,
                rmse_m=0.0012,
                prior_fitness=0.45,
                prior_rmse_m=0.0013,
                prior=prior,
                candidate=np.eye(4),
            )
        )

        self.assertFalse(accepted)
        self.assertIn("correction exceeds", reason)
        self.assertAlmostEqual(correction_m, 0.0204816, places=6)
        self.assertAlmostEqual(correction_deg, 10.0, places=6)

    def test_accepts_a_small_correction_that_improves_the_prior(self):
        candidate = np.eye(4)
        candidate[2, 3] = 0.001

        accepted, reason, _, _ = (
            reconstruct_pipeline.registration_result_is_acceptable(
                fitness=0.50,
                rmse_m=0.0012,
                prior_fitness=0.45,
                prior_rmse_m=0.0013,
                prior=np.eye(4),
                candidate=candidate,
            )
        )

        self.assertTrue(accepted)
        self.assertEqual(reason, "accepted")

    def test_rejects_rmse_too_close_to_the_correspondence_limit(self):
        accepted, reason, _, _ = (
            reconstruct_pipeline.registration_result_is_acceptable(
                fitness=0.50,
                rmse_m=0.00175,
                prior_fitness=0.45,
                prior_rmse_m=0.0020,
                prior=np.eye(4),
                candidate=np.eye(4),
            )
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "RMSE exceeds threshold")

    def test_rejects_a_small_correction_that_does_not_improve_the_prior(self):
        candidate = np.eye(4)
        candidate[2, 3] = 0.001

        accepted, reason, _, _ = (
            reconstruct_pipeline.registration_result_is_acceptable(
                fitness=0.45,
                rmse_m=0.0013,
                prior_fitness=0.45,
                prior_rmse_m=0.0013,
                prior=np.eye(4),
                candidate=candidate,
            )
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "ICP did not improve the pose prior")

    def test_registration_cloud_removes_statistical_outliers_before_icp(self):
        class FakeCloud:
            def __init__(self, points):
                self.points = np.asarray(points, dtype=float)
                self.sor_call = None

            def select_by_index(self, indices):
                return FakeCloud(self.points[indices])

            def voxel_down_sample(self, _voxel_size):
                return self

            def remove_statistical_outlier(self, nb_neighbors, std_ratio):
                self.sor_call = (nb_neighbors, std_ratio)
                filtered = FakeCloud(self.points[:-1])
                filtered.sor_call = self.sor_call
                return filtered, list(range(len(filtered.points)))

            def estimate_normals(self, _search):
                return None

        class FakeGeometry:
            PointCloud = FakeCloud

            class KDTreeSearchParamHybrid:
                def __init__(self, **_kwargs):
                    pass

        class FakeOpen3D:
            geometry = FakeGeometry

        cloud = FakeCloud(np.zeros((25, 3)))

        prepared = reconstruct_pipeline.prepare_registration_cloud(
            FakeOpen3D,
            cloud,
            np.eye(4),
        )

        self.assertEqual(len(prepared.points), 24)
        self.assertEqual(prepared.sor_call, (20, 1.5))

    def test_registration_crop_excludes_points_retained_by_the_final_crop(self):
        class FakeCloud:
            def __init__(self, points):
                self.points = np.asarray(points, dtype=float)

            def select_by_index(self, indices):
                return FakeCloud(self.points[indices])

            def voxel_down_sample(self, _voxel_size):
                return self

            def estimate_normals(self, _search):
                return None

        class FakeGeometry:
            PointCloud = FakeCloud

            class KDTreeSearchParamHybrid:
                def __init__(self, **_kwargs):
                    pass

        class FakeOpen3D:
            geometry = FakeGeometry

        points = np.array([
            [0.000, 0.0, 0.100],
            [0.099, 0.0, 0.100],
            [0.120, 0.0, 0.100],
        ])
        pivot = np.array([0.0, 0.0, 0.1])
        registration_bounds = reconstruct_pipeline.crop_bounds_around(pivot, 0.10)
        final_bounds = reconstruct_pipeline.crop_bounds_around(pivot, 0.15)

        prepared = reconstruct_pipeline.prepare_registration_cloud(
            FakeOpen3D,
            FakeCloud(points),
            np.eye(4),
            crop_bounds=registration_bounds,
        )

        self.assertEqual(len(prepared.points), 2)
        self.assertEqual(
            np.count_nonzero(
                reconstruct_pipeline.points_inside_bounds(points, final_bounds)
            ),
            3,
        )


class ReconstructionDiagnosticsTests(unittest.TestCase):
    def test_station_edge_records_both_x_positions(self):
        frames = [
            reconstruct_pipeline.RegistrationFrame(
                path=Path(f"frame_s0{station}.ply"),
                angle_deg=0.0,
                prior_pose=np.eye(4),
                station_index=station,
                x_position_mm=200.0 + 80.0 * station,
                x_offset_m=0.08 * station,
            )
            for station in (0, 1)
        ]
        edge = reconstruct_pipeline.RegistrationEdge(
            source_id=0,
            target_id=1,
            kind="station",
            transform=np.eye(4),
            information=np.eye(6),
            accepted=False,
            reason="used pose prior",
            fitness=0.0,
            rmse_m=float("inf"),
            correction_m=0.0,
            correction_deg=0.0,
        )

        diagnostics = reconstruct_pipeline.build_diagnostics(
            frames=frames,
            optimized_poses=[np.eye(4), np.eye(4)],
            edges=[edge],
            settings={},
        )

        recorded = diagnostics["edges"][0]
        self.assertEqual(recorded["kind"], "station")
        self.assertEqual(recorded["source_x_position_mm"], 200.0)
        self.assertEqual(recorded["target_x_position_mm"], 280.0)
        self.assertEqual(recorded["target_x_offset_m"], 0.08)

    def test_records_effective_radius_mode_and_zero_motor_pose_correction(self):
        prior = reconstruct_pipeline.rotation_about_axis(
            30.0,
            np.array([0.0, 0.0, 0.1175]),
            np.array([1.0, 0.0, 0.0]),
        )
        frame = reconstruct_pipeline.RegistrationFrame(
            path=Path("frame_30.0.ply"),
            angle_deg=30.0,
            prior_pose=prior,
        )

        diagnostics = reconstruct_pipeline.build_diagnostics(
            frames=[frame],
            optimized_poses=[prior.copy()],
            edges=[],
            settings={
                "registration_mode": "motor",
                "orbit_radius_m": 0.1175,
                "orbit_axis": [1.0, 0.0, 0.0],
            },
        )

        self.assertEqual(diagnostics["settings"]["registration_mode"], "motor")
        self.assertEqual(diagnostics["settings"]["orbit_radius_m"], 0.1175)
        self.assertEqual(
            diagnostics["optimized_pose_corrections"][0]["translation_m"],
            0.0,
        )


class ReconstructionEndToEndTests(unittest.TestCase):
    def test_aruco_pose_pipeline_uses_measured_full_transforms_without_a_radius(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_dir = root / "scan"
            output_dir = root / "reconstruction"
            input_dir.mkdir()

            reference_points = np.array(
                [
                    [x, y, z]
                    for x in np.linspace(-0.015, 0.015, 8)
                    for y in np.linspace(-0.010, 0.010, 6)
                    for z in (0.035, 0.045)
                ]
            )
            colors = np.tile([0.7, 0.2, 0.1], (len(reference_points), 1))
            angle_0_pose = np.eye(4)
            angle_90_pose = reconstruct_pipeline.rotation_about_axis(
                12.0, np.array([0.0, 0.0, 0.04]), np.array([0.2, 0.9, 0.1])
            )
            filenames = [
                "frame_s00_x150.0_y+000.0.ply",
                "frame_s00_x150.0_y+090.0.ply",
            ]
            for filename, camera_to_reference in zip(
                filenames, (angle_0_pose, angle_90_pose)
            ):
                cloud = o3d.geometry.PointCloud()
                raw_points = reconstruct_pipeline.transformed_points(
                    reference_points, np.linalg.inv(camera_to_reference)
                )
                cloud.points = o3d.utility.Vector3dVector(raw_points)
                cloud.colors = o3d.utility.Vector3dVector(colors)
                self.assertTrue(o3d.io.write_point_cloud(str(input_dir / filename), cloud))

            metadata = {
                "schema_version": 2,
                "captures": [
                    {
                        "filename": filename,
                        "station_index": 0,
                        "x_position_mm": 150.0,
                        "x_offset_m": 0.0,
                        "angle_deg": angle,
                    }
                    for filename, angle in zip(filenames, (0.0, 90.0))
                ],
            }
            (input_dir / "scan_metadata.json").write_text(json.dumps(metadata))
            pose_map = {
                "schema_version": 1,
                "quality_status": "valid",
                "x_position_mm": 150.0,
                "reference_angle_deg": 0.0,
                "profile_origin_in_reference_m": [0.0, 0.0, 0.04],
                "orbit_axis_reference": [1.0, 0.0, 0.0],
                "orbit_fit_profile_frame": None,
                "angles": [
                    {
                        "angle_deg": angle,
                        "pose_valid": True,
                        "camera_to_reference": pose.tolist(),
                    }
                    for angle, pose in ((0.0, angle_0_pose), (90.0, angle_90_pose))
                ],
            }
            pose_map_path = root / "orbit_pose_map.json"
            pose_map_path.write_text(json.dumps(pose_map))

            reconstruct_pipeline.main(
                [
                    "--input-dir", str(input_dir),
                    "--output-dir", str(output_dir),
                    "--registration-mode", "aruco",
                    "--pose-map", str(pose_map_path),
                    "--crop-radius-m", "0.08",
                    "--skip-per-scan-sor",
                ]
            )

            merged = o3d.io.read_point_cloud(str(output_dir / "merged_cloud.ply"))
            bounds = merged.get_axis_aligned_bounding_box()
            np.testing.assert_allclose(bounds.get_min_bound(), reference_points.min(axis=0), atol=0.002)
            np.testing.assert_allclose(bounds.get_max_bound(), reference_points.max(axis=0), atol=0.002)
            diagnostics = json.loads(
                (output_dir / "registration_diagnostics.json").read_text()
            )
            self.assertEqual(diagnostics["settings"]["registration_mode"], "aruco")
            self.assertEqual(diagnostics["settings"]["orbit_radius_m"], None)

    def test_two_station_motor_pipeline_fuses_both_x_positions(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_dir = root / "scan"
            output_dir = root / "reconstruction"
            input_dir.mkdir()

            phi = np.linspace(0.2, np.pi - 0.2, 10)
            theta = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
            world_points = np.array([
                [
                    0.04 + 0.02 * np.sin(p) * np.cos(t),
                    0.02 * np.sin(p) * np.sin(t),
                    0.1 + 0.02 * np.cos(p),
                ]
                for p in phi
                for t in theta
            ])
            colors = np.tile([0.8, 0.3, 0.1], (len(world_points), 1))
            filenames = [
                "frame_s00_x200.0_y+000.0.ply",
                "frame_s01_x280.0_y+000.0.ply",
            ]
            for filename, offset in zip(filenames, (0.0, 0.08)):
                cloud = o3d.geometry.PointCloud()
                local_points = world_points - np.array([offset, 0.0, 0.0])
                cloud.points = o3d.utility.Vector3dVector(local_points)
                cloud.colors = o3d.utility.Vector3dVector(colors)
                self.assertTrue(o3d.io.write_point_cloud(str(input_dir / filename), cloud))

            metadata = {
                "schema_version": 2,
                "orbit_radius_m": 0.1,
                "orbit_axis": [1.0, 0.0, 0.0],
                "reconstruction": {"crop_radius_m": 0.08},
                "captures": [
                    {
                        "filename": filename,
                        "station_index": station,
                        "x_position_mm": 200.0 + 80.0 * station,
                        "x_offset_m": 0.08 * station,
                        "angle_deg": 0.0,
                    }
                    for station, filename in enumerate(filenames)
                ],
            }
            (input_dir / "scan_metadata.json").write_text(json.dumps(metadata))

            reconstruct_pipeline.main([
                "--input-dir", str(input_dir),
                "--output-dir", str(output_dir),
                "--registration-mode", "motor",
                "--skip-per-scan-sor",
            ])

            merged = o3d.io.read_point_cloud(str(output_dir / "merged_cloud.ply"))
            center = np.mean(np.asarray(merged.points), axis=0)
            self.assertAlmostEqual(center[0], 0.04, delta=0.005)
            diagnostics = json.loads(
                (output_dir / "registration_diagnostics.json").read_text()
            )
            self.assertEqual(diagnostics["capture_count"], 2)
            self.assertEqual(len(diagnostics["settings"]["x_stations"]), 2)
            self.assertEqual(
                diagnostics["settings"]["processing"]["merge"]["input_clouds"],
                2,
            )

    def test_motor_pipeline_produces_compatible_outputs_without_cloudcompare(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_dir = root / "scan"
            output_dir = root / "reconstruction"
            input_dir.mkdir()

            phi = np.linspace(0.2, np.pi - 0.2, 10)
            theta = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
            points = np.array([
                [
                    0.02 * np.sin(p) * np.cos(t),
                    0.02 * np.sin(p) * np.sin(t),
                    0.1 + 0.02 * np.cos(p),
                ]
                for p in phi
                for t in theta
            ])
            colors = np.tile([0.8, 0.3, 0.1], (len(points), 1))
            for angle in (0.0, 5.0, 10.0):
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(points)
                cloud.colors = o3d.utility.Vector3dVector(colors)
                self.assertTrue(
                    o3d.io.write_point_cloud(
                        str(input_dir / f"frame_{angle:.1f}.ply"),
                        cloud,
                        write_ascii=False,
                    )
                )

            reconstruct_pipeline.main([
                "--input-dir", str(input_dir),
                "--output-dir", str(output_dir),
                "--orbit-radius-m", "0.1",
                "--crop-radius-m", "0.05",
                "--registration-mode", "motor",
                "--skip-per-scan-sor",
            ])

            transformed = sorted((output_dir / "01_transformed").glob("*.ply"))
            self.assertEqual(len(transformed), 3)
            self.assertTrue((output_dir / "matrices" / "frame_5.0_prior.txt").is_file())
            self.assertTrue(
                (output_dir / "matrices" / "frame_5.0_optimized_matrix.txt").is_file()
            )
            self.assertTrue((output_dir / "processing.log").is_file())
            self.assertFalse((output_dir / "cloudcompare.log").exists())

            merged = o3d.io.read_point_cloud(str(output_dir / "merged_cloud.ply"))
            self.assertTrue(merged.has_colors())
            self.assertTrue(merged.has_normals())
            diagnostics = json.loads(
                (output_dir / "registration_diagnostics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                diagnostics["settings"]["processing_backend"],
                "open3d+trimesh",
            )
            self.assertEqual(diagnostics["settings"]["crop_radius_m"], 0.05)
            self.assertEqual(
                diagnostics["settings"]["registration_crop_radius_m"],
                0.05,
            )
            self.assertEqual(
                diagnostics["settings"]["processing"]["merge"]["input_clouds"],
                3,
            )


if __name__ == "__main__":
    unittest.main()


class CylinderCropRegionTests(unittest.TestCase):
    def test_crop_bounds_around_defaults_to_a_cube(self):
        bounds = reconstruct_pipeline.crop_bounds_around(
            np.array([0.0, 0.0, 0.1]), 0.15,
        )
        np.testing.assert_allclose(
            bounds, (-0.15, -0.15, -0.05, 0.15, 0.15, 0.25), atol=1e-12,
        )

    def test_crop_bounds_around_builds_a_cylinder_when_given_an_axis(self):
        crop = reconstruct_pipeline.crop_bounds_around(
            np.array([0.0, 0.0, 0.1]),
            0.08,
            axis=np.array([1.0, 0.0, 0.0]),
            axial_half_length_m=0.15,
        )
        self.assertEqual(crop.radius_m, 0.08)
        self.assertEqual(crop.axial_half_length_m, 0.15)
        self.assertEqual(crop.center, (0.0, 0.0, 0.1))

    def test_a_cylinder_drops_the_enclosure_ring_a_cube_keeps(self):
        pivot = np.array([0.0, 0.0, 0.1])
        axis = np.array([1.0, 0.0, 0.0])
        # A forearm point 130 mm along the axis, and a ring point at the
        # orbit radius that a cube large enough to keep it would also keep.
        points = np.array([
            [0.130, 0.000, 0.100],
            [0.000, 0.070, 0.170],
        ])
        cube = reconstruct_pipeline.crop_bounds_around(pivot, 0.15)
        cylinder = reconstruct_pipeline.crop_bounds_around(
            pivot, 0.08, axis=axis, axial_half_length_m=0.15,
        )
        self.assertEqual(
            reconstruct_pipeline.points_inside_crop(points, cube).tolist(),
            [True, True],
        )
        self.assertEqual(
            reconstruct_pipeline.points_inside_crop(points, cylinder).tolist(),
            [True, False],
        )

    def test_axial_half_length_resolution_order(self):
        self.assertEqual(
            reconstruct_pipeline.resolve_crop_axial_half_length(0.2, {}), 0.2,
        )
        self.assertEqual(
            reconstruct_pipeline.resolve_crop_axial_half_length(
                None, {"crop_axial_half_length_m": 0.18},
            ),
            0.18,
        )
        self.assertEqual(
            reconstruct_pipeline.resolve_crop_axial_half_length(None, {}), 0.15,
        )
        with self.assertRaises(ValueError):
            reconstruct_pipeline.resolve_crop_axial_half_length(-0.1, {})

    def test_crop_shape_defaults_to_cube_and_accepts_cylinder(self):
        args = reconstruct_pipeline.parse_args(["--input-dir", "scan"])
        self.assertEqual(args.crop_shape, "cube")
        args = reconstruct_pipeline.parse_args(
            ["--input-dir", "scan", "--crop-shape", "cylinder",
             "--crop-axial-half-length-m", "0.16"],
        )
        self.assertEqual(args.crop_shape, "cylinder")
        self.assertEqual(args.crop_axial_half_length_m, 0.16)


class HighDriftFrameExclusionTests(unittest.TestCase):
    @staticmethod
    def edge(
        *,
        target_id=1,
        kind="sequential",
        correction_m=0.0,
        correction_deg=0.0,
        accepted=False,
    ):
        return reconstruct_pipeline.RegistrationEdge(
            source_id=target_id - 1,
            target_id=target_id,
            kind=kind,
            transform=np.eye(4),
            information=np.eye(6),
            accepted=accepted,
            reason="test edge",
            fitness=0.5,
            rmse_m=0.001,
            correction_m=correction_m,
            correction_deg=correction_deg,
        )

    def test_filter_is_opt_in(self):
        args = reconstruct_pipeline.parse_args(["--input-dir", "scan"])
        self.assertFalse(args.exclude_high_drift_frames)

        args = reconstruct_pipeline.parse_args(
            ["--input-dir", "scan", "--exclude-high-drift-frames"]
        )
        self.assertTrue(args.exclude_high_drift_frames)

    def test_excludes_sequential_target_above_translation_limit(self):
        edge = self.edge(
            target_id=2,
            correction_m=reconstruct_pipeline.ICP_MAX_CORRECTION_M + 0.001,
        )

        self.assertEqual(
            reconstruct_pipeline.high_drift_frame_ids([edge]),
            {2},
        )

    def test_excludes_sequential_target_above_rotation_limit(self):
        edge = self.edge(
            target_id=3,
            correction_deg=reconstruct_pipeline.ICP_MAX_CORRECTION_DEG + 0.1,
        )

        self.assertEqual(
            reconstruct_pipeline.high_drift_frame_ids([edge]),
            {3},
        )

    def test_does_not_exclude_for_non_sequential_or_ordinary_rejection(self):
        edges = [
            self.edge(target_id=1, correction_m=0.001),
            self.edge(
                target_id=2,
                kind="loop",
                correction_m=reconstruct_pipeline.ICP_MAX_CORRECTION_M + 0.001,
            ),
            self.edge(
                target_id=3,
                kind="station",
                correction_deg=reconstruct_pipeline.ICP_MAX_CORRECTION_DEG + 0.1,
            ),
        ]

        self.assertEqual(reconstruct_pipeline.high_drift_frame_ids(edges), set())


class CalibrationStationValidationTests(unittest.TestCase):
    @staticmethod
    def _captures(x_position_mm: float):
        return [
            reconstruct_pipeline.CaptureRecord(
                path=Path(f"frame_s00_x{x_position_mm}_y+000.0.ply"),
                angle_deg=0.0,
                station_index=0,
                x_position_mm=x_position_mm,
                x_offset_m=0.0,
            )
        ]

    def test_a_matching_station_is_accepted(self):
        reconstruct_pipeline.validate_calibration_station(
            {"motor": {"x_position_mm": 50.0}},
            self._captures(50.0),
            Path("radius_calibration.json"),
        )

    def test_a_mismatched_station_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            reconstruct_pipeline.validate_calibration_station(
                {"motor": {"x_position_mm": 100.0}},
                self._captures(50.0),
                Path("radius_calibration.json"),
            )
        self.assertIn("X=100.0", str(raised.exception))
        self.assertIn("X=50.0", str(raised.exception))

    def test_a_calibration_without_a_station_only_warns(self):
        with self.assertLogs(reconstruct_pipeline.logger, level="WARNING"):
            reconstruct_pipeline.validate_calibration_station(
                {}, self._captures(50.0), Path("radius_calibration.json"),
            )

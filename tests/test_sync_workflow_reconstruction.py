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
            self.assertEqual(
                diagnostics["settings"]["processing"]["merge"]["input_clouds"],
                3,
            )


if __name__ == "__main__":
    unittest.main()

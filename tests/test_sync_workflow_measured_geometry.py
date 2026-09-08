"""Calibration handoff and real reconstruction coverage without scan hardware."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/sync_workflow"))
import main_scan
import reconstruct_pipeline as reconstruction
import orbit_geometry as geometry


def calibration():
    return {
        "quality_status": "valid", "recommended_radius_m": 0.14,
        "orbit_geometry": {"pivot_m": [0.004, 0.001, 0.14], "axis": [1, 0, 0.02]},
        "motor": {"x_position_mm": 100},
        "camera": {"pointcloud_coordinate_frame": "color"},
    }


class MeasuredGeometryTests(unittest.TestCase):
    def test_file_precedence_and_default_independent_of_working_directory(self):
        self.assertTrue(geometry.DEFAULT_ORBIT_GEOMETRY.is_absolute())
        self.assertEqual(geometry.select_orbit_geometry(), geometry.DEFAULT_ORBIT_GEOMETRY)
        self.assertEqual(geometry.select_orbit_geometry(None, {"orbit_geometry_source": "/tmp/recorded.json"}), Path("/tmp/recorded.json"))
        self.assertEqual(geometry.select_orbit_geometry("/tmp/explicit.json", {"orbit_geometry_source": "/tmp/recorded.json"}), Path("/tmp/explicit.json"))

    def test_capture_forwards_measured_geometry_and_default_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps(calibration()))
            args = main_scan.parse_args(["--orbit-geometry", str(path), "--reconstruct"])
            main_scan.prepare_scan_geometry(args)
            metadata = main_scan.build_scan_metadata(args, active_disparity=256, captured_angles=[0])
            forwarded = reconstruction.parse_args(main_scan.build_reconstruction_command(args)[2:])
            self.assertEqual(metadata["orbit_geometry_source"], str(path))
            self.assertEqual(metadata["orbit_radius_m"], 0.14)
            self.assertEqual(forwarded.orbit_geometry, path)
            self.assertEqual(forwarded.registration_mode, "guarded-icp")
            self.assertAlmostEqual(np.linalg.norm(metadata["orbit_axis"]), 1)
            args.radius_m = 0.145
            main_scan.prepare_scan_geometry(args)
            self.assertEqual(args.radius_m, 0.145)

    def test_bad_calibration_fails_before_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.json"
            args = main_scan.parse_args(["--orbit-geometry", str(path)])
            with mock.patch.object(main_scan, "parse_args", return_value=args), mock.patch.object(main_scan, "MotorController") as motor, mock.patch.object(main_scan, "CameraController") as camera:
                with self.assertRaisesRegex(ValueError, "Cannot read orbit calibration"):
                    main_scan.main()
                motor.assert_not_called()
                camera.assert_not_called()
            for change in [{"quality_status": "invalid"}, {"orbit_geometry": {}}, {"recommended_radius_m": float("nan")}, {"motor": {}}, {"orbit_geometry": {"pivot_m": [0, 0, 0.14], "axis": [0, 0, 0]}}]:
                with self.subTest(change=change):
                    path.write_text(json.dumps(calibration() | change))
                    with self.assertRaises(ValueError):
                        geometry.load_orbit_geometry(path)

    def test_camera_frame_must_match(self):
        with self.assertRaisesRegex(ValueError, "coordinate-frame mismatch"):
            geometry.validate_camera_frame(calibration(), {"coordinate_frame": "depth"})

    def test_reconstruction_uses_recorded_calibration_across_stations_and_guarded_icp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "calibration.json"
            path.write_text(json.dumps(calibration()))
            loaded = geometry.load_orbit_geometry(path)
            axis = np.array(loaded["orbit_geometry"]["axis"])
            pivot = np.array(loaded["orbit_geometry"]["pivot_m"])
            # Object is centred at the station-adjusted crop centre for X=50.
            center = pivot + axis * 0.05
            rng = np.random.default_rng(12)
            points = rng.normal(size=(8000, 3))
            points /= np.linalg.norm(points, axis=1)[:, None]
            points = center + points * [0.015, 0.02, 0.01]
            captures = []
            for angle in [0, 10]:
                transform = reconstruction.rotation_about_axis(angle, pivot, axis)
                local = (points - transform[:3, 3]) @ transform[:3, :3]
                filename = f"frame_{angle}.ply"
                cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(local))
                cloud.colors = o3d.utility.Vector3dVector(np.ones_like(local) * 0.5)
                o3d.io.write_point_cloud(str(root / filename), cloud)
                captures.append({"filename": filename, "angle_deg": angle, "station_index": 0, "x_position_mm": 50, "x_offset_m": 0})
            (root / "scan_metadata.json").write_text(json.dumps({"orbit_geometry_source": str(path), "orbit_radius_m": 9, "captures": captures}))
            reconstruction.main(["--input-dir", str(root), "--crop-radius-m", "0.03"])
            output = root / "reconstruction"
            diagnostics = json.loads((output / "registration_diagnostics.json").read_text())
            settings = diagnostics["settings"]
            self.assertEqual(settings["registration_mode"], "guarded-icp")
            self.assertEqual(settings["orbit_radius_m"], 0.14)
            self.assertEqual(settings["orbit_radius_source"], "orbit_geometry")
            self.assertEqual(settings["calibration_x_position_mm"], 100)
            np.testing.assert_allclose(settings["pivot"], pivot)
            np.testing.assert_allclose(settings["orbit_axis"], axis)
            self.assertGreater(len(diagnostics["edges"]), 0)
            merged = o3d.io.read_point_cloud(str(output / "merged_cloud.ply"))
            self.assertGreater(len(merged.points), 100)
            np.testing.assert_allclose(np.mean(merged.points, axis=0), center, atol=0.003)

            # Explicit overrides still win and motor mode skips ICP.
            override_output = root / "override"
            reconstruction.main([
                "--input-dir", str(root), "--output-dir", str(override_output),
                "--orbit-geometry", str(path), "--registration-mode", "motor",
                "--orbit-radius-m", "0.15", "--pivot", *map(str, center),
                "--crop-radius-m", "0.03",
            ])
            override = json.loads((override_output / "registration_diagnostics.json").read_text())
            self.assertEqual(override["edges"], [])
            self.assertEqual(override["settings"]["orbit_radius_m"], 0.15)
            self.assertEqual(override["settings"]["orbit_radius_source"], "cli")
            self.assertIsNone(override["settings"]["calibration_x_position_mm"])
            np.testing.assert_allclose(override["settings"]["pivot"], center)

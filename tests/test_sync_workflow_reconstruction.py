from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


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


if __name__ == "__main__":
    unittest.main()

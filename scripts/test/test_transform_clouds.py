import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transform_clouds_pipeline import (
    CLOUDCOMPARE_COMMAND,
    RegistrationEdge,
    build_orbit_poses,
    cloudcompare_file_argument,
    edge_as_json,
    evaluate_registration_quality,
    find_ply_files,
    parse_args,
    registration_pairs,
    registration_result_is_acceptable,
    rotation_about_axis,
    runtime_configuration,
)


class TransformCloudsGeometryTests(unittest.TestCase):
    def test_orbit_poses_do_not_require_stage_planes(self):
        paths = [
            Path("angle_005p_00.ply"),
            Path("angle_010p_00.ply"),
        ]
        axis = np.array([0.0, 1.0, 0.0])

        poses = build_orbit_poses(paths, axis)

        self.assertEqual(len(poses), 2)
        np.testing.assert_allclose(poses[0], np.eye(4), atol=1e-12)
        np.testing.assert_allclose(
            poses[1],
            rotation_about_axis(
                5.0,
                np.array([0.025, 0.0, 0.175]),
                axis,
            ),
            atol=1e-12,
        )

    def test_runtime_configuration_accepts_lingbot_capture_directory(self):
        args = parse_args(
            [
                "--input-dir",
                "captures/Plastic_Hand_full_lingbot",
                "--reference-angle-deg",
                "10",
                "--orbit-radius-m",
                "0.18",
            ]
        )

        configuration = runtime_configuration(args)

        self.assertEqual(
            configuration.input_dir,
            Path("captures/Plastic_Hand_full_lingbot"),
        )
        self.assertEqual(
            configuration.output_dir,
            Path("captures/Plastic_Hand_full_lingbot")
            / "reconstruction_new_without_platform_alignment",
        )
        self.assertEqual(configuration.reference_angle_deg, 10.0)
        self.assertEqual(configuration.orbit_radius_m, 0.18)
        np.testing.assert_allclose(
            configuration.pivot_in_reference,
            [0.025, 0.0, 0.18],
        )

    def test_rotation_about_arbitrary_axis_preserves_pivot_and_axis(self):
        pivot = np.array([0.025, 0.0, 0.175])
        axis = np.array([0.03, 0.999, -0.026])
        axis /= np.linalg.norm(axis)

        transform = rotation_about_axis(137.0, pivot, axis)

        np.testing.assert_allclose(
            transform @ np.r_[pivot, 1.0],
            np.r_[pivot, 1.0],
            atol=1e-12,
        )
        point_on_axis = pivot + axis * 0.4
        np.testing.assert_allclose(
            transform @ np.r_[point_on_axis, 1.0],
            np.r_[point_on_axis, 1.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            transform[:3, :3].T @ transform[:3, :3],
            np.eye(3),
            atol=1e-12,
        )
        self.assertAlmostEqual(np.linalg.det(transform[:3, :3]), 1.0)

    def test_registration_guard_rejects_large_correction(self):
        prior = np.eye(4)
        candidate = np.eye(4)
        candidate[0, 3] = 0.02

        accepted, reason, correction_m, _ = (
            registration_result_is_acceptable(
                fitness=0.9,
                rmse_m=0.001,
                prior=prior,
                candidate=candidate,
                min_fitness=0.3,
                max_rmse_m=0.004,
                max_correction_m=0.01,
                max_correction_deg=4.0,
            )
        )

        self.assertFalse(accepted)
        self.assertIn("translation", reason)
        self.assertAlmostEqual(correction_m, 0.02)

    def test_registration_pairs_include_neighbors_and_loop_closure(self):
        pairs = registration_pairs(
            frame_count=5,
            neighbor_span=2,
            include_loop_closure=True,
        )

        self.assertEqual(
            pairs,
            [
                (0, 1, "sequential"),
                (1, 2, "sequential"),
                (2, 3, "sequential"),
                (3, 4, "sequential"),
                (0, 2, "neighbor"),
                (1, 3, "neighbor"),
                (2, 4, "neighbor"),
                (0, 4, "loop"),
            ],
        )

    def test_quality_gate_accepts_well_constrained_registration(self):
        poses = [np.eye(4) for _ in range(5)]
        edges = [
            self.registration_edge(index, index + 1, "sequential")
            for index in range(4)
        ]
        edges.append(self.registration_edge(0, 4, "loop"))

        quality = evaluate_registration_quality(
            optimized_poses=poses,
            initial_poses=poses,
            edges=edges,
            minimum_sequential_usable_fraction=0.9,
            maximum_loop_correction_m=0.003,
            maximum_loop_correction_deg=1.0,
        )

        self.assertTrue(quality["passed"])
        self.assertEqual(quality["failure_reasons"], [])

    def test_quality_gate_rejects_missing_loop_closure(self):
        poses = [np.eye(4), np.eye(4)]

        quality = evaluate_registration_quality(
            optimized_poses=poses,
            initial_poses=poses,
            edges=[],
            minimum_sequential_usable_fraction=0.9,
            maximum_loop_correction_m=0.003,
            maximum_loop_correction_deg=1.0,
        )

        self.assertFalse(quality["passed"])
        self.assertTrue(
            any("loop closure" in reason for reason in quality["failure_reasons"])
        )

    def test_quality_gate_counts_guarded_prior_fallback_as_usable(self):
        sequential = self.registration_edge(
            0,
            1,
            "sequential",
            accepted=False,
            usable=True,
        )
        loop = self.registration_edge(0, 1, "loop")
        poses = [np.eye(4), np.eye(4)]

        quality = evaluate_registration_quality(
            optimized_poses=poses,
            initial_poses=poses,
            edges=[sequential, loop],
            minimum_sequential_usable_fraction=0.9,
            maximum_loop_correction_m=0.003,
            maximum_loop_correction_deg=1.0,
        )

        self.assertTrue(quality["passed"])
        self.assertEqual(quality["sequential_usable_fraction"], 1.0)
        self.assertEqual(quality["sequential_icp_acceptance_fraction"], 0.0)

    def test_loop_quality_compares_optimized_poses_with_orbit_priors(self):
        poses = [np.eye(4), np.eye(4)]
        loop = self.registration_edge(0, 1, "loop")

        quality = evaluate_registration_quality(
            optimized_poses=poses,
            initial_poses=poses,
            edges=[loop],
            minimum_sequential_usable_fraction=0.0,
            maximum_loop_correction_m=0.003,
            maximum_loop_correction_deg=1.0,
        )

        self.assertTrue(quality["loop_closure_passed"])
        self.assertEqual(quality["loop_closure_translation_m"], 0.0)
        self.assertEqual(quality["loop_closure_rotation_deg"], 0.0)

    def test_edge_diagnostics_convert_non_finite_values_to_json_null(self):
        edge = self.registration_edge(
            0,
            1,
            "sequential",
            accepted=False,
            rmse_m=float("inf"),
        )

        encoded = json.dumps({"edge": edge_as_json(edge)}, allow_nan=False)
        decoded = json.loads(encoded)

        self.assertIsNone(decoded["edge"]["rmse_m"])
        self.assertIsNone(decoded["edge"]["prior_rmse_m"])

    @staticmethod
    def registration_edge(
        source_id,
        target_id,
        kind,
        *,
        accepted=True,
        usable=False,
        rmse_m=0.001,
    ):
        return RegistrationEdge(
            source_id=source_id,
            target_id=target_id,
            kind=kind,
            transform=np.eye(4),
            information=np.eye(6),
            accepted=accepted,
            reason="accepted" if accepted else "used motor prior",
            fitness=0.9,
            rmse_m=rmse_m,
            correction_m=0.001,
            correction_deg=0.2,
            usable=usable,
        )


class TransformCloudsCompatibilityTests(unittest.TestCase):
    def test_cloudcompare_uses_flatpak(self):
        self.assertEqual(
            CLOUDCOMPARE_COMMAND,
            ["/usr/bin/flatpak", "run", "org.cloudcompare.CloudCompare"],
        )

    def test_find_ply_files_orders_angles_numerically(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            capture_dir = Path(directory)
            five = capture_dir / "angle_0005p00.ply"
            ten = capture_dir / "angle_0010p00.ply"
            five.touch()
            ten.touch()
            (capture_dir / "unrelated.ply").touch()

            self.assertEqual(find_ply_files(capture_dir), [five, ten])

    def test_save_filename_is_quoted_for_cloudcompare_internal_parser(self):
        output_path = Path("/media/Shared Data/transformed cloud.ply")

        self.assertEqual(
            cloudcompare_file_argument(output_path),
            f'"{output_path}"',
        )


if __name__ == "__main__":
    unittest.main()

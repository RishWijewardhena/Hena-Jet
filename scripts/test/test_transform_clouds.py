import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transform_clouds import (
    PlaneModel,
    RegistrationEdge,
    build_plane_corrected_poses,
    cloudcompare_command,
    cloudcompare_file_argument,
    estimate_orbit_axis,
    enforce_platform_alignment,
    edge_as_json,
    evaluate_registration_quality,
    fit_platform_plane,
    find_ply_files,
    plane_alignment_transform,
    plane_as_json,
    refine_plane_least_squares,
    registration_pairs,
    registration_result_is_acceptable,
    rotation_about_axis,
    transform_plane,
)


class TransformCloudsGeometryTests(unittest.TestCase):
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

    def test_plane_alignment_corrects_normal_and_offset(self):
        source = PlaneModel(
            normal=np.array([0.04, 0.9987, -0.03]),
            offset=-0.062,
            inlier_count=5000,
            candidate_count=6000,
            rmse_m=0.0005,
            reliable=True,
            reason="accepted",
        ).normalized()
        target = PlaneModel(
            normal=np.array([0.0, 1.0, 0.0]),
            offset=-0.060,
            inlier_count=5000,
            candidate_count=6000,
            rmse_m=0.0005,
            reliable=True,
            reason="accepted",
        )

        correction = plane_alignment_transform(source, target)
        corrected = transform_plane(source, correction)

        np.testing.assert_allclose(corrected.normal, target.normal, atol=1e-10)
        self.assertAlmostEqual(corrected.offset, target.offset, places=10)

    def test_plane_alignment_about_anchor_does_not_shift_anchor(self):
        anchor = np.array([0.025, 0.060, 0.175])
        source_normal = np.array([0.03, 0.998, -0.05])
        source_normal /= np.linalg.norm(source_normal)
        source = PlaneModel(
            normal=source_normal,
            offset=-float(source_normal @ anchor),
            inlier_count=2000,
            candidate_count=3000,
            rmse_m=0.0005,
            reliable=True,
            reason="accepted",
        )
        target = PlaneModel(
            normal=np.array([0.0, 1.0, 0.0]),
            offset=-anchor[1],
            inlier_count=2000,
            candidate_count=3000,
            rmse_m=0.0005,
            reliable=True,
            reason="accepted",
        )

        correction = plane_alignment_transform(source, target, anchor=anchor)
        transformed_anchor = correction[:3, :3] @ anchor + correction[:3, 3]

        np.testing.assert_allclose(transformed_anchor, anchor, atol=1e-9)

    def test_plane_corrected_pose_aligns_synthetic_capture_to_reference(self):
        reference = PlaneModel(
            normal=np.array([0.0, 1.0, 0.0]),
            offset=-0.060,
            inlier_count=5000,
            candidate_count=6000,
            rmse_m=0.0005,
            reliable=True,
            reason="accepted",
        )
        tilted = PlaneModel(
            normal=np.array([0.02, 0.9997, -0.014]),
            offset=-0.062,
            inlier_count=5000,
            candidate_count=6000,
            rmse_m=0.0005,
            reliable=True,
            reason="accepted",
        ).normalized()

        _, corrected, _ = build_plane_corrected_poses(
            [Path("angle_0005p00.ply"), Path("angle_0010p00.ply")],
            [reference, tilted],
            np.array([0.0, 1.0, 0.0]),
        )
        transformed = transform_plane(tilted, corrected[1])

        np.testing.assert_allclose(
            transformed.normal,
            reference.normal,
            atol=1e-10,
        )
        self.assertAlmostEqual(transformed.offset, reference.offset, places=10)

    def test_enforce_platform_alignment_removes_optimized_pose_tilt(self):
        reference = PlaneModel(
            normal=np.array([0.0, 1.0, 0.0]),
            offset=-0.060,
            inlier_count=5000,
            candidate_count=6000,
            rmse_m=0.0005,
            reliable=True,
            reason="accepted",
        )
        pose = rotation_about_axis(
            2.0,
            np.array([0.025, 0.060, 0.175]),
            np.array([1.0, 0.0, 0.0]),
        )

        corrected = enforce_platform_alignment(
            [reference],
            [pose],
            reference,
        )[0]
        transformed = transform_plane(reference, corrected)

        np.testing.assert_allclose(transformed.normal, reference.normal, atol=1e-9)
        self.assertAlmostEqual(transformed.offset, reference.offset, places=9)

    def test_platform_plane_fit_accepts_dense_low_residual_plane(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(-0.06, 0.11, 1600)
        z = rng.uniform(0.095, 0.255, 1600)
        keep = np.hypot(x - 0.025, z - 0.175) >= 0.055
        points = np.c_[x[keep], np.full(np.count_nonzero(keep), 0.060), z[keep]]

        class FakeCloud:
            def __init__(self, cloud_points):
                self.points = cloud_points

            def select_by_index(self, indices):
                return FakeCloud(self.points[indices])

            def segment_plane(
                self,
                *,
                distance_threshold,
                ransac_n,
                num_iterations,
            ):
                del distance_threshold, ransac_n, num_iterations
                return [0.0, 1.0, 0.0, -0.060], list(range(len(self.points)))

        plane = fit_platform_plane(None, FakeCloud(points))

        self.assertTrue(plane.reliable)
        self.assertGreaterEqual(plane.inlier_count, 1000)
        np.testing.assert_allclose(
            plane.normal,
            [0.0, 1.0, 0.0],
            atol=1e-12,
        )
        self.assertAlmostEqual(plane.offset, -0.060)

    def test_plane_refinement_uses_all_ransac_inliers(self):
        rng = np.random.default_rng(7)
        x = rng.uniform(-0.08, 0.08, 1000)
        z = rng.uniform(0.10, 0.25, 1000)
        y = 0.060 + 0.01 * x - 0.015 * z
        points = np.c_[x, y + rng.normal(0.0, 0.0002, len(x)), z]
        points = np.vstack([points, [[0.0, 0.09, 0.17]]])

        normal, offset, indices, rmse = refine_plane_least_squares(
            points,
            np.arange(1000),
            distance_threshold_m=0.0015,
        )

        expected = np.array([-0.01, 1.0, 0.015])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(normal, expected, atol=0.001)
        self.assertAlmostEqual(offset, -0.060, places=3)
        self.assertEqual(len(indices), 1000)
        self.assertLess(rmse, 0.0003)

    def test_orbit_axis_uses_robust_plane_average(self):
        planes = [
            PlaneModel(
                normal=np.array([0.03, 0.9992, -0.026]),
                offset=-0.06,
                inlier_count=2000,
                candidate_count=3000,
                rmse_m=0.0004,
                reliable=True,
                reason="accepted",
            ),
            PlaneModel(
                normal=np.array([0.032, 0.9991, -0.025]),
                offset=-0.06,
                inlier_count=2100,
                candidate_count=3000,
                rmse_m=0.0005,
                reliable=True,
                reason="accepted",
            ),
            PlaneModel(
                normal=np.array([1.0, 0.0, 0.0]),
                offset=0.0,
                inlier_count=0,
                candidate_count=3000,
                rmse_m=1.0,
                reliable=False,
                reason="rejected",
            ),
        ]

        axis, used_fallback = estimate_orbit_axis(
            planes,
            fallback=np.array([0.0, 1.0, 0.0]),
            auto_calibrate=True,
        )

        self.assertFalse(used_fallback)
        self.assertGreater(axis[1], 0.999)
        self.assertGreater(axis[0], 0.02)
        self.assertLess(axis[2], -0.02)

    def test_orbit_axis_falls_back_without_reliable_planes(self):
        fallback = np.array([0.03, 0.999, -0.026])
        plane = PlaneModel(
            normal=np.array([0.0, 1.0, 0.0]),
            offset=0.0,
            inlier_count=0,
            candidate_count=0,
            rmse_m=float("inf"),
            reliable=False,
            reason="no candidates",
        )

        axis, used_fallback = estimate_orbit_axis(
            [plane],
            fallback=fallback,
            auto_calibrate=True,
        )

        self.assertTrue(used_fallback)
        np.testing.assert_allclose(axis, fallback / np.linalg.norm(fallback))

    def test_registration_guard_rejects_large_correction(self):
        prior = np.eye(4)
        candidate = np.eye(4)
        candidate[0, 3] = 0.02

        accepted, reason, correction_m, _ = registration_result_is_acceptable(
            fitness=0.9,
            rmse_m=0.001,
            prior=prior,
            candidate=candidate,
            min_fitness=0.3,
            max_rmse_m=0.004,
            max_correction_m=0.01,
            max_correction_deg=4.0,
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
        planes = [
            PlaneModel(
                normal=np.array([0.0, 1.0, 0.0]),
                offset=-0.060 + index * 0.00005,
                inlier_count=5000,
                candidate_count=6000,
                rmse_m=0.0005,
                reliable=True,
                reason="accepted",
            )
            for index in range(5)
        ]
        poses = [np.eye(4) for _ in planes]
        edges = [
            RegistrationEdge(
                source_id=index,
                target_id=index + 1,
                kind="sequential",
                transform=np.eye(4),
                information=np.eye(6),
                accepted=True,
                reason="accepted",
                fitness=0.9,
                rmse_m=0.001,
                correction_m=0.001,
                correction_deg=0.2,
            )
            for index in range(4)
        ]
        edges.append(
            RegistrationEdge(
                source_id=0,
                target_id=4,
                kind="loop",
                transform=np.eye(4),
                information=np.eye(6),
                accepted=True,
                reason="accepted",
                fitness=0.8,
                rmse_m=0.001,
                correction_m=0.002,
                correction_deg=0.5,
            )
        )

        quality = evaluate_registration_quality(
            planes=planes,
            optimized_poses=poses,
            edges=edges,
            pivot=np.array([0.025, 0.0, 0.175]),
            minimum_plane_fit_fraction=0.8,
            minimum_sequential_usable_fraction=0.9,
            maximum_plane_normal_p90_deg=0.5,
            maximum_plane_height_span_m=0.002,
            maximum_loop_correction_m=0.003,
            maximum_loop_correction_deg=1.0,
        )

        self.assertTrue(quality["passed"])
        self.assertEqual(quality["failure_reasons"], [])

    def test_quality_gate_rejects_missing_loop_closure(self):
        plane = PlaneModel(
            normal=np.array([0.0, 1.0, 0.0]),
            offset=-0.060,
            inlier_count=5000,
            candidate_count=6000,
            rmse_m=0.0005,
            reliable=True,
            reason="accepted",
        )

        quality = evaluate_registration_quality(
            planes=[plane, plane],
            optimized_poses=[np.eye(4), np.eye(4)],
            edges=[],
            pivot=np.array([0.025, 0.0, 0.175]),
            minimum_plane_fit_fraction=0.8,
            minimum_sequential_usable_fraction=0.9,
            maximum_plane_normal_p90_deg=0.5,
            maximum_plane_height_span_m=0.002,
            maximum_loop_correction_m=0.003,
            maximum_loop_correction_deg=1.0,
        )

        self.assertFalse(quality["passed"])
        self.assertTrue(
            any("loop closure" in reason for reason in quality["failure_reasons"])
        )

    def test_quality_gate_can_run_without_platform_for_hand_scans(self):
        no_plane = PlaneModel(
            normal=np.array([0.0, 1.0, 0.0]),
            offset=0.0,
            inlier_count=0,
            candidate_count=0,
            rmse_m=float("inf"),
            reliable=False,
            reason="platform alignment disabled",
        )
        sequential = RegistrationEdge(
            source_id=0,
            target_id=1,
            kind="sequential",
            transform=np.eye(4),
            information=np.eye(6),
            accepted=False,
            reason="used motor prior",
            fitness=0.8,
            rmse_m=0.002,
            correction_m=0.02,
            correction_deg=5.0,
            usable=True,
        )
        loop = RegistrationEdge(
            source_id=0,
            target_id=1,
            kind="loop",
            transform=np.eye(4),
            information=np.eye(6),
            accepted=True,
            reason="accepted",
            fitness=0.8,
            rmse_m=0.002,
            correction_m=0.001,
            correction_deg=0.2,
        )

        quality = evaluate_registration_quality(
            planes=[no_plane, no_plane],
            optimized_poses=[np.eye(4), np.eye(4)],
            plane_corrected_poses=[np.eye(4), np.eye(4)],
            edges=[sequential, loop],
            pivot=np.array([0.025, 0.0, 0.175]),
            require_platform_alignment=False,
            minimum_plane_fit_fraction=0.8,
            minimum_sequential_usable_fraction=0.9,
            maximum_plane_normal_p90_deg=0.5,
            maximum_plane_height_span_m=0.002,
            maximum_loop_correction_m=0.003,
            maximum_loop_correction_deg=1.0,
        )

        self.assertTrue(quality["passed"])
        self.assertFalse(quality["platform_alignment_required"])
        self.assertIsNone(quality["plane_normal_residual_p90_deg"])
        self.assertIsNone(quality["plane_height_span_m"])

    def test_quality_gate_counts_guarded_prior_fallback_as_usable(self):
        plane = PlaneModel(
            normal=np.array([0.0, 1.0, 0.0]),
            offset=-0.060,
            inlier_count=5000,
            candidate_count=6000,
            rmse_m=0.0005,
            reliable=True,
            reason="accepted",
        )
        sequential = RegistrationEdge(
            source_id=0,
            target_id=1,
            kind="sequential",
            transform=np.eye(4),
            information=np.eye(6),
            accepted=False,
            reason="ICP correction rejected; used pose prior",
            fitness=0.9,
            rmse_m=0.001,
            correction_m=0.012,
            correction_deg=3.0,
            usable=True,
            prior_fitness=0.85,
            prior_rmse_m=0.0015,
        )
        loop = RegistrationEdge(
            source_id=0,
            target_id=1,
            kind="loop",
            transform=np.eye(4),
            information=np.eye(6),
            accepted=True,
            reason="accepted",
            fitness=0.9,
            rmse_m=0.001,
            correction_m=0.001,
            correction_deg=0.2,
        )

        quality = evaluate_registration_quality(
            planes=[plane, plane],
            optimized_poses=[np.eye(4), np.eye(4)],
            edges=[sequential, loop],
            pivot=np.array([0.025, 0.0, 0.175]),
            minimum_plane_fit_fraction=0.8,
            minimum_sequential_usable_fraction=0.9,
            maximum_plane_normal_p90_deg=0.5,
            maximum_plane_height_span_m=0.002,
            maximum_loop_correction_m=0.003,
            maximum_loop_correction_deg=1.0,
        )

        self.assertTrue(quality["passed"])
        self.assertEqual(quality["sequential_usable_fraction"], 1.0)
        self.assertEqual(quality["sequential_icp_acceptance_fraction"], 0.0)

    def test_loop_quality_measures_optimized_orbit_against_pose_prior(self):
        plane = PlaneModel(
            normal=np.array([0.0, 1.0, 0.0]),
            offset=-0.060,
            inlier_count=5000,
            candidate_count=6000,
            rmse_m=0.0005,
            reliable=True,
            reason="accepted",
        )
        loop = RegistrationEdge(
            source_id=0,
            target_id=1,
            kind="loop",
            transform=np.eye(4),
            information=np.eye(6),
            accepted=True,
            reason="accepted",
            fitness=0.9,
            rmse_m=0.001,
            correction_m=0.009,
            correction_deg=3.0,
        )

        quality = evaluate_registration_quality(
            planes=[plane, plane],
            optimized_poses=[np.eye(4), np.eye(4)],
            plane_corrected_poses=[np.eye(4), np.eye(4)],
            edges=[loop],
            pivot=np.array([0.025, 0.0, 0.175]),
            minimum_plane_fit_fraction=0.8,
            minimum_sequential_usable_fraction=0.0,
            maximum_plane_normal_p90_deg=0.5,
            maximum_plane_height_span_m=0.002,
            maximum_loop_correction_m=0.003,
            maximum_loop_correction_deg=1.0,
        )

        self.assertTrue(quality["loop_closure_passed"])
        self.assertEqual(quality["loop_closure_translation_m"], 0.0)
        self.assertEqual(quality["loop_closure_rotation_deg"], 0.0)

    def test_diagnostics_convert_non_finite_values_to_json_null(self):
        plane = PlaneModel(
            normal=np.array([0.0, 1.0, 0.0]),
            offset=-0.060,
            inlier_count=0,
            candidate_count=0,
            rmse_m=float("inf"),
            reliable=False,
            reason="rejected",
        )
        edge = RegistrationEdge(
            source_id=0,
            target_id=1,
            kind="sequential",
            transform=np.eye(4),
            information=np.eye(6),
            accepted=False,
            reason="rejected",
            fitness=0.0,
            rmse_m=float("inf"),
            correction_m=0.0,
            correction_deg=0.0,
        )

        encoded = json.dumps(
            {
                "plane": plane_as_json(plane),
                "edge": edge_as_json(edge),
            },
            allow_nan=False,
        )

        decoded = json.loads(encoded)
        self.assertIsNone(decoded["plane"]["rmse_m"])
        self.assertIsNone(decoded["edge"]["rmse_m"])
        self.assertIsNone(decoded["edge"]["prior_rmse_m"])


class TransformCloudsCompatibilityTests(unittest.TestCase):
    def test_cloudcompare_command_prefers_native_executable(self):
        paths = {
            "CloudCompare": "/usr/bin/CloudCompare",
            "flatpak": "/usr/bin/flatpak",
        }

        command = cloudcompare_command(
            which=lambda name: paths.get(name),
            flatpak_installed=lambda _flatpak, _app_id: True,
        )

        self.assertEqual(command, ["/usr/bin/CloudCompare"])

    def test_cloudcompare_command_uses_installed_flatpak(self):
        paths = {"flatpak": "/usr/bin/flatpak"}

        command = cloudcompare_command(
            which=lambda name: paths.get(name),
            flatpak_installed=lambda _flatpak, app_id: (
                app_id == "org.cloudcompare.CloudCompare"
            ),
        )

        self.assertEqual(
            command,
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

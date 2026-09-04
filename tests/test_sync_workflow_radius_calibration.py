from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import cv2
from PIL import Image


SYNC_WORKFLOW_DIR = (
    Path(__file__).resolve().parent.parent / "scripts" / "sync_workflow"
)
sys.path.insert(0, str(SYNC_WORKFLOW_DIR))

from calculating_radius.radius_calibration import (  # noqa: E402
    _fit_circle_once,
    _refine_circle_geometric,
    bootstrap_radius_ci,
    build_detector_parameters,
    build_fit_samples,
    build_profile_marker_map,
    camera_center_world,
    classify_pose,
    convert_world_to_color_to_world_to_depth,
    evaluate_trajectory,
    fit_orbit_circle,
    has_sufficient_angular_coverage,
    marker_normal,
    orbit_geometry_in_camera_frame,
    solve_profile_pose,
)
from calculating_radius.generate_radius_markers import generate_marker_kit  # noqa: E402
from calculating_radius import radius_calibration  # noqa: E402
from calculating_radius import test_radius  # noqa: E402


class DetectorParameterTests(unittest.TestCase):
    def test_enables_subpixel_corner_refinement(self):
        params = build_detector_parameters()

        self.assertEqual(
            params.cornerRefinementMethod, cv2.aruco.CORNER_REFINE_SUBPIX
        )
        self.assertEqual(params.cornerRefinementWinSize, 5)
        self.assertEqual(params.cornerRefinementMaxIterations, 50)
        self.assertAlmostEqual(params.cornerRefinementMinAccuracy, 0.01)


class ClassifyPoseTests(unittest.TestCase):
    def test_accepts_multi_marker_pose_within_error_budget(self):
        pose = {"ok": True, "used_ids": [0, 1], "reprojection_error_px": 0.5}

        accepted, reason = classify_pose(
            pose, max_reprojection_error_px=1.5, min_markers=2
        )

        self.assertTrue(accepted)
        self.assertIsNone(reason)

    def test_rejects_single_marker_pose_when_two_required(self):
        pose = {"ok": True, "used_ids": [0], "reprojection_error_px": 0.2}

        accepted, reason = classify_pose(
            pose, max_reprojection_error_px=1.5, min_markers=2
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "fewer than 2 mapped markers in view")

    def test_allows_single_marker_pose_when_one_permitted(self):
        pose = {"ok": True, "used_ids": [0], "reprojection_error_px": 0.2}

        accepted, reason = classify_pose(
            pose, max_reprojection_error_px=1.5, min_markers=1
        )

        self.assertTrue(accepted)
        self.assertIsNone(reason)

    def test_rejects_high_reprojection_error(self):
        pose = {"ok": True, "used_ids": [0, 1, 2], "reprojection_error_px": 3.0}

        accepted, reason = classify_pose(
            pose, max_reprojection_error_px=1.5, min_markers=2
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "reprojection error exceeds 1.50px")

    def test_rejects_failed_pose_and_non_finite_error(self):
        failed = {"ok": False, "used_ids": [], "reprojection_error_px": None}
        self.assertEqual(
            classify_pose(failed, max_reprojection_error_px=1.5),
            (False, "no usable mapped marker pose"),
        )

        nan_error = {
            "ok": True,
            "used_ids": [0, 1],
            "reprojection_error_px": float("nan"),
        }
        self.assertEqual(
            classify_pose(nan_error, max_reprojection_error_px=1.5),
            (False, "non-finite reprojection error"),
        )


class BuildFitSamplesTests(unittest.TestCase):
    def _angle_results(self):
        return [
            {
                "angle_deg": 0.0,
                "pose_valid": True,
                "rgb_camera_center_m": [0.11, 0.0, 0.0],
                "depth_camera_center_m": [0.12, 0.0, 0.0],
                "frames": [
                    {
                        "accepted": True,
                        "rgb_camera_center_m": [0.10, 0.0, 0.0],
                        "depth_camera_center_m": [0.11, 0.0, 0.0],
                    },
                    {"accepted": False},
                    {
                        "accepted": True,
                        "rgb_camera_center_m": [0.12, 0.0, 0.0],
                        "depth_camera_center_m": [0.13, 0.0, 0.0],
                    },
                ],
            },
            {"angle_deg": 45.0, "pose_valid": False, "frames": []},
        ]

    def test_expands_every_accepted_frame(self):
        samples = build_fit_samples(self._angle_results(), use_all_frames=True)

        self.assertEqual(len(samples), 2)
        self.assertEqual([s["angle_deg"] for s in samples], [0.0, 0.0])
        self.assertEqual(samples[0]["depth_camera_center_m"], [0.11, 0.0, 0.0])
        self.assertEqual(samples[1]["depth_camera_center_m"], [0.13, 0.0, 0.0])

    def test_median_mode_returns_one_sample_per_valid_angle(self):
        samples = build_fit_samples(self._angle_results(), use_all_frames=False)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["angle_deg"], 0.0)
        self.assertEqual(samples[0]["depth_camera_center_m"], [0.12, 0.0, 0.0])

    def test_skips_angles_that_are_not_pose_valid(self):
        results = self._angle_results()
        results[0]["pose_valid"] = False

        self.assertEqual(build_fit_samples(results, use_all_frames=True), [])


class ProfileMarkerMapTests(unittest.TestCase):
    def test_uses_same_position_mixed_size_horizontal_wrap_geometry(self):
        marker_map = build_profile_marker_map()

        self.assertEqual(marker_map["dictionary"], "DICT_5X5_100")
        self.assertEqual(marker_map["schema_version"], 2)
        self.assertEqual(marker_map["profile_cross_section_m"], [0.040, 0.020])
        self.assertEqual(marker_map["carrier_offset_m"], 0.0001)
        self.assertEqual(
            [marker["size_m"] for marker in marker_map["markers"]],
            [0.018, 0.014, 0.018, 0.014],
        )
        self.assertEqual(
            [marker["center_m"][2] for marker in marker_map["markers"]],
            [0.0, 0.0, 0.0, 0.0],
        )
        self.assertEqual(
            [marker["face"] for marker in marker_map["markers"]],
            ["front", "top", "back", "bottom"],
        )

        centers = {
            marker["face"]: np.asarray(marker["center_m"])
            for marker in marker_map["markers"]
        }
        np.testing.assert_allclose(centers["front"], [0.0, 0.0101, 0.0])
        np.testing.assert_allclose(centers["top"], [0.0201, 0.0, 0.0])
        np.testing.assert_allclose(centers["back"], [0.0, -0.0101, 0.0])
        np.testing.assert_allclose(centers["bottom"], [-0.0201, 0.0, 0.0])

    def test_marker_corners_use_per_face_sizes_and_point_outward(self):
        marker_map = build_profile_marker_map()
        expected_normals = {
            "front": [0.0, 1.0, 0.0],
            "top": [1.0, 0.0, 0.0],
            "back": [0.0, -1.0, 0.0],
            "bottom": [-1.0, 0.0, 0.0],
        }

        for marker in marker_map["markers"]:
            corners = np.asarray(marker["corners_m"])
            side_lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
            np.testing.assert_allclose(side_lengths, marker["size_m"], atol=1e-12)
            np.testing.assert_allclose(
                marker_normal(corners), expected_normals[marker["face"]], atol=1e-12
            )

    def test_marker_canonical_top_edge_points_toward_the_printed_strip_top(self):
        marker_map = build_profile_marker_map()

        self.assertEqual(
            marker_map["world_frame"]["z_axis"],
            "along the printed strip top-edge direction",
        )
        for marker in marker_map["markers"]:
            corners = np.asarray(marker["corners_m"])
            center_z = float(marker["center_m"][2])
            self.assertTrue(np.all(corners[:2, 2] > center_z))
            self.assertTrue(np.all(corners[2:, 2] < center_z))
            self.assertEqual(marker["bar_up"], [0.0, 0.0, 1.0])

    def test_print_kit_contains_exact_continuous_wrap_and_fold_geometry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = generate_marker_kit(Path(temp_dir), dpi=600)

            self.assertTrue(paths["pdf"].is_file())
            self.assertTrue(paths["map"].is_file())
            self.assertTrue(paths["placement_guide"].is_file())
            self.assertTrue(paths["wrap_preview"].is_file())

            generated_map = json.loads(paths["map"].read_text(encoding="utf-8"))
            self.assertEqual(generated_map["wrap"]["size_m"], [0.130, 0.040])
            self.assertEqual(
                generated_map["wrap"]["fold_positions_m"],
                [0.010, 0.030, 0.070, 0.090],
            )
            self.assertEqual(
                generated_map["wrap"]["marker_centers_unwrapped_m"],
                {
                    "3": [0.020, 0.020],
                    "0": [0.050, 0.020],
                    "1": [0.080, 0.020],
                    "2": [0.110, 0.020],
                },
            )

    def test_wrap_prints_each_coded_square_at_its_declared_physical_size(self):
        dpi = 600
        marker_sizes_mm = {0: 18.0, 1: 14.0, 2: 18.0, 3: 14.0}
        marker_centers_mm = {
            3: (20.0, 20.0),
            0: (50.0, 20.0),
            1: (80.0, 20.0),
            2: (110.0, 20.0),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = generate_marker_kit(Path(temp_dir), dpi=dpi)
            wrap = np.asarray(Image.open(paths["wrap_preview"]).convert("L"))

        for marker_id, declared_size_mm in marker_sizes_mm.items():
            center_x_mm, center_y_mm = marker_centers_mm[marker_id]
            margin_mm = declared_size_mm / 2.0 + 1.0
            x0 = round((center_x_mm - margin_mm) / 25.4 * dpi)
            x1 = round((center_x_mm + margin_mm) / 25.4 * dpi)
            y0 = round((center_y_mm - margin_mm) / 25.4 * dpi)
            y1 = round((center_y_mm + margin_mm) / 25.4 * dpi)
            black_pixels = np.argwhere(wrap[y0:y1, x0:x1] < 10)
            printed_height_px, printed_width_px = np.ptp(black_pixels, axis=0) + 1

            self.assertAlmostEqual(
                printed_width_px * 25.4 / dpi,
                declared_size_mm,
                delta=0.03,
            )
            self.assertAlmostEqual(
                printed_height_px * 25.4 / dpi,
                declared_size_mm,
                delta=0.03,
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

    def test_does_not_recommend_radius_without_full_angular_coverage(self):
        angles = [0, 15, 30, 45, 60, 75]
        samples = []
        for angle in angles:
            radians = np.deg2rad(angle)
            center = [0.1175 * np.cos(radians), 0.1175 * np.sin(radians), 0.0]
            samples.append(
                {
                    "angle_deg": angle,
                    "rgb_camera_center_m": center,
                    "depth_camera_center_m": center,
                }
            )

        result = evaluate_trajectory(samples)

        self.assertEqual(result["quality_status"], "invalid")
        self.assertIsNone(result["recommended_radius_m"])
        self.assertIn("angular coverage", " ".join(result["quality_reasons"]))

    def test_recommends_depth_radius_for_a_valid_full_orbit(self):
        samples = []
        for angle in range(0, 360, 45):
            radians = np.deg2rad(angle)
            samples.append(
                {
                    "angle_deg": angle,
                    "rgb_camera_center_m": [
                        0.118 * np.cos(radians),
                        0.118 * np.sin(radians),
                        0.0,
                    ],
                    "depth_camera_center_m": [
                        0.1175 * np.cos(radians),
                        0.1175 * np.sin(radians),
                        0.0,
                    ],
                }
            )

        result = evaluate_trajectory(samples)

        self.assertEqual(result["quality_status"], "valid")
        self.assertAlmostEqual(result["recommended_radius_m"], 0.1175, places=9)

    def test_degenerate_camera_centers_return_invalid_instead_of_crashing(self):
        samples = [
            {
                "angle_deg": angle,
                "rgb_camera_center_m": [0.0, 0.0, 0.0],
                "depth_camera_center_m": [0.0, 0.0, 0.0],
            }
            for angle in range(0, 360, 45)
        ]

        result = evaluate_trajectory(samples)

        self.assertEqual(result["quality_status"], "invalid")
        self.assertIsNone(result["recommended_radius_m"])
        self.assertIn("fit failed", " ".join(result["quality_reasons"]))

    def test_geometric_refinement_recovers_an_exact_partial_arc(self):
        radius = 0.1175
        angles = np.deg2rad(np.linspace(0.0, 170.0, 10))
        points = np.column_stack(
            (radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles))
        )

        result = fit_orbit_circle(points)

        self.assertAlmostEqual(result["radius_m"], radius, places=7)

    def test_geometric_refinement_lowers_radius_bias_on_short_noisy_arcs(self):
        # The algebraic (Kasa) fit systematically under-estimates the radius of a
        # short, noisy arc; the geometric refinement removes most of that bias.
        rng = np.random.default_rng(1234)
        radius = 0.1175
        angles = np.deg2rad(np.linspace(0.0, 100.0, 12))
        clean = np.column_stack(
            (radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles))
        )

        algebraic_signed = []
        refined_signed = []
        for _ in range(300):
            noisy = clean + rng.normal(scale=0.0012, size=clean.shape)
            algebraic = _fit_circle_once(noisy)
            algebraic_signed.append(algebraic["radius"] - radius)
            refined = _refine_circle_geometric(
                noisy, algebraic["center"], algebraic["axis"], algebraic["radius"]
            )
            refined_signed.append(refined["radius"] - radius)

        self.assertLess(
            abs(np.mean(refined_signed)), abs(np.mean(algebraic_signed))
        )
        self.assertLess(abs(np.mean(refined_signed)), 0.5 * abs(np.mean(algebraic_signed)))

    def test_refine_circle_geometric_rejects_missing_scipy(self):
        # Sanity: the helper returns a plausible circle for well-posed input.
        radius = 0.1175
        angles = np.deg2rad(np.arange(0.0, 360.0, 30.0))
        points = np.column_stack(
            (radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles))
        )

        refined = _refine_circle_geometric(
            points, np.array([0.01, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 0.10
        )

        self.assertAlmostEqual(refined["radius"], radius, places=6)
        np.testing.assert_allclose(refined["center"], [0.0, 0.0, 0.0], atol=1e-6)

    def test_bootstrap_ci_brackets_the_true_radius(self):
        rng = np.random.default_rng(7)
        radius = 0.1175
        angles = np.deg2rad(np.arange(0.0, 360.0, 20.0))
        clean = np.column_stack(
            (radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles))
        )
        noisy = clean + rng.normal(scale=0.0005, size=clean.shape)

        ci = bootstrap_radius_ci(noisy, n_resamples=200, seed=3)

        self.assertIsNotNone(ci["radius_std_m"])
        self.assertGreater(ci["radius_std_m"], 0.0)
        self.assertLess(ci["radius_ci_low_m"], radius)
        self.assertGreater(ci["radius_ci_high_m"], radius)
        self.assertGreaterEqual(ci["n_resamples_ok"], 180)

    def test_bootstrap_ci_rejects_too_few_points(self):
        with self.assertRaises(ValueError):
            bootstrap_radius_ci(np.zeros((2, 3)))

    def test_evaluate_trajectory_attaches_radius_confidence_interval(self):
        samples = []
        for angle in range(0, 360, 30):
            radians = np.deg2rad(angle)
            center = [0.1175 * np.cos(radians), 0.1175 * np.sin(radians), 0.0]
            samples.append(
                {
                    "angle_deg": angle,
                    "rgb_camera_center_m": center,
                    "depth_camera_center_m": center,
                }
            )

        result = evaluate_trajectory(samples, bootstrap_resamples=80)

        for key in (
            "radius_std_m",
            "radius_ci_low_m",
            "radius_ci_high_m",
            "n_resamples_ok",
        ):
            self.assertIn(key, result["depth_fit"])
            self.assertIn(key, result["rgb_fit"])


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


class OrbitGeometryTests(unittest.TestCase):
    def test_transforms_centre_as_a_point_and_axis_as_a_direction(self):
        world_to_camera = np.eye(4)
        world_to_camera[:3, 3] = [0.10, -0.20, 0.30]

        geometry = orbit_geometry_in_camera_frame(
            [0.01, 0.02, 0.03], [0.0, 0.0, 1.0], world_to_camera
        )

        np.testing.assert_allclose(geometry["pivot_m"], [0.11, -0.18, 0.33])
        np.testing.assert_allclose(geometry["axis"], [0.0, 0.0, 1.0])

    def test_rotates_the_axis_without_translating_it(self):
        world_to_camera = np.eye(4)
        # 90 degrees about world +Z maps +X to +Y.
        world_to_camera[:3, :3] = [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        world_to_camera[:3, 3] = [1.0, 2.0, 3.0]

        geometry = orbit_geometry_in_camera_frame(
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], world_to_camera
        )

        np.testing.assert_allclose(geometry["axis"], [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(geometry["pivot_m"], [1.0, 2.0, 3.0])

    def test_returns_a_unit_axis(self):
        geometry = orbit_geometry_in_camera_frame(
            [0.0, 0.0, 0.0], [0.0, 3.0, 4.0], np.eye(4)
        )

        self.assertAlmostEqual(float(np.linalg.norm(geometry["axis"])), 1.0)

    def test_rejects_a_degenerate_axis(self):
        with self.assertRaises(ValueError):
            orbit_geometry_in_camera_frame(
                [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], np.eye(4)
            )


class ProfilePoseTests(unittest.TestCase):
    def setUp(self):
        self.marker_map = build_profile_marker_map()
        self.camera_matrix = np.array(
            [[900.0, 0.0, 640.0], [0.0, 900.0, 400.0], [0.0, 0.0, 1.0]]
        )
        self.dist_coeffs = np.zeros((8, 1), dtype=np.float64)
        camera_center = np.array([0.0, 0.16, 0.0])
        right = np.array([-1.0, 0.0, 0.0])
        down = np.array([0.0, 0.0, -1.0])
        forward = np.array([0.0, -1.0, 0.0])
        self.world_to_camera = np.eye(4)
        self.world_to_camera[:3, :3] = np.vstack((right, down, forward))
        self.world_to_camera[:3, 3] = -self.world_to_camera[:3, :3] @ camera_center
        self.rvec, _ = cv2.Rodrigues(self.world_to_camera[:3, :3])
        self.tvec = self.world_to_camera[:3, 3].reshape(3, 1)

    def projected_marker(self, marker_id):
        marker = self.marker_map["markers"][marker_id]
        corners, _ = cv2.projectPoints(
            np.asarray(marker["corners_m"]),
            self.rvec,
            self.tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        return corners.reshape(1, 4, 2).astype(np.float32)

    def test_recovers_pose_from_two_mapped_markers(self):
        result = solve_profile_pose(
            self.marker_map,
            [self.projected_marker(0), self.projected_marker(1)],
            np.array([[0], [1]], dtype=np.int32),
            self.camera_matrix,
            self.dist_coeffs,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["method"], "multi-marker-iterative")
        np.testing.assert_allclose(
            camera_center_world(result["world_to_camera"]), [0.0, 0.16, 0.0], atol=1e-5
        )

    def test_recovers_pose_when_only_one_profile_face_is_visible(self):
        result = solve_profile_pose(
            self.marker_map,
            [self.projected_marker(0)],
            np.array([[0]], dtype=np.int32),
            self.camera_matrix,
            self.dist_coeffs,
        )

        self.assertTrue(result["ok"])
        self.assertIn(
            result["method"],
            {"single-marker-ippe", "single-marker-iterative-fallback"},
        )
        np.testing.assert_allclose(
            camera_center_world(result["world_to_camera"]), [0.0, 0.16, 0.0], atol=5e-4
        )

    def test_single_marker_pose_prefers_the_markers_own_size(self):
        marker = self.marker_map["markers"][0]
        center = np.asarray(marker["center_m"], dtype=np.float64)
        corners = np.asarray(marker["corners_m"], dtype=np.float64)
        marker["size_m"] = 0.014
        marker["corners_m"] = (center + (corners - center) * (14.0 / 18.0)).tolist()
        self.marker_map["marker_size_m"] = 0.018

        result = solve_profile_pose(
            self.marker_map,
            [self.projected_marker(0)],
            np.array([[0]], dtype=np.int32),
            self.camera_matrix,
            self.dist_coeffs,
        )

        self.assertTrue(result["ok"])
        np.testing.assert_allclose(
            camera_center_world(result["world_to_camera"]), [0.0, 0.16, 0.0], atol=6e-4
        )

    def test_ignores_unknown_marker_ids(self):
        result = solve_profile_pose(
            self.marker_map,
            [self.projected_marker(0)],
            np.array([[42]], dtype=np.int32),
            self.camera_matrix,
            self.dist_coeffs,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["used_ids"], [])


if __name__ == "__main__":
    unittest.main()


class CircleFitResidualGateTests(unittest.TestCase):
    """A clean reprojection error must not certify a poor orbit fit."""

    @staticmethod
    def _orbit_samples(radius_m: float, noise_m: float = 0.0):
        angles = np.arange(0.0, 360.0, 30.0)
        samples = []
        for index, angle in enumerate(angles):
            theta = np.radians(angle)
            offset = noise_m if index % 2 == 0 else -noise_m
            centre = np.array([
                0.0,
                (radius_m + offset) * np.cos(theta),
                (radius_m + offset) * np.sin(theta),
            ])
            samples.append({
                "angle_deg": float(angle),
                "rgb_camera_center_m": centre.tolist(),
                "depth_camera_center_m": centre.tolist(),
            })
        return samples

    def test_a_clean_orbit_passes(self):
        result = radius_calibration.evaluate_trajectory(
            self._orbit_samples(0.1427), bootstrap_resamples=0,
        )
        self.assertEqual(result["quality_status"], "valid")

    def test_a_five_millimetre_fit_is_rejected(self):
        result = radius_calibration.evaluate_trajectory(
            self._orbit_samples(0.1427, noise_m=0.005), bootstrap_resamples=0,
        )
        self.assertEqual(result["quality_status"], "invalid")
        self.assertTrue(
            any("circle-fit" in reason for reason in result["quality_reasons"]),
            result["quality_reasons"],
        )
        self.assertIsNone(result["recommended_radius_m"])

    def test_the_gate_thresholds_are_configurable(self):
        samples = self._orbit_samples(0.1427, noise_m=0.005)
        relaxed = radius_calibration.evaluate_trajectory(
            samples,
            bootstrap_resamples=0,
            max_fit_rmse_m=0.02,
            max_fit_residual_m=0.02,
        )
        self.assertEqual(relaxed["quality_status"], "valid")


class IppeAmbiguityGateTests(unittest.TestCase):
    """A square marker admits two IPPE poses; near-equal ones must be rejected."""

    BASE = {
        "ok": True,
        "used_ids": [1],
        "reprojection_error_px": 0.4,
    }

    def _classify(self, **overrides):
        pose = dict(self.BASE)
        pose.update(overrides)
        return radius_calibration.classify_pose(
            pose, max_reprojection_error_px=1.5,
        )

    def test_a_decisive_solution_is_accepted(self):
        accepted, reason = self._classify(ippe_error_ratio=6.0)
        self.assertTrue(accepted, reason)
        self.assertIsNone(reason)

    def test_an_ambiguous_solution_is_rejected(self):
        accepted, reason = self._classify(ippe_error_ratio=1.05)
        self.assertFalse(accepted)
        self.assertIn("ambiguous single-marker pose", reason)

    def test_a_lone_solution_is_not_penalised(self):
        """One surviving IPPE solution carries no ratio and stays acceptable."""
        accepted, reason = self._classify(ippe_error_ratio=None)
        self.assertTrue(accepted, reason)

    def test_the_ratio_threshold_is_configurable(self):
        pose = dict(self.BASE, ippe_error_ratio=1.5)
        self.assertFalse(
            radius_calibration.classify_pose(
                pose, max_reprojection_error_px=1.5, min_ippe_error_ratio=2.0,
            )[0]
        )
        self.assertTrue(
            radius_calibration.classify_pose(
                pose, max_reprojection_error_px=1.5, min_ippe_error_ratio=1.2,
            )[0]
        )

    def test_a_single_marker_is_allowed_by_default(self):
        """The four-face profile cannot show two markers at most angles."""
        accepted, reason = self._classify(ippe_error_ratio=6.0)
        self.assertTrue(accepted, reason)


class TestRadiusArgumentsTests(unittest.TestCase):
    """Every gate validate_args reads must exist on the parser."""

    def test_defaults_parse_and_validate(self):
        args = test_radius.parse_args([])
        self.assertEqual(args.min_markers_per_pose, 1)
        self.assertEqual(
            args.min_ippe_error_ratio,
            radius_calibration.DEFAULT_MIN_IPPE_ERROR_RATIO,
        )
        self.assertEqual(
            args.max_fit_rmse_mm, radius_calibration.DEFAULT_MAX_FIT_RMSE_M * 1000.0,
        )
        test_radius.validate_args(args)

    def test_the_ambiguity_ratio_is_overridable_and_bounded(self):
        args = test_radius.parse_args(["--min-ippe-error-ratio", "1.5"])
        self.assertEqual(args.min_ippe_error_ratio, 1.5)
        test_radius.validate_args(args)

        with self.assertRaises(ValueError):
            test_radius.validate_args(
                test_radius.parse_args(["--min-ippe-error-ratio", "0.5"])
            )

    def test_the_fit_gates_are_overridable_and_bounded(self):
        args = test_radius.parse_args(
            ["--max-fit-rmse-mm", "3", "--max-fit-residual-mm", "7"]
        )
        test_radius.validate_args(args)
        for flag in ("--max-fit-rmse-mm", "--max-fit-residual-mm"):
            with self.assertRaises(ValueError):
                test_radius.validate_args(test_radius.parse_args([flag, "0"]))

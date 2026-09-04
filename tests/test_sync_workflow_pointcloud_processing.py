from __future__ import annotations

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

import pointcloud_processing


def write_cloud(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=float))
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False):
        raise RuntimeError(f"Could not create test cloud: {path}")


class TransformAndCleanTests(unittest.TestCase):
    def test_uses_a_different_crop_center_for_each_station(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            first = root / "frame_station_0.ply"
            second = root / "frame_station_1.ply"
            points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            write_cloud(first, points, colors)
            write_cloud(second, points, colors)

            paths, _ = pointcloud_processing.transform_and_clean_clouds(
                [first, second],
                [np.eye(4), np.eye(4)],
                root / "transformed",
                root / "matrices",
                crop_bounds=None,
                crop_bounds_by_cloud=[
                    (-0.1, -0.1, -0.1, 0.1, 0.1, 0.1),
                    (0.9, -0.1, -0.1, 1.1, 0.1, 0.1),
                ],
                skip_sor=True,
                sor_neighbors=10,
                sor_sigma=2.0,
                max_workers=1,
            )

            np.testing.assert_allclose(
                np.asarray(o3d.io.read_point_cloud(str(paths[0])).points),
                [[0.0, 0.0, 0.0]],
            )
            np.testing.assert_allclose(
                np.asarray(o3d.io.read_point_cloud(str(paths[1])).points),
                [[1.0, 0.0, 0.0]],
            )

    def test_transforms_crops_and_preserves_colors_and_matrix_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_path = root / "frame_5.0.ply"
            write_cloud(
                source_path,
                np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            )
            transform = np.eye(4)
            transform[0, 3] = 0.1

            paths, stats = pointcloud_processing.transform_and_clean_clouds(
                [source_path],
                [transform],
                root / "01_transformed",
                root / "matrices",
                crop_bounds=(-0.01, -0.1, -0.1, 0.2, 0.1, 0.1),
                skip_sor=True,
                sor_neighbors=10,
                sor_sigma=2.0,
                max_workers=1,
            )

            self.assertEqual(paths, [root / "01_transformed" / "frame_5.0_transformed.ply"])
            transformed = o3d.io.read_point_cloud(str(paths[0]))
            np.testing.assert_allclose(np.asarray(transformed.points), [[0.1, 0.0, 0.0]])
            np.testing.assert_allclose(np.asarray(transformed.colors), [[1.0, 0.0, 0.0]])
            np.testing.assert_allclose(
                np.loadtxt(root / "matrices" / "frame_5.0_optimized_matrix.txt"),
                transform,
            )
            self.assertEqual(stats[0]["input_points"], 2)
            self.assertEqual(stats[0]["output_points"], 1)

    def test_rejects_mismatched_path_and_pose_counts(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            pointcloud_processing.transform_and_clean_clouds(
                [Path("frame_0.0.ply")],
                [],
                Path("transformed"),
                Path("matrices"),
                crop_bounds=None,
                skip_sor=True,
                sor_neighbors=10,
                sor_sigma=2.0,
            )


class MergeAndFinalizeTests(unittest.TestCase):
    def test_minimum_distance_selection_removes_cross_voxel_neighbors(self):
        points = np.array([
            [0.0000, 0.0, 0.0],
            [0.0008, 0.0, 0.0],
            [0.0020, 0.0, 0.0],
        ])

        indices = pointcloud_processing.minimum_distance_sample_indices(
            points,
            radius_m=0.001,
        )

        selected = points[indices]
        distances = np.linalg.norm(selected[:, None, :] - selected[None, :, :], axis=2)
        distances[np.diag_indices_from(distances)] = np.inf
        self.assertEqual(len(selected), 2)
        self.assertGreaterEqual(distances.min(), 0.001)

    def test_quantized_duplicate_selection_keeps_matching_attributes(self):
        points = np.array([
            [0.00001, 0.0, 0.0],
            [0.00004, 0.0, 0.0],
            [0.00100, 0.0, 0.0],
        ])

        indices = pointcloud_processing.quantized_unique_indices(
            points,
            tolerance_m=0.0001,
        )

        np.testing.assert_array_equal(indices, [0, 2])

    def test_merges_cleans_estimates_normals_and_validates_with_trimesh(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            phi = np.linspace(0.2, np.pi - 0.2, 12)
            theta = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
            points = np.array([
                [
                    0.02 * np.sin(p) * np.cos(t),
                    0.02 * np.sin(p) * np.sin(t),
                    0.1 + 0.02 * np.cos(p),
                ]
                for p in phi
                for t in theta
            ])
            colors = np.tile([0.2, 0.4, 0.8], (len(points), 1))
            first = root / "frame_0.0_transformed.ply"
            second = root / "frame_5.0_transformed.ply"
            write_cloud(first, points, colors)
            write_cloud(second, points, colors)
            output_path = root / "merged_cloud.ply"

            stats = pointcloud_processing.merge_and_finalize_clouds(
                [first, second],
                output_path,
                pivot=np.array([0.0, 0.0, 0.1]),
                spatial_subsample_m=0.002,
                duplicate_distance_m=0.0001,
                sor_neighbors=8,
                sor_sigma=2.0,
                normal_radius_m=0.008,
                normal_max_neighbors=30,
                normal_mst_neighbors=8,
            )

            merged = o3d.io.read_point_cloud(str(output_path))
            merged_points = np.asarray(merged.points)
            merged_colors = np.asarray(merged.colors)
            merged_normals = np.asarray(merged.normals)
            self.assertGreater(len(merged_points), 50)
            self.assertEqual(len(merged_points), len(merged_colors))
            self.assertEqual(len(merged_points), len(merged_normals))
            self.assertTrue(np.isfinite(merged_points).all())
            np.testing.assert_allclose(
                np.linalg.norm(merged_normals, axis=1),
                1.0,
                atol=1e-5,
            )
            self.assertEqual(stats["processing_backend"], "open3d+trimesh")
            self.assertEqual(stats["input_clouds"], 2)
            self.assertEqual(stats["validated_points"], len(merged_points))


if __name__ == "__main__":
    unittest.main()


class CylinderCropTests(unittest.TestCase):
    def test_keeps_points_inside_the_radius_and_axial_bounds(self):
        crop = pointcloud_processing.CylinderCrop(
            center=(0.0, 0.0, 0.1),
            axis=(1.0, 0.0, 0.0),
            radius_m=0.08,
            axial_half_length_m=0.15,
        )
        points = np.array([
            [0.000, 0.00, 0.100],   # on the axis
            [0.140, 0.00, 0.100],   # far along the axis, inside the axial bound
            [0.000, 0.05, 0.100],   # radial 0.05, inside
            [0.000, 0.10, 0.100],   # radial 0.10, outside
            [0.160, 0.00, 0.100],   # beyond the axial bound
            [0.000, 0.07, 0.170],   # radial 0.099 on the ring, outside
        ])
        self.assertEqual(
            crop.mask(points).tolist(),
            [True, True, True, False, False, False],
        )

    def test_normalizes_an_unnormalized_axis(self):
        crop = pointcloud_processing.CylinderCrop(
            center=(0.0, 0.0, 0.0),
            axis=(3.0, 0.0, 0.0),
            radius_m=0.05,
            axial_half_length_m=0.10,
        )
        points = np.array([[0.09, 0.0, 0.0], [0.11, 0.0, 0.0]])
        self.assertEqual(crop.mask(points).tolist(), [True, False])

    def test_rejects_degenerate_geometry(self):
        for kwargs in (
            {"axis": (0.0, 0.0, 0.0), "radius_m": 0.05, "axial_half_length_m": 0.1},
            {"axis": (1.0, 0.0, 0.0), "radius_m": 0.0, "axial_half_length_m": 0.1},
            {"axis": (1.0, 0.0, 0.0), "radius_m": 0.05, "axial_half_length_m": -0.1},
        ):
            with self.assertRaises(ValueError):
                pointcloud_processing.CylinderCrop(center=(0.0, 0.0, 0.0), **kwargs)

    def test_transform_and_clean_applies_a_cylinder_crop(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "frame.ply"
            # Two points near the axis and two on the enclosure ring; a cube
            # crop wide enough for the axial spread would keep the ring points.
            points = np.array([
                [0.000, 0.00, 0.00],
                [0.130, 0.00, 0.00],
                [0.000, 0.10, 0.00],
                [0.130, 0.00, 0.10],
            ])
            write_cloud(source, points, np.tile([0.5, 0.5, 0.5], (len(points), 1)))

            outputs, stats = pointcloud_processing.transform_and_clean_clouds(
                [source],
                [np.eye(4)],
                root / "transformed",
                root / "matrices",
                crop_bounds=pointcloud_processing.CylinderCrop(
                    center=(0.0, 0.0, 0.0),
                    axis=(1.0, 0.0, 0.0),
                    radius_m=0.08,
                    axial_half_length_m=0.15,
                ),
                skip_sor=True,
                sor_neighbors=10,
                sor_sigma=2.0,
            )

            kept = np.asarray(o3d.io.read_point_cloud(str(outputs[0])).points)
            self.assertEqual(stats[0]["cropped_points"], 2)
            np.testing.assert_allclose(
                np.sort(kept[:, 0]), [0.0, 0.13], atol=1e-6,
            )


class LargestComponentTests(unittest.TestCase):
    """SOR cannot see a compact blob that floats clear of the object."""

    @staticmethod
    def _cloud(points):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
        return cloud

    @staticmethod
    def _patch(origin, size_m, spacing_m=0.001):
        """A dense surface patch, at the ~1 mm spacing a real merged scan has."""
        n = int(size_m / spacing_m)
        grid = np.arange(n) * spacing_m
        xs, ys = np.meshgrid(grid, grid)
        points = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
        return points + np.asarray(origin, dtype=float)

    def test_a_detached_blob_is_discarded(self):
        # 2500-point body, 16-point blob: below the 1 percent (25 point) bar.
        body = self._patch([0.0, 0.0, 0.14], 0.05)
        debris = self._patch([0.09, 0.0, 0.14], 0.004)
        cloud = self._cloud(np.vstack([body, debris]))
        keep = pointcloud_processing.largest_component_indices(
            o3d, cloud, eps_m=0.003, min_points=10, min_fraction=0.01,
        )
        self.assertEqual(len(keep), len(body))
        self.assertLess(np.asarray(cloud.points)[keep][:, 0].max(), 0.06)

    def test_a_large_detached_region_is_kept(self):
        """A real disconnected part of the object must survive."""
        body = self._patch([0.0, 0.0, 0.14], 0.05)
        second = self._patch([0.09, 0.0, 0.14], 0.04)
        cloud = self._cloud(np.vstack([body, second]))
        keep = pointcloud_processing.largest_component_indices(
            o3d, cloud, eps_m=0.003, min_points=10, min_fraction=0.01,
        )
        self.assertEqual(len(keep), len(body) + len(second))

    def test_a_zero_fraction_keeps_everything(self):
        cloud = self._cloud(np.vstack([
            self._patch([0.0, 0.0, 0.14], 0.02),
            self._patch([0.09, 0.0, 0.14], 0.003),
        ]))
        keep = pointcloud_processing.largest_component_indices(
            o3d, cloud, eps_m=0.003, min_points=10, min_fraction=0.0,
        )
        self.assertEqual(len(keep), len(cloud.points))

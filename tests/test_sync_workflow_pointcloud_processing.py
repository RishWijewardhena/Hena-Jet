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

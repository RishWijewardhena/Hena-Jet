from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

SYNC_WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "scripts" / "sync_workflow"
sys.path.insert(0, str(SYNC_WORKFLOW_DIR))

from pointcloud_export import backproject_to_points  # noqa: E402
import tsdf_fusion  # noqa: E402


INTRINSICS = {"width": 64, "height": 48, "fx": 60.0, "fy": 60.0, "cx": 31.5, "cy": 23.5}


def plane_depth(distance_m: float, shape=(48, 64)) -> np.ndarray:
    return np.full(shape, distance_m, dtype=np.float32)


def write_camera_frame_ply(path: Path, depth: np.ndarray, color: np.ndarray) -> None:
    """Save a PLY exactly the way the capture path does, in the camera frame."""
    import open3d as o3d

    points, colors = backproject_to_points(depth, color, INTRINSICS)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(path), cloud)


class DepthFromPlyTests(unittest.TestCase):
    """The PLY reprojection has to be exact, or fusion silently degrades."""

    def test_round_trip_recovers_depth_and_colour(self):
        depth = np.zeros((48, 64), dtype=np.float32)
        depth[10:40, 8:56] = np.linspace(0.12, 0.22, 48 * 30).reshape(30, 48)
        color = np.zeros((48, 64, 3), dtype=np.uint8)
        color[..., 0] = 200
        color[..., 1] = 100

        with TemporaryDirectory() as tmp:
            ply = Path(tmp) / "frame.ply"
            write_camera_frame_ply(ply, depth, color)
            recovered, recovered_color = tsdf_fusion.depth_from_ply(ply, INTRINSICS)

        np.testing.assert_allclose(recovered, depth, atol=0.0)
        valid = depth > 0
        np.testing.assert_array_equal(recovered_color[valid], color[valid])

    def test_rejects_a_cloud_that_is_not_in_the_camera_frame(self):
        depth = plane_depth(0.18)
        color = np.zeros((48, 64, 3), dtype=np.uint8)
        with TemporaryDirectory() as tmp:
            import open3d as o3d

            ply = Path(tmp) / "frame.ply"
            write_camera_frame_ply(ply, depth, color)
            cloud = o3d.io.read_point_cloud(str(ply))
            # A pose-transformed cloud no longer lands on the pixel grid.
            rotated = np.eye(4)
            rotated[:3, 3] = [0.05, 0.03, 0.01]
            cloud.transform(rotated)
            moved = Path(tmp) / "moved.ply"
            o3d.io.write_point_cloud(str(moved), cloud)

            with self.assertRaisesRegex(ValueError, "camera-frame"):
                tsdf_fusion.depth_from_ply(moved, INTRINSICS)

    def test_missing_intrinsics_keys_are_named(self):
        with TemporaryDirectory() as tmp:
            ply = Path(tmp) / "frame.ply"
            write_camera_frame_ply(ply, plane_depth(0.18), np.zeros((48, 64, 3), np.uint8))
            with self.assertRaisesRegex(ValueError, "fx"):
                tsdf_fusion.depth_from_ply(ply, {"width": 64, "height": 48, "cy": 1.0})


class IntegrationTests(unittest.TestCase):
    def _plane_frames(self, distance_m: float, poses):
        color = np.full((48, 64, 3), 180, dtype=np.uint8)
        return [
            tsdf_fusion.FusionFrame(plane_depth(distance_m), color, np.asarray(p, float))
            for p in poses
        ]

    def test_identity_pose_puts_the_surface_at_the_measured_depth(self):
        frames = self._plane_frames(0.18, [np.eye(4)])
        volume = tsdf_fusion.integrate(
            frames, INTRINSICS, voxel_length_m=0.002, sdf_trunc_m=0.006,
            depth_min_m=0.05, depth_max_m=0.30,
        )
        _, cloud = tsdf_fusion.extract(volume)
        z = np.asarray(cloud.points)[:, 2]
        self.assertGreater(len(z), 0)
        self.assertAlmostEqual(float(np.median(z)), 0.18, delta=0.004)

    def test_pose_is_inverted_for_integration(self):
        """A translated pose must move the surface with the camera, not against it."""
        pose = np.eye(4)
        pose[2, 3] = 0.10  # camera sits 0.10 m along +Z in the reference frame
        frames = self._plane_frames(0.18, [pose])
        volume = tsdf_fusion.integrate(
            frames, INTRINSICS, voxel_length_m=0.002, sdf_trunc_m=0.006,
            depth_min_m=0.05, depth_max_m=0.30,
        )
        _, cloud = tsdf_fusion.extract(volume)
        z = np.asarray(cloud.points)[:, 2]
        # Surface belongs at 0.10 + 0.18; using the pose uncorrected gives 0.08.
        self.assertAlmostEqual(float(np.median(z)), 0.28, delta=0.004)

    def test_crop_restricts_the_extracted_surface(self):
        frames = self._plane_frames(0.18, [np.eye(4)])
        volume = tsdf_fusion.integrate(
            frames, INTRINSICS, voxel_length_m=0.002, sdf_trunc_m=0.006,
            depth_min_m=0.05, depth_max_m=0.30,
        )
        bounds = (-0.01, -0.01, 0.17, 0.01, 0.01, 0.19)
        _, cloud = tsdf_fusion.extract(volume, crop_bounds=bounds)
        points = np.asarray(cloud.points)
        self.assertGreater(len(points), 0)
        self.assertTrue((points[:, 0] >= -0.0101).all() and (points[:, 0] <= 0.0101).all())
        self.assertTrue((points[:, 2] >= 0.1699).all() and (points[:, 2] <= 0.1901).all())

    def test_crop_bounds_limit_what_is_integrated(self):
        """Masking at integration time is what keeps the volume bounded."""
        frames = self._plane_frames(0.18, [np.eye(4)])
        wide = tsdf_fusion.integrate(
            frames, INTRINSICS, voxel_length_m=0.002, sdf_trunc_m=0.006,
            depth_min_m=0.05, depth_max_m=0.30,
        )
        narrow = tsdf_fusion.integrate(
            frames, INTRINSICS, voxel_length_m=0.002, sdf_trunc_m=0.006,
            depth_min_m=0.05, depth_max_m=0.30,
            crop_bounds=(-0.01, -0.01, 0.17, 0.01, 0.01, 0.19),
        )
        _, wide_cloud = tsdf_fusion.extract(wide)
        _, narrow_cloud = tsdf_fusion.extract(narrow)
        self.assertLess(len(narrow_cloud.points), len(wide_cloud.points))
        self.assertGreater(len(narrow_cloud.points), 0)

    def test_crop_that_excludes_everything_is_reported(self):
        frames = self._plane_frames(0.18, [np.eye(4)])
        with self.assertRaisesRegex(RuntimeError, "fusion range"):
            tsdf_fusion.integrate(
                frames, INTRINSICS, voxel_length_m=0.002, sdf_trunc_m=0.006,
                depth_min_m=0.05, depth_max_m=0.30,
                crop_bounds=(1.0, 1.0, 1.0, 1.1, 1.1, 1.1),
            )

    def test_truncation_below_voxel_length_is_rejected(self):
        frames = self._plane_frames(0.18, [np.eye(4)])
        with self.assertRaisesRegex(ValueError, "interpolate"):
            tsdf_fusion.integrate(
                frames, INTRINSICS, voxel_length_m=0.004, sdf_trunc_m=0.001,
                depth_min_m=0.05, depth_max_m=0.30,
            )

    def test_frames_outside_the_depth_range_are_reported(self):
        frames = self._plane_frames(0.50, [np.eye(4)])
        with self.assertRaisesRegex(RuntimeError, "fusion range"):
            tsdf_fusion.integrate(
                frames, INTRINSICS, voxel_length_m=0.002, sdf_trunc_m=0.006,
                depth_min_m=0.05, depth_max_m=0.30,
            )

    def test_averaging_many_noisy_views_beats_a_single_view(self):
        """The reason for fusing at all: independent per-view error cancels."""
        rng = np.random.default_rng(0)
        color = np.full((48, 64, 3), 180, dtype=np.uint8)
        noisy = [
            tsdf_fusion.FusionFrame(
                (plane_depth(0.18) + rng.normal(0, 0.002, (48, 64))).astype(np.float32),
                color, np.eye(4),
            )
            for _ in range(24)
        ]
        settings = dict(voxel_length_m=0.001, sdf_trunc_m=0.004,
                        depth_min_m=0.05, depth_max_m=0.30)
        _, one = tsdf_fusion.extract(tsdf_fusion.integrate(noisy[:1], INTRINSICS, **settings))
        _, many = tsdf_fusion.extract(tsdf_fusion.integrate(noisy, INTRINSICS, **settings))
        spread_one = float(np.std(np.asarray(one.points)[:, 2]))
        spread_many = float(np.std(np.asarray(many.points)[:, 2]))
        self.assertLess(spread_many, spread_one)


class MeshCleanupTests(unittest.TestCase):
    def _mesh_with_missing_face(self, side_m: float):
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh.create_box(
            width=side_m, height=side_m, depth=side_m,
        )
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        top = np.flatnonzero(np.all(vertices[triangles, 2] == side_m, axis=1))
        mesh.remove_triangles_by_index(top.tolist())
        mesh.remove_unreferenced_vertices()
        return mesh

    def test_cleanup_fills_only_a_small_hole(self):
        mesh = self._mesh_with_missing_face(0.002)

        cleaned, stats = tsdf_fusion.clean_mesh(mesh)

        self.assertGreater(len(cleaned.triangles), len(mesh.triangles))
        self.assertGreaterEqual(stats["repaired_holes"], 1)

    def test_cleanup_preserves_a_hole_larger_than_three_mm(self):
        mesh = self._mesh_with_missing_face(0.006)

        cleaned, stats = tsdf_fusion.clean_mesh(mesh)

        self.assertEqual(len(cleaned.triangles), len(mesh.triangles))
        self.assertEqual(stats["repaired_holes"], 0)

    def test_cleanup_removes_a_small_detached_fragment_and_smooths_ripples(self):
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.02, resolution=10)
        vertices = np.asarray(mesh.vertices)
        vertices[:, 2] += 0.0008 * np.sin(vertices[:, 0] * 1200.0)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        raw_radial_spread = float(np.std(np.linalg.norm(vertices, axis=1)))
        fragment = o3d.geometry.TriangleMesh.create_tetrahedron(radius=0.0005)
        fragment.translate((0.10, 0.0, 0.0))
        mesh += fragment

        cleaned, stats = tsdf_fusion.clean_mesh(mesh)

        cleaned_extent = cleaned.get_axis_aligned_bounding_box().get_extent()
        cleaned_radial_spread = float(
            np.std(np.linalg.norm(np.asarray(cleaned.vertices), axis=1))
        )
        self.assertLess(len(cleaned.triangles), len(mesh.triangles))
        self.assertGreaterEqual(stats["removed_components"], 1)
        self.assertLess(float(cleaned_extent[0]), 0.05)
        self.assertLess(cleaned_radial_spread, raw_radial_spread)
        self.assertTrue(cleaned.is_edge_manifold(allow_boundary_edges=True))

    def test_fusion_keeps_raw_mesh_and_writes_cleaned_mesh(self):
        depth = plane_depth(0.18)
        color = np.full((48, 64, 3), 180, dtype=np.uint8)
        with TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "scan"
            input_dir.mkdir()
            ply_path = input_dir / "frame_0.0.ply"
            write_camera_frame_ply(ply_path, depth, color)
            output_dir = Path(tmp) / "reconstruction"

            stats = tsdf_fusion.fuse_captures(
                [ply_path], [np.eye(4)], INTRINSICS,
                input_dir=input_dir,
                output_dir=output_dir,
                voxel_length_m=0.002,
                sdf_trunc_m=0.006,
                depth_min_m=0.05,
                depth_max_m=0.30,
            )

            self.assertTrue((output_dir / "tsdf_mesh.ply").is_file())
            self.assertTrue((output_dir / "tsdf_mesh_cleaned.ply").is_file())
            self.assertEqual(stats["mesh_cleanup"]["taubin_iterations"], 10)
            self.assertEqual(stats["cleaned_mesh_path"], str(output_dir / "tsdf_mesh_cleaned.ply"))


if __name__ == "__main__":
    unittest.main()

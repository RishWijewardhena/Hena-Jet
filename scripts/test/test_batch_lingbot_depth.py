import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_lingbot_depth import (
    FramePaths,
    capture_arrays,
    combine_depth,
    discover_frames,
    frame_is_complete,
    normalized_intrinsics_matrix,
    output_paths_for_frame,
    write_point_cloud,
)


class BatchLingBotDepthTests(unittest.TestCase):
    def test_discover_frames_orders_angles_and_requires_matching_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for stem, angle in (("angle_0020p00", 20.0), ("angle_0010p00", 10.0)):
                np.savez(root / f"{stem}.npz", depth_image_m=np.ones((1, 1)))
                (root / f"{stem}.json").write_text(
                    json.dumps({"angle_deg": angle}),
                    encoding="utf-8",
                )

            frames = discover_frames(root)

            self.assertEqual([frame.stem for frame in frames], [
                "angle_0010p00",
                "angle_0020p00",
            ])

    def test_output_paths_keep_transform_compatible_ply_at_root(self):
        output_root = Path("captures/Plastic_Hand_full_lingbot")
        frame = FramePaths(
            stem="angle_0010p00",
            angle_deg=10.0,
            npz_path=Path("input.npz"),
            json_path=Path("input.json"),
        )

        paths = output_paths_for_frame(output_root, frame)

        self.assertEqual(paths.ply, output_root / "angle_0010p00.ply")
        self.assertEqual(
            paths.npz,
            output_root / "metadata" / "angle_0010p00.npz",
        )
        self.assertEqual(
            paths.json,
            output_root / "metadata" / "angle_0010p00.json",
        )
        self.assertEqual(
            paths.comparison,
            output_root / "qc" / "angle_0010p00_comparison.png",
        )

    def test_combine_depth_preserves_zed_and_filters_after_completion(self):
        raw = np.array([[0.20, 0.40, 0.00, np.nan]], dtype=np.float32)
        predicted = np.array([[0.21, 0.22, 0.25, 0.35]], dtype=np.float32)

        completed, raw_valid, filled = combine_depth(
            raw,
            predicted,
            max_output_depth_m=0.30,
        )

        np.testing.assert_allclose(completed, [[0.20, 0.00, 0.25, 0.00]])
        np.testing.assert_array_equal(raw_valid, [[True, True, False, False]])
        np.testing.assert_array_equal(filled, [[False, False, True, False]])

    def test_write_point_cloud_emits_readable_colored_ply(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "angle_0010p00.ply"
            points = np.array(
                [[0.0, 0.0, 0.2], [0.01, 0.0, 0.2]],
                dtype=np.float32,
            )
            colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)

            write_point_cloud(path, points, colors)

            import open3d as o3d

            cloud = o3d.io.read_point_cloud(str(path))
            self.assertEqual(len(cloud.points), 2)
            np.testing.assert_allclose(
                np.asarray(cloud.colors),
                colors.astype(np.float64) / 255.0,
            )

    def test_capture_arrays_validate_rgb_depth_and_intrinsics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            npz_path = root / "angle_0010p00.npz"
            json_path = root / "angle_0010p00.json"
            np.savez(
                npz_path,
                depth_image_m=np.full((2, 3), 0.2, dtype=np.float32),
                color_image=np.zeros((2, 3, 3), dtype=np.uint8),
                confidence_image=np.zeros((2, 3), dtype=np.uint8),
            )
            json_path.write_text(
                json.dumps(
                    {
                        "angle_deg": 10.0,
                        "camera_intrinsics": {
                            "fx": 2.0,
                            "fy": 2.0,
                            "cx": 1.0,
                            "cy": 0.5,
                            "width": 3,
                            "height": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )
            frame = FramePaths(
                stem="angle_0010p00",
                angle_deg=10.0,
                npz_path=npz_path,
                json_path=json_path,
            )

            depth, color, confidence, metadata = capture_arrays(frame)
            normalized = normalized_intrinsics_matrix(
                metadata["camera_intrinsics"]
            )

            self.assertEqual(depth.shape, (2, 3))
            self.assertEqual(color.shape, (2, 3, 3))
            self.assertEqual(confidence.shape, (2, 3))
            np.testing.assert_allclose(
                normalized,
                [[2.0 / 3.0, 0.0, 1.0 / 3.0],
                 [0.0, 1.0, 0.25],
                 [0.0, 0.0, 1.0]],
            )

    def test_frame_is_complete_requires_every_output_artifact(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            frame = FramePaths(
                stem="angle_0010p00",
                angle_deg=10.0,
                npz_path=Path("input.npz"),
                json_path=Path("input.json"),
            )
            paths = output_paths_for_frame(output_root, frame)
            for path in paths.required_files():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"complete")

            self.assertTrue(frame_is_complete(paths))
            paths.filled_mask.unlink()
            self.assertFalse(frame_is_complete(paths))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "orbbec_aruco_alignment"))
from aruco_common import (
    apply_roi_mask,
    camera_matrix_from_intrinsics,
    clean_depth_image,
    color_frame_to_rgb,
    detect_markers,
    extrinsic_from_rvec_tvec,
    load_board,
    make_rgbd,
    mask_marker_depth,
    match_board_corners,
    open3d_intrinsic_from_dict,
    pose_summary,
    reprojection_error_px,
    solve_board_pose,
)


class TestOrbbecArucoAlignment(unittest.TestCase):
    def test_color_frame_to_rgb(self):
        img_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        img_bgr[:, :, 0] = 255  # Blue channel
        rgb = color_frame_to_rgb(img_bgr)
        self.assertEqual(rgb.shape, (100, 100, 3))
        self.assertEqual(rgb.dtype, np.uint8)

    def test_clean_depth_image_and_mask(self):
        depth = np.ones((100, 100), dtype=np.float32) * 0.5
        depth[0:10, 0:10] = 0.05  # below min
        depth[90:100, 90:100] = 1.5  # above max

        cleaned = clean_depth_image(depth, min_depth_m=0.1, max_depth_m=1.0, roi=(0.1, 0.1, 0.9, 0.9))
        self.assertEqual(cleaned[0:10, 0:10].max(), 0.0)
        self.assertEqual(cleaned[90:100, 90:100].max(), 0.0)
        self.assertEqual(cleaned[50, 50], 0.5)

    def test_board_detection_and_pnp(self):
        board_json = Path(__file__).resolve().parent.parent / "outputs" / "aruco_table_board" / "aruco_table_board.json"
        board_png = Path(__file__).resolve().parent.parent / "outputs" / "aruco_table_board" / "aruco_table_board.png"
        if not board_json.exists() or not board_png.exists():
            self.skipTest("Generated board files not found")

        board = load_board(board_json)
        img = cv2.imread(str(board_png))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        corners, ids, _ = detect_markers(rgb, board)
        self.assertEqual(len(ids), 12)

        h, w = rgb.shape[:2]
        intrinsics = {"fx": 800.0, "fy": 800.0, "cx": w / 2.0, "cy": h / 2.0, "width": w, "height": h}
        camera_matrix = camera_matrix_from_intrinsics(intrinsics)
        dist_coeffs = np.zeros((8, 1), dtype=np.float64)

        obj_pts, img_pts, used_ids = match_board_corners(board, corners, ids)
        ok, rvec, tvec = solve_board_pose(obj_pts, img_pts, camera_matrix, dist_coeffs, 2, used_ids)
        self.assertTrue(ok)
        err = reprojection_error_px(obj_pts, img_pts, rvec, tvec, camera_matrix, dist_coeffs)
        self.assertLess(err, 2.0)

        extrinsic = extrinsic_from_rvec_tvec(rvec, tvec)
        summary = pose_summary(extrinsic)
        self.assertIn("tilt_from_table_normal_deg", summary)


if __name__ == "__main__":
    unittest.main()

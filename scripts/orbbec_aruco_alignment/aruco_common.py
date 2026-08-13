#!/usr/bin/env python3
"""Shared helpers for ZED-M ArUco table-board pose alignment."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d


ARUCO_DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}


def get_aruco_dictionary(name: str):
    if name not in ARUCO_DICTIONARIES:
        valid = ", ".join(sorted(ARUCO_DICTIONARIES))
        raise ValueError(f"Unsupported ArUco dictionary {name!r}. Valid values: {valid}")
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[name])


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_board(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        board = json.load(file)
    board["markers_by_id"] = {int(marker["id"]): marker for marker in board["markers"]}
    return board


def make_detector(dictionary_name: str):
    dictionary = get_aruco_dictionary(dictionary_name)
    parameters = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def normalized_roi_to_pixels(
    roi: tuple[float, float, float, float],
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max = roi
    if not (0.0 <= x_min < x_max <= 1.0 and 0.0 <= y_min < y_max <= 1.0):
        raise ValueError("--roi must satisfy 0<=X_MIN<X_MAX<=1 and 0<=Y_MIN<Y_MAX<=1")
    height, width = image_shape[:2]
    x0 = int(round(x_min * width))
    x1 = int(round(x_max * width))
    y0 = int(round(y_min * height))
    y1 = int(round(y_max * height))
    return x0, y0, x1, y1


def apply_roi_mask(
    image: np.ndarray,
    roi: tuple[float, float, float, float],
    fill_value: float | int = 0,
) -> np.ndarray:
    if roi == (0.0, 0.0, 1.0, 1.0):
        return image
    x0, y0, x1, y1 = normalized_roi_to_pixels(roi, image.shape)
    result = np.full_like(image, fill_value)
    result[y0:y1, x0:x1] = image[y0:y1, x0:x1]
    return result


def color_image_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2).astype(np.uint8)
    if image.shape[2] >= 3:
        # ZED returns BGRA. Take BGR and reverse to RGB.
        return image[:, :, :3][:, :, ::-1].astype(np.uint8)
    raise ValueError(f"Unsupported color image shape: {image.shape}")


def clean_depth_image(
    depth_image: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    roi: tuple[float, float, float, float],
) -> np.ndarray:
    depth = np.asarray(depth_image, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    valid = np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    cleaned = np.zeros(depth.shape, dtype=np.float32)
    cleaned[valid] = depth[valid]
    return apply_roi_mask(cleaned, roi, fill_value=0)


def camera_matrix_from_intrinsics(intrinsics: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
            [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def open3d_intrinsic_from_dict(intrinsics: dict[str, float]) -> o3d.camera.PinholeCameraIntrinsic:
    return o3d.camera.PinholeCameraIntrinsic(
        int(intrinsics["width"]),
        int(intrinsics["height"]),
        float(intrinsics["fx"]),
        float(intrinsics["fy"]),
        float(intrinsics["cx"]),
        float(intrinsics["cy"]),
    )


def make_rgbd(color_rgb: np.ndarray, depth_m: np.ndarray, depth_trunc_m: float) -> o3d.geometry.RGBDImage:
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(color_rgb)),
        o3d.geometry.Image(np.ascontiguousarray(depth_m)),
        depth_scale=1.0,
        depth_trunc=depth_trunc_m,
        convert_rgb_to_intensity=False,
    )


def detect_markers(color_rgb: np.ndarray, board: dict[str, Any]):
    detector = make_detector(board["dictionary"])
    gray = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    if ids is None:
        ids = np.empty((0, 1), dtype=np.int32)
    return corners, ids, rejected


def match_board_corners(
    board: dict[str, Any],
    marker_corners,
    marker_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_marker_ids: list[int] = []
    markers_by_id = board["markers_by_id"]

    for corners, marker_id_array in zip(marker_corners, marker_ids.reshape(-1)):
        marker_id = int(marker_id_array)
        if marker_id not in markers_by_id:
            continue
        marker = markers_by_id[marker_id]
        object_points.extend(np.asarray(marker["corners_m"], dtype=np.float64))
        image_points.extend(np.asarray(corners, dtype=np.float64).reshape(4, 2))
        used_marker_ids.append(marker_id)

    return (
        np.asarray(object_points, dtype=np.float64),
        np.asarray(image_points, dtype=np.float64),
        used_marker_ids,
    )


def solve_board_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    min_markers: int,
    used_marker_ids: list[int],
) -> tuple[bool, np.ndarray, np.ndarray]:
    if len(set(used_marker_ids)) < min_markers or object_points.shape[0] < 8:
        return False, np.zeros((3, 1), dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    return bool(ok), rvec, tvec


def extrinsic_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(rvec)
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return extrinsic


def transform_inverse(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def reprojection_error_px(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    if object_points.size == 0:
        return float("inf")
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - image_points.reshape(-1, 2), axis=1)
    return float(np.mean(errors))


def mask_marker_depth(
    depth_m: np.ndarray,
    marker_corners,
    padding_px: int,
) -> np.ndarray:
    if padding_px < 0:
        raise ValueError("--marker-mask-padding-px must be zero or positive")
    masked = depth_m.copy()
    for corners in marker_corners:
        polygon = np.asarray(corners, dtype=np.int32).reshape(4, 2)
        if padding_px > 0:
            center = polygon.mean(axis=0, keepdims=True)
            direction = polygon.astype(np.float32) - center
            norms = np.linalg.norm(direction, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            polygon = np.round(polygon + direction / norms * padding_px).astype(np.int32)
        cv2.fillConvexPoly(masked, polygon, 0.0)
    return masked


def draw_pose_overlay(
    color_rgb: np.ndarray,
    marker_corners,
    marker_ids: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray | None = None,
    tvec: np.ndarray | None = None,
    axis_length_m: float = 0.05,
) -> np.ndarray:
    overlay_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    if marker_ids is not None and marker_ids.size > 0:
        cv2.aruco.drawDetectedMarkers(overlay_bgr, marker_corners, marker_ids)
    if rvec is not None and tvec is not None:
        cv2.drawFrameAxes(overlay_bgr, camera_matrix, dist_coeffs, rvec, tvec, axis_length_m)
    return overlay_bgr


def pose_summary(world_to_camera: np.ndarray) -> dict[str, Any]:
    camera_to_world = transform_inverse(world_to_camera)
    position = camera_to_world[:3, 3]
    forward_world = camera_to_world[:3, 2]
    table_normal_camera = world_to_camera[:3, :3] @ np.array([0.0, 0.0, 1.0])
    tilt_from_table_normal_deg = math.degrees(
        math.acos(float(np.clip(table_normal_camera[2], -1.0, 1.0)))
    )
    return {
        "world_to_camera": world_to_camera.tolist(),
        "camera_to_world": camera_to_world.tolist(),
        "camera_position_world_m": position.tolist(),
        "camera_forward_world": forward_world.tolist(),
        "table_normal_in_camera": table_normal_camera.tolist(),
        "tilt_from_table_normal_deg": tilt_from_table_normal_deg,
    }

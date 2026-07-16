#!/usr/bin/env python3
"""Estimate RGB-D frame poses with ORB/SIFT feature matches and depth RANSAC.

This is a Stage 1 markerless tracking baseline. It reads an Open3D-style ZED
dataset:

    dataset/
      image/000000.png
      depth/000000.png
      intrinsic.json
      capture_config.json

For each accepted frame pair it:

1. detects ORB or SIFT features in the RGB images,
2. matches descriptors between the last accepted frame and the candidate frame,
3. lifts matched pixels into 3D using the depth images,
4. estimates a rigid transform with RANSAC,
5. saves camera-to-world poses and a JSON diagnostics report.

The output is meant to answer whether markerless tracking is viable before
connecting this to TSDF fusion.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


DEFAULT_DATASET = Path("datasets/zed_m_open3d")
DEFAULT_POSES_OUT = Path("outputs/feature_tracking/rgbd_feature_poses.npy")
DEFAULT_REPORT_OUT = Path("outputs/feature_tracking/rgbd_feature_report.json")
DEFAULT_DEBUG_DIR = Path("outputs/feature_tracking/debug_matches")


@dataclass
class FramePaths:
    index: int
    stem: str
    color_path: Path
    depth_path: Path


@dataclass
class FrameData:
    paths: FramePaths
    color_bgr: np.ndarray
    gray: np.ndarray
    depth_m: np.ndarray
    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray | None
    camera_to_world: np.ndarray


@dataclass
class PairResult:
    accepted: bool
    reason: str
    source_index: int
    target_index: int
    raw_matches: int
    depth_valid_matches: int
    inliers: int
    inlier_ratio: float
    translation_m: float
    rotation_deg: float
    transform_source_from_target: np.ndarray
    inlier_match_indices: np.ndarray
    all_matches: list[cv2.DMatch]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track an RGB-D sequence with ORB/SIFT + depth RANSAC."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--intrinsic", type=Path, default=None)
    parser.add_argument("--poses-out", type=Path, default=DEFAULT_POSES_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--method", choices=["orb", "sift"], default="orb")
    parser.add_argument("--max-features", type=int, default=4000)
    parser.add_argument("--ratio-test", type=float, default=0.75)
    parser.add_argument("--depth-scale", type=float, default=None)
    parser.add_argument("--depth-min-m", type=float, default=None)
    parser.add_argument("--depth-max-m", type=float, default=None)
    parser.add_argument("--ransac-iterations", type=int, default=800)
    parser.add_argument("--ransac-threshold-m", type=float, default=0.012)
    parser.add_argument("--min-depth-matches", type=int, default=20)
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.30)
    parser.add_argument("--max-step-translation-m", type=float, default=0.08)
    parser.add_argument("--max-step-rotation-deg", type=float, default=20.0)
    parser.add_argument(
        "--debug-pairs",
        type=int,
        default=20,
        help="Write match overlay images for the first N accepted/rejected pairs.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset does not exist: {args.dataset}")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be zero or positive")
    if args.max_features <= 0:
        raise ValueError("--max-features must be positive")
    if not 0.0 < args.ratio_test < 1.0:
        raise ValueError("--ratio-test must be between 0 and 1")
    if args.depth_scale is not None and args.depth_scale <= 0:
        raise ValueError("--depth-scale must be positive")
    if args.ransac_iterations <= 0:
        raise ValueError("--ransac-iterations must be positive")
    if args.ransac_threshold_m <= 0:
        raise ValueError("--ransac-threshold-m must be positive")
    if args.min_depth_matches < 3:
        raise ValueError("--min-depth-matches must be at least 3")
    if args.min_inliers < 3:
        raise ValueError("--min-inliers must be at least 3")
    if not 0.0 < args.min_inlier_ratio <= 1.0:
        raise ValueError("--min-inlier-ratio must be between 0 and 1")
    if args.max_step_translation_m <= 0:
        raise ValueError("--max-step-translation-m must be positive")
    if args.max_step_rotation_deg <= 0:
        raise ValueError("--max-step-rotation-deg must be positive")
    if args.debug_pairs < 0:
        raise ValueError("--debug-pairs must be zero or positive")


def read_capture_config(dataset: Path) -> dict:
    path = dataset / "capture_config.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_intrinsic_path(args: argparse.Namespace, config: dict) -> Path:
    if args.intrinsic is not None:
        return args.intrinsic
    if config.get("path_intrinsic"):
        configured = Path(config["path_intrinsic"])
        if configured.exists():
            return configured
        candidate = args.dataset / configured.name
        if candidate.exists():
            return candidate
    return args.dataset / "intrinsic.json"


def find_frame_pairs(dataset: Path, stride: int, max_frames: int) -> list[FramePaths]:
    image_dir = dataset / "image"
    depth_dir = dataset / "depth"
    if not image_dir.exists() or not depth_dir.exists():
        raise FileNotFoundError(f"Expected image/ and depth/ folders in {dataset}")

    color_paths = {path.stem: path for path in sorted(image_dir.glob("*.png"))}
    depth_paths = {path.stem: path for path in sorted(depth_dir.glob("*.png"))}
    stems = sorted(set(color_paths) & set(depth_paths))
    stems = stems[::stride]
    if max_frames:
        stems = stems[:max_frames]
    if len(stems) < 2:
        raise ValueError("Need at least two matching RGB/depth frames")
    return [
        FramePaths(i, stem, color_paths[stem], depth_paths[stem])
        for i, stem in enumerate(stems)
    ]


def make_detector(method: str, max_features: int) -> cv2.Feature2D:
    if method == "orb":
        return cv2.ORB_create(nfeatures=max_features, fastThreshold=7)
    if method == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("This OpenCV build does not expose cv2.SIFT_create()")
        return cv2.SIFT_create(nfeatures=max_features)
    raise ValueError(f"Unsupported method: {method}")


def read_depth_meters(path: Path, depth_scale: float) -> np.ndarray:
    depth_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Could not read depth image: {path}")
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]
    return depth_raw.astype(np.float32) / float(depth_scale)


def read_frame(
    paths: FramePaths,
    detector: cv2.Feature2D,
    depth_scale: float,
    camera_to_world: np.ndarray,
) -> FrameData:
    color_bgr = cv2.imread(str(paths.color_path), cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise FileNotFoundError(f"Could not read color image: {paths.color_path}")
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    depth_m = read_depth_meters(paths.depth_path, depth_scale)
    return FrameData(
        paths=paths,
        color_bgr=color_bgr,
        gray=gray,
        depth_m=depth_m,
        keypoints=tuple(keypoints),
        descriptors=descriptors,
        camera_to_world=camera_to_world,
    )


def match_descriptors(
    source: FrameData,
    target: FrameData,
    method: str,
    ratio_test: float,
) -> list[cv2.DMatch]:
    if source.descriptors is None or target.descriptors is None:
        return []
    if len(source.descriptors) < 2 or len(target.descriptors) < 2:
        return []

    norm = cv2.NORM_HAMMING if method == "orb" else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm, crossCheck=False)
    pairs = matcher.knnMatch(source.descriptors, target.descriptors, k=2)
    good: list[cv2.DMatch] = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < ratio_test * second.distance:
            good.append(first)
    return good


def backproject_pixel(
    keypoint: cv2.KeyPoint,
    depth_m: np.ndarray,
    intrinsic_matrix: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
) -> np.ndarray | None:
    u_float, v_float = keypoint.pt
    u = int(round(u_float))
    v = int(round(v_float))
    if v < 0 or v >= depth_m.shape[0] or u < 0 or u >= depth_m.shape[1]:
        return None
    z = float(depth_m[v, u])
    if not math.isfinite(z) or z < depth_min_m or z > depth_max_m:
        return None
    fx = float(intrinsic_matrix[0, 0])
    fy = float(intrinsic_matrix[1, 1])
    cx = float(intrinsic_matrix[0, 2])
    cy = float(intrinsic_matrix[1, 2])
    x = (u_float - cx) * z / fx
    y = (v_float - cy) * z / fy
    return np.array([x, y, z], dtype=np.float64)


def lift_matches_to_3d(
    source: FrameData,
    target: FrameData,
    matches: list[cv2.DMatch],
    intrinsic_matrix: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
) -> tuple[np.ndarray, np.ndarray, list[cv2.DMatch]]:
    source_points = []
    target_points = []
    valid_matches = []
    for match in matches:
        source_point = backproject_pixel(
            source.keypoints[match.queryIdx],
            source.depth_m,
            intrinsic_matrix,
            depth_min_m,
            depth_max_m,
        )
        target_point = backproject_pixel(
            target.keypoints[match.trainIdx],
            target.depth_m,
            intrinsic_matrix,
            depth_min_m,
            depth_max_m,
        )
        if source_point is None or target_point is None:
            continue
        source_points.append(source_point)
        target_points.append(target_point)
        valid_matches.append(match)
    if not source_points:
        return np.empty((0, 3)), np.empty((0, 3)), []
    return np.vstack(source_points), np.vstack(target_points), valid_matches


def rigid_transform(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """Return transform that maps source_points onto target_points."""
    source_centroid = source_points.mean(axis=0)
    target_centroid = target_points.mean(axis=0)
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid
    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_centroid - rotation @ source_centroid
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def rotation_degrees(transform: np.ndarray) -> float:
    cos_angle = (float(np.trace(transform[:3, :3])) - 1.0) / 2.0
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def estimate_rigid_ransac(
    source_points: np.ndarray,
    target_points: np.ndarray,
    iterations: int,
    threshold_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(source_points) < 3:
        return np.eye(4, dtype=np.float64), np.zeros(len(source_points), dtype=bool)

    rng = np.random.default_rng(42)
    best_inliers = np.zeros(len(source_points), dtype=bool)
    for _ in range(iterations):
        sample = rng.choice(len(source_points), size=3, replace=False)
        try:
            candidate = rigid_transform(source_points[sample], target_points[sample])
        except np.linalg.LinAlgError:
            continue
        errors = np.linalg.norm(transform_points(candidate, source_points) - target_points, axis=1)
        inliers = errors <= threshold_m
        if int(inliers.sum()) > int(best_inliers.sum()):
            best_inliers = inliers

    if int(best_inliers.sum()) >= 3:
        transform = rigid_transform(source_points[best_inliers], target_points[best_inliers])
    else:
        transform = np.eye(4, dtype=np.float64)
    return transform, best_inliers


def estimate_pair_transform(
    source: FrameData,
    target: FrameData,
    args: argparse.Namespace,
    intrinsic_matrix: np.ndarray,
) -> PairResult:
    """Estimate source_from_target transform.

    `source` is the last accepted frame. `target` is the candidate frame.
    """
    matches = match_descriptors(source, target, args.method, args.ratio_test)
    source_points, target_points, valid_matches = lift_matches_to_3d(
        source,
        target,
        matches,
        intrinsic_matrix,
        args.depth_min_m,
        args.depth_max_m,
    )
    if len(valid_matches) < args.min_depth_matches:
        return PairResult(
            False,
            "too_few_depth_valid_matches",
            source.paths.index,
            target.paths.index,
            len(matches),
            len(valid_matches),
            0,
            0.0,
            0.0,
            0.0,
            np.eye(4),
            np.array([], dtype=np.int64),
            matches,
        )

    source_from_target, inliers = estimate_rigid_ransac(
        target_points,
        source_points,
        args.ransac_iterations,
        args.ransac_threshold_m,
    )
    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / max(1, len(valid_matches))
    translation_m = float(np.linalg.norm(source_from_target[:3, 3]))
    rotation_deg = rotation_degrees(source_from_target)
    accepted = True
    reason = "accepted"
    if inlier_count < args.min_inliers:
        accepted = False
        reason = "too_few_inliers"
    elif inlier_ratio < args.min_inlier_ratio:
        accepted = False
        reason = "low_inlier_ratio"
    elif translation_m > args.max_step_translation_m:
        accepted = False
        reason = "translation_jump"
    elif rotation_deg > args.max_step_rotation_deg:
        accepted = False
        reason = "rotation_jump"

    return PairResult(
        accepted,
        reason,
        source.paths.index,
        target.paths.index,
        len(matches),
        len(valid_matches),
        inlier_count,
        inlier_ratio,
        translation_m,
        rotation_deg,
        source_from_target,
        np.flatnonzero(inliers),
        valid_matches,
    )


def write_debug_matches(
    source: FrameData,
    target: FrameData,
    result: PairResult,
    debug_dir: Path,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    inlier_set = set(int(index) for index in result.inlier_match_indices)
    matches = [
        match for i, match in enumerate(result.all_matches)
        if not result.inlier_match_indices.size or i in inlier_set
    ]
    image = cv2.drawMatches(
        source.color_bgr,
        list(source.keypoints),
        target.color_bgr,
        list(target.keypoints),
        matches[:120],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    status = "accepted" if result.accepted else f"rejected_{result.reason}"
    out = debug_dir / (
        f"{source.paths.stem}_to_{target.paths.stem}_{status}.jpg"
    )
    cv2.imwrite(str(out), image)


def pair_result_to_dict(result: PairResult) -> dict:
    return {
        "accepted": result.accepted,
        "reason": result.reason,
        "source_index": result.source_index,
        "target_index": result.target_index,
        "raw_matches": result.raw_matches,
        "depth_valid_matches": result.depth_valid_matches,
        "inliers": result.inliers,
        "inlier_ratio": result.inlier_ratio,
        "translation_m": result.translation_m,
        "rotation_deg": result.rotation_deg,
        "source_from_target": result.transform_source_from_target.tolist(),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = read_capture_config(args.dataset)
    args.depth_scale = args.depth_scale or float(config.get("depth_scale", 1000.0))
    args.depth_min_m = (
        args.depth_min_m if args.depth_min_m is not None else float(config.get("depth_min", 0.0))
    )
    args.depth_max_m = (
        args.depth_max_m if args.depth_max_m is not None else float(config.get("depth_max", 1.0))
    )
    if args.depth_max_m <= args.depth_min_m:
        raise ValueError("--depth-max-m must be greater than --depth-min-m")

    intrinsic_path = resolve_intrinsic_path(args, config)
    if not intrinsic_path.exists():
        raise FileNotFoundError(f"Camera intrinsic file not found: {intrinsic_path}")
    intrinsic = o3d.io.read_pinhole_camera_intrinsic(str(intrinsic_path))
    intrinsic_matrix = np.asarray(intrinsic.intrinsic_matrix, dtype=np.float64)

    frame_paths = find_frame_pairs(args.dataset, args.stride, args.max_frames)
    detector = make_detector(args.method, args.max_features)

    print(f"Dataset: {args.dataset}")
    print(f"Intrinsics: {intrinsic_path}")
    print(f"Frames selected: {len(frame_paths)}")
    print(f"Method: {args.method.upper()} + depth RANSAC")

    first_frame = read_frame(
        frame_paths[0],
        detector,
        args.depth_scale,
        np.eye(4, dtype=np.float64),
    )
    accepted_frames = [first_frame]
    pair_results: list[PairResult] = []

    debug_written = 0
    for paths in frame_paths[1:]:
        target = read_frame(
            paths,
            detector,
            args.depth_scale,
            accepted_frames[-1].camera_to_world.copy(),
        )
        result = estimate_pair_transform(
            accepted_frames[-1],
            target,
            args,
            intrinsic_matrix,
        )
        pair_results.append(result)
        if debug_written < args.debug_pairs:
            write_debug_matches(accepted_frames[-1], target, result, args.debug_dir)
            debug_written += 1

        print(
            f"{accepted_frames[-1].paths.stem}->{target.paths.stem} "
            f"{result.reason} matches={result.raw_matches} "
            f"depth={result.depth_valid_matches} inliers={result.inliers} "
            f"ratio={result.inlier_ratio:.2f} "
            f"step={result.translation_m:.4f}m/{result.rotation_deg:.2f}deg"
        )
        if not result.accepted:
            continue

        target.camera_to_world = (
            accepted_frames[-1].camera_to_world @ result.transform_source_from_target
        )
        accepted_frames.append(target)

    if len(accepted_frames) < 2:
        raise RuntimeError("Only the first frame was accepted; markerless tracking failed")

    poses = np.stack([frame.camera_to_world for frame in accepted_frames], axis=0)
    args.poses_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.poses_out, poses)

    selected_stems = [paths.stem for paths in frame_paths]
    accepted_indices = [frame.paths.index for frame in accepted_frames]
    accepted_stems = [frame.paths.stem for frame in accepted_frames]
    report = {
        "dataset": str(args.dataset),
        "intrinsic": str(intrinsic_path),
        "method": args.method,
        "depth_scale": args.depth_scale,
        "depth_min_m": args.depth_min_m,
        "depth_max_m": args.depth_max_m,
        "ransac_threshold_m": args.ransac_threshold_m,
        "selected_frames": len(frame_paths),
        "selected_frame_stems": selected_stems,
        "accepted_frames": len(accepted_frames),
        "accepted_frame_indices": accepted_indices,
        "accepted_frame_stems": accepted_stems,
        "pairs": [pair_result_to_dict(result) for result in pair_results],
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with args.report_out.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    print(f"Accepted frames: {len(accepted_frames)}/{len(frame_paths)}")
    print(f"Poses: {args.poses_out}")
    print(f"Report: {args.report_out}")
    if args.debug_pairs:
        print(f"Debug matches: {args.debug_dir}")


if __name__ == "__main__":
    main()

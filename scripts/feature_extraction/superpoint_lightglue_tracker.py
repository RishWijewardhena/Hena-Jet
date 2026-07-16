#!/usr/bin/env python3
"""Estimate RGB-D poses with SuperPoint + LightGlue + depth RANSAC.

This is the Stage 2 markerless tracker. It uses learned image matching for the
2D correspondence step, then keeps the same geometry test as Stage 1:

1. SuperPoint extracts learned keypoints and descriptors.
2. LightGlue matches keypoints between the last accepted frame and candidate.
3. Matched pixels are lifted to 3D using ZED depth.
4. A rigid transform is estimated with RANSAC.
5. Accepted camera-to-world poses are saved for TSDF fusion.

LightGlue usage follows the official cvg/LightGlue README:
https://github.com/cvg/LightGlue
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d

from rgbd_feature_ransac_tracker import FramePaths


DEFAULT_DATASET = Path("datasets/zed_m_open3d")
DEFAULT_POSES_OUT = Path("outputs/feature_tracking/lightglue_feature_poses.npy")
DEFAULT_REPORT_OUT = Path("outputs/feature_tracking/lightglue_feature_report.json")
DEFAULT_DEBUG_DIR = Path("outputs/feature_tracking/lightglue_debug_matches")
DEFAULT_MODEL_CACHE = Path("outputs/feature_tracking/model_cache")


@dataclass
class LearnedFrame:
    paths: FramePaths
    color_bgr: np.ndarray
    depth_m: np.ndarray
    image_tensor: Any
    features: dict[str, Any]
    camera_to_world: np.ndarray


@dataclass
class LearnedPairResult:
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
    source_pixels: np.ndarray
    target_pixels: np.ndarray
    inlier_mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track an RGB-D sequence with SuperPoint + LightGlue + depth RANSAC."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--intrinsic", type=Path, default=None)
    parser.add_argument("--poses-out", type=Path, default=DEFAULT_POSES_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=DEFAULT_MODEL_CACHE,
        help="Torch model cache folder for downloaded SuperPoint/LightGlue weights.",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--filter-threshold", type=float, default=0.1)
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
    parser.add_argument("--debug-pairs", type=int, default=20)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset does not exist: {args.dataset}")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be zero or positive")
    if args.max_keypoints <= 0:
        raise ValueError("--max-keypoints must be positive")
    if args.filter_threshold < 0:
        raise ValueError("--filter-threshold must be zero or positive")
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


def import_lightglue_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import rbd
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Stage 2 needs PyTorch and LightGlue.\n"
            "Install PyTorch for your Jetson/CUDA setup first, then install LightGlue:\n"
            "  git clone https://github.com/cvg/LightGlue.git\n"
            "  cd LightGlue\n"
            "  python -m pip install -e .\n"
            "Official source: https://github.com/cvg/LightGlue"
        ) from exc
    return torch, LightGlue, SuperPoint, rbd


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


def read_depth_meters(path: Path, depth_scale: float) -> np.ndarray:
    depth_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Could not read depth image: {path}")
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]
    return depth_raw.astype(np.float32) / depth_scale


def image_tensor_from_bgr(color_bgr: np.ndarray, torch: Any, device: str) -> Any:
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(color_rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1).contiguous()
    return tensor.to(device)


def make_models(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, str]:
    args.model_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(args.model_cache_dir))
    torch, LightGlue, SuperPoint, rbd = import_lightglue_dependencies()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

    extractor = SuperPoint(max_num_keypoints=args.max_keypoints).eval().to(device)
    matcher = (
        LightGlue(features="superpoint", filter_threshold=args.filter_threshold)
        .eval()
        .to(device)
    )
    return torch, extractor, matcher, rbd, device


def read_learned_frame(
    paths: FramePaths,
    torch: Any,
    extractor: Any,
    device: str,
    depth_scale: float,
    camera_to_world: np.ndarray,
) -> LearnedFrame:
    color_bgr = cv2.imread(str(paths.color_path), cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise FileNotFoundError(f"Could not read color image: {paths.color_path}")
    depth_m = read_depth_meters(paths.depth_path, depth_scale)
    image_tensor = image_tensor_from_bgr(color_bgr, torch, device)
    with torch.inference_mode():
        features = extractor.extract(image_tensor, resize=None)
    return LearnedFrame(
        paths=paths,
        color_bgr=color_bgr,
        depth_m=depth_m,
        image_tensor=image_tensor,
        features=features,
        camera_to_world=camera_to_world,
    )


def match_learned_features(
    source: LearnedFrame,
    target: LearnedFrame,
    torch: Any,
    matcher: Any,
    rbd: Any,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        matches01 = matcher({"image0": source.features, "image1": target.features})
    feats0, feats1, matches01 = [
        rbd(value) for value in (source.features, target.features, matches01)
    ]
    matches = matches01["matches"]
    keypoints0 = feats0["keypoints"]
    keypoints1 = feats1["keypoints"]
    if matches.numel() == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    source_pixels = keypoints0[matches[:, 0]].detach().cpu().numpy().astype(np.float64)
    target_pixels = keypoints1[matches[:, 1]].detach().cpu().numpy().astype(np.float64)
    return source_pixels, target_pixels


def backproject_pixel(
    pixel: np.ndarray,
    depth_m: np.ndarray,
    intrinsic_matrix: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
) -> np.ndarray | None:
    u_float = float(pixel[0])
    v_float = float(pixel[1])
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
    source: LearnedFrame,
    target: LearnedFrame,
    source_pixels: np.ndarray,
    target_pixels: np.ndarray,
    intrinsic_matrix: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_points = []
    target_points = []
    valid_source_pixels = []
    valid_target_pixels = []
    for source_pixel, target_pixel in zip(source_pixels, target_pixels):
        source_point = backproject_pixel(
            source_pixel,
            source.depth_m,
            intrinsic_matrix,
            depth_min_m,
            depth_max_m,
        )
        target_point = backproject_pixel(
            target_pixel,
            target.depth_m,
            intrinsic_matrix,
            depth_min_m,
            depth_max_m,
        )
        if source_point is None or target_point is None:
            continue
        source_points.append(source_point)
        target_points.append(target_point)
        valid_source_pixels.append(source_pixel)
        valid_target_pixels.append(target_pixel)

    if not source_points:
        return (
            np.empty((0, 3)),
            np.empty((0, 3)),
            np.empty((0, 2)),
            np.empty((0, 2)),
        )
    return (
        np.vstack(source_points),
        np.vstack(target_points),
        np.vstack(valid_source_pixels),
        np.vstack(valid_target_pixels),
    )


def rigid_transform(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
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
    source: LearnedFrame,
    target: LearnedFrame,
    torch: Any,
    matcher: Any,
    rbd: Any,
    args: argparse.Namespace,
    intrinsic_matrix: np.ndarray,
) -> LearnedPairResult:
    source_pixels, target_pixels = match_learned_features(source, target, torch, matcher, rbd)
    source_points, target_points, valid_source_pixels, valid_target_pixels = lift_matches_to_3d(
        source,
        target,
        source_pixels,
        target_pixels,
        intrinsic_matrix,
        args.depth_min_m,
        args.depth_max_m,
    )
    if len(valid_source_pixels) < args.min_depth_matches:
        return LearnedPairResult(
            False,
            "too_few_depth_valid_matches",
            source.paths.index,
            target.paths.index,
            len(source_pixels),
            len(valid_source_pixels),
            0,
            0.0,
            0.0,
            0.0,
            np.eye(4),
            valid_source_pixels,
            valid_target_pixels,
            np.zeros(len(valid_source_pixels), dtype=bool),
        )

    source_from_target, inliers = estimate_rigid_ransac(
        target_points,
        source_points,
        args.ransac_iterations,
        args.ransac_threshold_m,
    )
    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / max(1, len(valid_source_pixels))
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

    return LearnedPairResult(
        accepted,
        reason,
        source.paths.index,
        target.paths.index,
        len(source_pixels),
        len(valid_source_pixels),
        inlier_count,
        inlier_ratio,
        translation_m,
        rotation_deg,
        source_from_target,
        valid_source_pixels,
        valid_target_pixels,
        inliers,
    )


def write_debug_matches(
    source: LearnedFrame,
    target: LearnedFrame,
    result: LearnedPairResult,
    debug_dir: Path,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    source_image = source.color_bgr
    target_image = target.color_bgr
    height = max(source_image.shape[0], target_image.shape[0])
    width = source_image.shape[1] + target_image.shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[: source_image.shape[0], : source_image.shape[1]] = source_image
    canvas[: target_image.shape[0], source_image.shape[1] :] = target_image

    inlier_indices = np.flatnonzero(result.inlier_mask)
    if len(inlier_indices):
        draw_indices = inlier_indices[:120]
    else:
        draw_indices = np.arange(min(120, len(result.source_pixels)))
    for index in draw_indices:
        p0 = result.source_pixels[index]
        p1 = result.target_pixels[index] + np.array([source_image.shape[1], 0.0])
        color = (0, 255, 0) if result.inlier_mask[index] else (0, 0, 255)
        p0_int = (int(round(p0[0])), int(round(p0[1])))
        p1_int = (int(round(p1[0])), int(round(p1[1])))
        cv2.circle(canvas, p0_int, 3, color, -1)
        cv2.circle(canvas, p1_int, 3, color, -1)
        cv2.line(canvas, p0_int, p1_int, color, 1)

    status = "accepted" if result.accepted else f"rejected_{result.reason}"
    out = debug_dir / f"{source.paths.stem}_to_{target.paths.stem}_{status}.jpg"
    cv2.imwrite(str(out), canvas)


def pair_result_to_dict(result: LearnedPairResult) -> dict:
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

    torch, extractor, matcher, rbd, device = make_models(args)
    frame_paths = find_frame_pairs(args.dataset, args.stride, args.max_frames)

    print(f"Dataset: {args.dataset}")
    print(f"Intrinsics: {intrinsic_path}")
    print(f"Frames selected: {len(frame_paths)}")
    print(f"Method: SuperPoint + LightGlue + depth RANSAC")
    print(f"Device: {device}")

    first_frame = read_learned_frame(
        frame_paths[0],
        torch,
        extractor,
        device,
        args.depth_scale,
        np.eye(4, dtype=np.float64),
    )
    accepted_frames = [first_frame]
    pair_results: list[LearnedPairResult] = []

    debug_written = 0
    for paths in frame_paths[1:]:
        target = read_learned_frame(
            paths,
            torch,
            extractor,
            device,
            args.depth_scale,
            accepted_frames[-1].camera_to_world.copy(),
        )
        result = estimate_pair_transform(
            accepted_frames[-1],
            target,
            torch,
            matcher,
            rbd,
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
        raise RuntimeError("Only the first frame was accepted; learned tracking failed")

    poses = np.stack([frame.camera_to_world for frame in accepted_frames], axis=0)
    args.poses_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.poses_out, poses)

    report = {
        "dataset": str(args.dataset),
        "intrinsic": str(intrinsic_path),
        "method": "superpoint_lightglue",
        "device": device,
        "max_keypoints": args.max_keypoints,
        "filter_threshold": args.filter_threshold,
        "depth_scale": args.depth_scale,
        "depth_min_m": args.depth_min_m,
        "depth_max_m": args.depth_max_m,
        "ransac_threshold_m": args.ransac_threshold_m,
        "selected_frames": len(frame_paths),
        "selected_frame_stems": [paths.stem for paths in frame_paths],
        "accepted_frames": len(accepted_frames),
        "accepted_frame_indices": [frame.paths.index for frame in accepted_frames],
        "accepted_frame_stems": [frame.paths.stem for frame in accepted_frames],
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

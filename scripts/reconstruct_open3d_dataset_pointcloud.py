#!/usr/bin/env python3
"""Reconstruct one merged point cloud from an Open3D-style RGB-D dataset."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This script needs Open3D. Install it in the Python environment you use "
        "for reconstruction, for example: python3 -m pip install --user open3d"
    ) from exc


DEFAULT_DATASET = Path("datasets/zed_m_open3d")
DEFAULT_CLOUD_OUT = Path("outputs/zed_m_open3d_merged_cloud.ply")
DEFAULT_MESH_OUT = Path("outputs/zed_m_open3d_merged_mesh.ply")
DEFAULT_POSEGRAPH_OUT = Path("outputs/zed_m_open3d_posegraph.json")
DEFAULT_POSES_OUT = Path("outputs/zed_m_open3d_poses.npy")


@dataclass
class FramePaths:
    index: int
    color_path: Path
    depth_path: Path


@dataclass
class ReconstructionFrame:
    paths: FramePaths
    rgbd: o3d.geometry.RGBDImage
    pose: np.ndarray


@dataclass
class Keyframe:
    """A sparse, cached point cloud used as a loop-closure candidate."""

    frame_index: int
    cloud: o3d.geometry.PointCloud
    feature: o3d.pipelines.registration.Feature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read image/*.png + depth/*.png from a ZED/Open3D capture, estimate "
            "RGB-D odometry, fuse frames with TSDF, and save one merged cloud."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--intrinsic", type=Path, default=None)
    parser.add_argument("--cloud-out", type=Path, default=DEFAULT_CLOUD_OUT)
    parser.add_argument("--mesh-out", type=Path, default=DEFAULT_MESH_OUT)
    parser.add_argument("--posegraph-out", type=Path, default=DEFAULT_POSEGRAPH_OUT)
    parser.add_argument("--poses-out", type=Path, default=DEFAULT_POSES_OUT)
    parser.add_argument("--depth-scale", type=float, default=None)
    parser.add_argument("--depth-max-m", type=float, default=None)
    parser.add_argument("--depth-min-m", type=float, default=None)
    parser.add_argument("--depth-diff-max-m", type=float, default=0.08)
    parser.add_argument("--voxel-length-m", type=float, default=0.004)
    parser.add_argument("--sdf-trunc-m", type=float, default=0.016)
    parser.add_argument("--icp-voxel-m", type=float, default=0.006)
    parser.add_argument("--max-step-translation-m", type=float, default=0.04)
    parser.add_argument("--max-step-rotation-deg", type=float, default=12.0)
    parser.add_argument("--min-valid-depth-px", type=int, default=1000)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--no-posegraph-opt", action="store_true")
    parser.add_argument(
        "--no-loop-closure",
        action="store_true",
        help=(
            "Disable loop-closure search. Without it, poses come from a single "
            "chain of frame-to-frame odometry edges, which has no redundancy for "
            "global_optimization to correct drift with (this is the usual cause "
            "of a shifted/warped result on a scan that orbits an object)."
        ),
    )
    parser.add_argument(
        "--loop-closure-stride",
        type=int,
        default=4,
        help="Cache every Nth accepted frame as a loop-closure candidate keyframe.",
    )
    parser.add_argument(
        "--loop-closure-min-gap",
        type=int,
        default=20,
        help="Minimum frame-index gap before two frames are eligible for loop closure.",
    )
    parser.add_argument(
        "--loop-closure-search-radius-m",
        type=float,
        default=0.25,
        help=(
            "Only attempt loop closure against keyframes whose current estimated "
            "position is within this radius of the new frame. Set this to roughly "
            "the size of the area/object you scanned."
        ),
    )
    parser.add_argument(
        "--loop-closure-voxel-m",
        type=float,
        default=0.012,
        help="Downsample voxel size used for feature-based loop-closure matching.",
    )
    parser.add_argument(
        "--loop-closure-fitness-min",
        type=float,
        default=0.45,
        help="Minimum ICP fitness required to accept a loop-closure edge.",
    )
    parser.add_argument(
        "--loop-closure-max-transform-m",
        type=float,
        default=0.10,
        help=(
            "Reject a loop-closure edge whose ICP relative transform implies "
            "more than this much translation. A good loop closure between two "
            "views of the same area should have a small relative transform."
        ),
    )
    parser.add_argument(
        "--loop-closure-max-rotation-deg",
        type=float,
        default=45.0,
        help=(
            "Reject a loop-closure edge whose ICP relative transform implies "
            "more than this much rotation (degrees)."
        ),
    )
    parser.add_argument(
        "--loop-closure-ransac-fitness-min",
        type=float,
        default=0.15,
        help=(
            "Minimum RANSAC fitness before even attempting ICP refinement. "
            "Saves time and rejects clearly wrong matches early."
        ),
    )
    parser.add_argument(
        "--loop-closure-search-all",
        action="store_true",
        help=(
            "Ignore the spatial search radius and attempt loop closure against "
            "every cached keyframe. More expensive but catches matches that "
            "drift has moved outside the search radius."
        ),
    )
    parser.add_argument(
        "--depth-edge-diff-max-m",
        type=float,
        default=0.02,
        help=(
            "Discard a depth pixel if it differs from any of its 4 neighbors by "
            "more than this much. Removes 'flying pixel' noise at silhouette "
            "edges, which is the classic cause of stereo cameras (ZED, RealSense) "
            "producing spiky/comb-like artifacts in the fused mesh."
        ),
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip final outlier-point and small-mesh-fragment removal.",
    )
    parser.add_argument(
        "--cleanup-outlier-neighbors",
        type=int,
        default=20,
        help="Neighbor count used by statistical outlier removal on the final point cloud.",
    )
    parser.add_argument(
        "--cleanup-outlier-std-ratio",
        type=float,
        default=2.0,
        help="Std-dev ratio used by statistical outlier removal on the final point cloud.",
    )
    parser.add_argument(
        "--cleanup-min-cluster-fraction",
        type=float,
        default=0.02,
        help=(
            "Mesh triangle clusters smaller than this fraction of the largest "
            "cluster are deleted as noise (e.g. floating spikes/shards)."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {args.dataset}")
    if args.depth_scale is not None and args.depth_scale <= 0:
        raise ValueError("--depth-scale must be positive")
    if args.depth_max_m is not None and args.depth_max_m <= 0:
        raise ValueError("--depth-max-m must be positive")
    if args.depth_min_m is not None and args.depth_min_m < 0:
        raise ValueError("--depth-min-m must be zero or positive")
    if (
        args.depth_min_m is not None
        and args.depth_max_m is not None
        and args.depth_max_m <= args.depth_min_m
    ):
        raise ValueError("--depth-max-m must be greater than --depth-min-m")
    if args.depth_diff_max_m <= 0:
        raise ValueError("--depth-diff-max-m must be positive")
    if args.voxel_length_m <= 0:
        raise ValueError("--voxel-length-m must be positive")
    if args.sdf_trunc_m <= args.voxel_length_m:
        raise ValueError("--sdf-trunc-m must be greater than --voxel-length-m")
    if args.icp_voxel_m <= 0:
        raise ValueError("--icp-voxel-m must be positive")
    if args.max_step_translation_m <= 0:
        raise ValueError("--max-step-translation-m must be positive")
    if args.max_step_rotation_deg <= 0:
        raise ValueError("--max-step-rotation-deg must be positive")
    if args.min_valid_depth_px < 0:
        raise ValueError("--min-valid-depth-px must be zero or positive")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be zero or positive")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.loop_closure_stride <= 0:
        raise ValueError("--loop-closure-stride must be positive")
    if args.loop_closure_min_gap <= 0:
        raise ValueError("--loop-closure-min-gap must be positive")
    if args.loop_closure_search_radius_m <= 0:
        raise ValueError("--loop-closure-search-radius-m must be positive")
    if args.loop_closure_voxel_m <= 0:
        raise ValueError("--loop-closure-voxel-m must be positive")
    if not 0.0 <= args.loop_closure_fitness_min <= 1.0:
        raise ValueError("--loop-closure-fitness-min must be between 0 and 1")
    if args.loop_closure_max_transform_m <= 0:
        raise ValueError("--loop-closure-max-transform-m must be positive")
    if args.loop_closure_max_rotation_deg <= 0:
        raise ValueError("--loop-closure-max-rotation-deg must be positive")
    if not 0.0 <= args.loop_closure_ransac_fitness_min <= 1.0:
        raise ValueError("--loop-closure-ransac-fitness-min must be between 0 and 1")
    if args.depth_edge_diff_max_m <= 0:
        raise ValueError("--depth-edge-diff-max-m must be positive")
    if args.cleanup_outlier_neighbors <= 0:
        raise ValueError("--cleanup-outlier-neighbors must be positive")
    if args.cleanup_outlier_std_ratio <= 0:
        raise ValueError("--cleanup-outlier-std-ratio must be positive")
    if not 0.0 < args.cleanup_min_cluster_fraction <= 1.0:
        raise ValueError("--cleanup-min-cluster-fraction must be between 0 and 1")


def read_capture_config(dataset: Path) -> dict:
    config_path = dataset / "capture_config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_frame_pairs(dataset: Path, stride: int, max_frames: int) -> list[FramePaths]:
    image_dir = dataset / "image"
    depth_dir = dataset / "depth"
    if not image_dir.exists() or not depth_dir.exists():
        raise FileNotFoundError(
            f"Expected dataset to contain image/ and depth/: {dataset}"
        )

    color_paths = {path.stem: path for path in sorted(image_dir.glob("*.png"))}
    depth_paths = {path.stem: path for path in sorted(depth_dir.glob("*.png"))}
    common_stems = sorted(set(color_paths) & set(depth_paths))
    if not common_stems:
        raise ValueError(f"No matching image/depth PNG pairs found in {dataset}")

    selected_stems = common_stems[::stride]
    if max_frames:
        selected_stems = selected_stems[:max_frames]

    return [
        FramePaths(index=i, color_path=color_paths[stem], depth_path=depth_paths[stem])
        for i, stem in enumerate(selected_stems)
    ]


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


def make_odometry_options(args: argparse.Namespace) -> o3d.pipelines.odometry.OdometryOption:
    try:
        return o3d.pipelines.odometry.OdometryOption(
            depth_diff_max=args.depth_diff_max_m,
            depth_min=args.depth_min_m or 0.0,
            depth_max=args.depth_max_m,
        )
    except TypeError:
        options = o3d.pipelines.odometry.OdometryOption()
        options.depth_diff_max = args.depth_diff_max_m
        if args.depth_min_m is not None:
            options.depth_min = args.depth_min_m
        if args.depth_max_m is not None:
            options.depth_max = args.depth_max_m
        return options


def read_depth_array(path: Path) -> np.ndarray:
    return np.asarray(o3d.io.read_image(str(path))).copy()


def clean_depth_array(
    depth_raw: np.ndarray,
    depth_scale: float,
    depth_min_m: float,
    depth_max_m: float,
    edge_diff_max_m: float,
) -> np.ndarray:
    """Zero out depth pixels Open3D's own depth_trunc can't catch.

    `RGBDImage.create_from_color_and_depth(..., depth_trunc=depth_max_m)`
    already discards anything *farther* than depth_max_m, but Open3D has no
    equivalent knob for a *minimum* range, so a configured depth_min (e.g.
    from capture_config.json) was previously parsed and validated but never
    actually applied anywhere. Close-range stereo cameras (ZED, RealSense)
    are especially unreliable right at and below their minimum range, and
    at silhouette edges in general ("flying pixels") -- both show up as
    spiky/comb-like noise once fused into a TSDF mesh. This removes both.
    """
    depth_m = depth_raw.astype(np.float32) / depth_scale
    invalid = (depth_m <= 0.0) | (depth_m < depth_min_m) | (depth_m > depth_max_m)

    padded = np.pad(depth_m, 1, mode="edge")
    max_neighbor_diff = np.zeros_like(depth_m)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = padded[1 + dy : 1 + dy + depth_m.shape[0], 1 + dx : 1 + dx + depth_m.shape[1]]
        max_neighbor_diff = np.maximum(max_neighbor_diff, np.abs(depth_m - shifted))
    invalid |= max_neighbor_diff > edge_diff_max_m

    cleaned_m = np.where(invalid, 0.0, depth_m)
    return (cleaned_m * depth_scale).astype(depth_raw.dtype)


def read_rgbd(
    frame: FramePaths,
    depth_scale: float,
    depth_min_m: float,
    depth_max_m: float,
    edge_diff_max_m: float,
) -> o3d.geometry.RGBDImage:
    color = o3d.io.read_image(str(frame.color_path))
    depth_raw = read_depth_array(frame.depth_path)
    cleaned_raw = clean_depth_array(
        depth_raw, depth_scale, depth_min_m, depth_max_m, edge_diff_max_m
    )
    depth = o3d.geometry.Image(cleaned_raw)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=depth_scale,
        depth_trunc=depth_max_m,
        convert_rgb_to_intensity=False,
    )


def count_valid_depth_pixels(rgbd: o3d.geometry.RGBDImage) -> int:
    depth = np.asarray(rgbd.depth)
    return int(np.count_nonzero(np.isfinite(depth) & (depth > 0)))


def pose_delta(prev_pose: np.ndarray, curr_pose: np.ndarray) -> tuple[float, float]:
    relative = np.linalg.inv(prev_pose) @ curr_pose
    translation_m = float(np.linalg.norm(relative[:3, 3]))
    cos_angle = (float(np.trace(relative[:3, :3])) - 1.0) / 2.0
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return translation_m, math.degrees(math.acos(cos_angle))


def make_loop_closure_cloud(
    rgbd: o3d.geometry.RGBDImage,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    voxel_size: float,
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    """Build a downsampled cloud + FPFH feature set for loop-closure matching.

    Uses identity extrinsic on purpose: the cloud stays in the frame's own
    local camera coordinates, matching the convention `compute_rgbd_odometry`
    and `registration_icp` expect (same convention used for `transform` in
    `estimate_poses`).
    """
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    cloud = cloud.voxel_down_sample(voxel_size)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
    )
    return cloud, feature


def attempt_loop_closure(
    candidate_cloud: o3d.geometry.PointCloud,
    candidate_feature: o3d.pipelines.registration.Feature,
    keyframe: Keyframe,
    args: argparse.Namespace,
) -> tuple[bool, np.ndarray, np.ndarray, float]:
    """Try to register `candidate_cloud` (source) onto `keyframe.cloud` (target).

    Returns (success, transform_target_from_source, information, fitness).
    """
    distance_threshold = args.loop_closure_voxel_m * 1.5
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        candidate_cloud,
        keyframe.cloud,
        candidate_feature,
        keyframe.feature,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4_000_000, 500),
    )
    if ransac_result.fitness <= 0.0:
        return False, np.eye(4), np.eye(6), 0.0

    # Phase 1d: RANSAC pre-check — reject clearly bad matches before ICP
    if ransac_result.fitness < args.loop_closure_ransac_fitness_min:
        return False, np.eye(4), np.eye(6), 0.0

    icp_distance = args.loop_closure_voxel_m * 1.4
    icp_result = o3d.pipelines.registration.registration_icp(
        candidate_cloud,
        keyframe.cloud,
        icp_distance,
        ransac_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    if icp_result.fitness < args.loop_closure_fitness_min:
        return False, np.eye(4), np.eye(6), icp_result.fitness

    # Phase 1a: geometric validation — reject loop closures with implausible
    # relative transforms (e.g. 29cm translation + 100° rotation is clearly
    # a false match, not two views of the same area).
    transform = icp_result.transformation
    lc_trans = float(np.linalg.norm(transform[:3, 3]))
    lc_cos = (float(np.trace(transform[:3, :3])) - 1.0) / 2.0
    lc_cos = max(-1.0, min(1.0, lc_cos))
    lc_rot_deg = math.degrees(math.acos(lc_cos))
    if lc_trans > args.loop_closure_max_transform_m:
        return False, np.eye(4), np.eye(6), icp_result.fitness
    if lc_rot_deg > args.loop_closure_max_rotation_deg:
        return False, np.eye(4), np.eye(6), icp_result.fitness

    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        candidate_cloud,
        keyframe.cloud,
        icp_distance,
        icp_result.transformation,
    )
    return True, icp_result.transformation, information, icp_result.fitness


def find_loop_closures(
    frames: list[ReconstructionFrame],
    keyframes: list[Keyframe],
    target_id: int,
    candidate_cloud: o3d.geometry.PointCloud,
    candidate_feature: o3d.pipelines.registration.Feature,
    args: argparse.Namespace,
) -> list[o3d.pipelines.registration.PoseGraphEdge]:
    """Search cached keyframes for ones spatially near the new frame's
    *current* (possibly drifted) estimated position, and try to register
    against those. This is what lets global_optimization actually correct
    drift instead of just re-stating the same single chain of poses.
    """
    edges: list[o3d.pipelines.registration.PoseGraphEdge] = []
    current_translation = frames[target_id].pose[:3, 3]
    for keyframe in keyframes:
        if target_id - keyframe.frame_index < args.loop_closure_min_gap:
            continue
        # Phase 3c: optionally skip the spatial radius check to catch matches
        # that odometry drift has moved outside the search window.
        if not args.loop_closure_search_all:
            keyframe_translation = frames[keyframe.frame_index].pose[:3, 3]
            if (
                np.linalg.norm(current_translation - keyframe_translation)
                > args.loop_closure_search_radius_m
            ):
                continue

        success, transform, information, fitness = attempt_loop_closure(
            candidate_cloud, candidate_feature, keyframe, args
        )
        if not success:
            continue

        edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                keyframe.frame_index,
                target_id,
                np.linalg.inv(transform),
                information,
                uncertain=True,
            )
        )
        print(
            f"  loop-closure {keyframe.frame_index}<->{target_id}  "
            f"fitness={fitness:.3f}"
        )
    return edges


def estimate_poses(
    frame_paths: list[FramePaths],
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    args: argparse.Namespace,
) -> tuple[list[ReconstructionFrame], o3d.pipelines.registration.PoseGraph]:
    if len(frame_paths) < 2:
        raise ValueError("Need at least two RGB-D frames to reconstruct")

    options = make_odometry_options(args)
    jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
    pose_graph = o3d.pipelines.registration.PoseGraph()

    first_rgbd = read_rgbd(
        frame_paths[0],
        args.depth_scale,
        args.depth_min_m,
        args.depth_max_m,
        args.depth_edge_diff_max_m,
    )
    first_pose = np.eye(4, dtype=np.float64)
    frames = [ReconstructionFrame(frame_paths[0], first_rgbd, first_pose)]
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(first_pose))

    keyframes: list[Keyframe] = []
    if not args.no_loop_closure:
        cloud, feature = make_loop_closure_cloud(
            first_rgbd, intrinsic, args.loop_closure_voxel_m
        )
        keyframes.append(Keyframe(0, cloud, feature))

    print("Estimating RGB-D odometry...")
    for candidate_paths in frame_paths[1:]:
        candidate_rgbd = read_rgbd(
            candidate_paths,
            args.depth_scale,
            args.depth_min_m,
            args.depth_max_m,
            args.depth_edge_diff_max_m,
        )
        valid_depth = count_valid_depth_pixels(candidate_rgbd)
        if valid_depth < args.min_valid_depth_px:
            print(
                f"  skip {candidate_paths.color_path.name}: "
                f"only {valid_depth} valid depth pixels"
            )
            continue

        previous = frames[-1]
        success, transform, information = o3d.pipelines.odometry.compute_rgbd_odometry(
            previous.rgbd,
            candidate_rgbd,
            intrinsic,
            np.eye(4, dtype=np.float64),
            jacobian,
            options,
        )
        if not success:
            print(f"  skip {candidate_paths.color_path.name}: odometry failed")
            continue

        pose = previous.pose @ np.linalg.inv(transform)
        move_m, rot_deg = pose_delta(previous.pose, pose)
        if (
            move_m > args.max_step_translation_m
            or rot_deg > args.max_step_rotation_deg
        ):
            print(
                f"  skip {candidate_paths.color_path.name}: "
                f"jump {move_m:.4f}m / {rot_deg:.2f}deg"
            )
            continue

        source_id = len(frames) - 1
        target_id = len(frames)
        frames.append(ReconstructionFrame(candidate_paths, candidate_rgbd, pose))
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose))
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source_id,
                target_id,
                transform,
                information,
                uncertain=False,
            )
        )
        print(
            f"  edge {source_id}->{target_id}  "
            f"{candidate_paths.color_path.name}  "
            f"move={move_m:.4f}m  rot={rot_deg:.2f}deg"
        )

        if not args.no_loop_closure and target_id % args.loop_closure_stride == 0:
            candidate_cloud, candidate_feature = make_loop_closure_cloud(
                candidate_rgbd, intrinsic, args.loop_closure_voxel_m
            )
            pose_graph.edges.extend(
                find_loop_closures(
                    frames, keyframes, target_id, candidate_cloud, candidate_feature, args
                )
            )
            keyframes.append(Keyframe(target_id, candidate_cloud, candidate_feature))

    # Phase 3b: force a loop-closure attempt on the very last frame even if it
    # didn't land on a stride boundary — closing the trajectory is critical.
    if not args.no_loop_closure and len(frames) >= 2:
        last_id = len(frames) - 1
        if last_id % args.loop_closure_stride != 0:  # wasn't already handled
            last_frame = frames[last_id]
            last_cloud, last_feature = make_loop_closure_cloud(
                last_frame.rgbd, intrinsic, args.loop_closure_voxel_m
            )
            pose_graph.edges.extend(
                find_loop_closures(
                    frames, keyframes, last_id, last_cloud, last_feature, args
                )
            )

    return frames, pose_graph


def optimize_pose_graph(
    pose_graph: o3d.pipelines.registration.PoseGraph,
    args: argparse.Namespace,
) -> None:
    if args.no_posegraph_opt or len(pose_graph.nodes) < 2:
        return
    print("Optimizing pose graph...")
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=args.icp_voxel_m * 8.0,
            edge_prune_threshold=0.25,
            reference_node=0,
        ),
    )


def clean_point_cloud(
    cloud: o3d.geometry.PointCloud, nb_neighbors: int, std_ratio: float
) -> o3d.geometry.PointCloud:
    if len(cloud.points) == 0:
        return cloud
    cleaned, _ = cloud.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    return cleaned


def clean_mesh(
    mesh: o3d.geometry.TriangleMesh, min_cluster_fraction: float
) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) == 0:
        return mesh
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    keep_threshold = max(1, int(cluster_n_triangles.max() * min_cluster_fraction))
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < keep_threshold
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    return mesh


def make_tsdf_volume(args: argparse.Namespace) -> o3d.pipelines.integration.ScalableTSDFVolume:
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_length_m,
        sdf_trunc=args.sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )


def integrate_and_save(
    frames: list[ReconstructionFrame],
    pose_graph: o3d.pipelines.registration.PoseGraph,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    args: argparse.Namespace,
) -> None:
    print("Integrating frames into TSDF...")
    volume = make_tsdf_volume(args)
    for i, (frame, node) in enumerate(zip(frames, pose_graph.nodes), start=1):
        frame.pose = np.asarray(node.pose, dtype=np.float64)
        volume.integrate(frame.rgbd, intrinsic, np.linalg.inv(frame.pose))
        print(f"  integrated {i}/{len(frames)}", end="\r")
    print()

    mesh = volume.extract_triangle_mesh()
    if not args.no_cleanup:
        print("Removing small disconnected mesh fragments...")
        before = len(mesh.triangles)
        mesh = clean_mesh(mesh, args.cleanup_min_cluster_fraction)
        print(f"  triangles {before} -> {len(mesh.triangles)}")
    mesh.compute_vertex_normals()
    args.mesh_out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(args.mesh_out), mesh)
    print(f"Mesh: {args.mesh_out} ({len(mesh.vertices)} vertices)")

    cloud = volume.extract_point_cloud()
    if not args.no_cleanup:
        print("Removing statistical outlier points...")
        before = len(cloud.points)
        cloud = clean_point_cloud(
            cloud, args.cleanup_outlier_neighbors, args.cleanup_outlier_std_ratio
        )
        print(f"  points {before} -> {len(cloud.points)}")
    args.cloud_out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(args.cloud_out), cloud)
    print(f"Merged point cloud: {args.cloud_out} ({len(cloud.points)} points)")

    args.posegraph_out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_pose_graph(str(args.posegraph_out), pose_graph)
    print(f"Pose graph: {args.posegraph_out}")

    poses = np.stack([np.asarray(node.pose, dtype=np.float64) for node in pose_graph.nodes])
    args.poses_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.poses_out, poses)
    print(f"Frame poses: {args.poses_out}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = read_capture_config(args.dataset)
    args.depth_scale = args.depth_scale or float(config.get("depth_scale", 1000.0))
    args.depth_max_m = args.depth_max_m or float(config.get("depth_max", 1.0))
    args.depth_min_m = (
        args.depth_min_m if args.depth_min_m is not None else float(config.get("depth_min", 0.0))
    )

    intrinsic_path = resolve_intrinsic_path(args, config)
    if not intrinsic_path.exists():
        raise FileNotFoundError(f"Camera intrinsic file not found: {intrinsic_path}")

    intrinsic = o3d.io.read_pinhole_camera_intrinsic(str(intrinsic_path))
    frame_paths = find_frame_pairs(args.dataset, args.stride, args.max_frames)
    print(f"Dataset: {args.dataset}")
    print(f"Intrinsics: {intrinsic_path}")
    print(f"RGB-D pairs selected: {len(frame_paths)}")
    print(
        f"Depth scale={args.depth_scale:g}, "
        f"depth range={args.depth_min_m:g}m..{args.depth_max_m:g}m"
    )

    frames, pose_graph = estimate_poses(frame_paths, intrinsic, args)
    if len(frames) < 2:
        raise RuntimeError("Not enough odometry-linked frames to reconstruct")

    odometry_edges = sum(1 for e in pose_graph.edges if not e.uncertain)
    loop_edges = sum(1 for e in pose_graph.edges if e.uncertain)
    print(
        f"Using {len(frames)} frames, {odometry_edges} odometry edges, "
        f"{loop_edges} loop-closure edges"
    )
    optimize_pose_graph(pose_graph, args)
    integrate_and_save(frames, pose_graph, intrinsic, args)


if __name__ == "__main__":
    main()
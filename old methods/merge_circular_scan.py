#!/usr/bin/env python3
"""Merge ZED point cloud captures from a fixed-object, orbiting-camera scan."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the orbiting-camera merge.

    --capture-dir: Folder containing angle_*.json and matching angle_*.npz files.
    --out: Output PLY path for the merged colored point cloud.
    --voxel-m: Final output voxel size in meters. Use 0 to disable downsampling.
    --max-points-per-frame: Safety limit for very dense captures.
    --max-depth-over-radius-m: Camera-local depth crop. For example, if the
        scanner radius is 0.20 and this value is 0.05, points deeper than
        0.25 m from the camera are removed before merging.
    --min-world-z and --max-world-z: Optional height crop after transforming
        each frame into scanner/world coordinates.
    --max-world-radius-m: Optional XY radius crop around the scanner center.
    --icp: Optionally refine each new frame against the accumulated cloud with
        guarded Open3D point-to-plane ICP.
    --icp-* options: Control ICP downsampling, matching distance, iteration
        count, and safety gates for accepting only small corrections.
    --rig-pivot: Where the physical rig rotates. "left" means the turntable axis
        passes through the left optical center (default). "mid" means it passes
        through the midpoint between the two lenses. "right" means it passes
        through the right optical center. This setting is used together with the
        baseline_m stored in each capture's JSON to shift points into world
        space correctly before rotation is applied.
    """
    parser = argparse.ArgumentParser(
        description="Merge ZED captures for a fixed object with camera orbiting around Z."
    )
    parser.add_argument("--capture-dir", type=Path, default=Path("captures/zed_m_first_scan"))
    parser.add_argument("--out", type=Path, default=Path("outputs/zed_m_merged_scan.ply"))
    parser.add_argument("--voxel-m", type=float, default=0.002)
    parser.add_argument("--max-points-per-frame", type=int, default=250_000)
    parser.add_argument("--max-depth-over-radius-m", type=float, default=0.05)
    parser.add_argument("--min-world-z", type=float, default=None)
    parser.add_argument("--max-world-z", type=float, default=None)
    parser.add_argument("--max-world-radius-m", type=float, default=None)
    parser.add_argument("--icp", action="store_true")
    parser.add_argument("--icp-voxel-m", type=float, default=0.005)
    parser.add_argument("--icp-distance-m", type=float, default=0.02)
    parser.add_argument("--icp-iterations", type=int, default=50)
    parser.add_argument("--icp-min-fitness", type=float, default=0.65)
    parser.add_argument("--icp-max-translation-m", type=float, default=0.02)
    parser.add_argument("--icp-max-rotation-deg", type=float, default=8.0)
    parser.add_argument(
        "--rig-pivot",
        choices=["left", "mid", "right"],
        default="left",
        help=(
            "Which lens center the physical rig rotates around. "
            "'left' (default) requires no baseline correction because all ZED "
            "point clouds are already expressed in the left-lens frame. "
            "'mid' shifts points by +half_baseline along the camera X axis before "
            "applying the rotation. 'right' shifts by +full_baseline."
        ),
    )
    return parser.parse_args()


def build_camera_axes(angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the four world-space unit vectors that describe camera orientation.

    The camera sits at angle_deg around the scanner Z axis, pointing inward
    toward the origin. Returns (forward, right, left, up) as float32 arrays.

    FIX: The original code computed left = cross(up, forward) which actually
    produces the RIGHT vector under the right-hand rule, then set right = -left,
    flipping the Y axis for all RIGHT_HANDED_Z_UP_X_FWD captures. The correct
    derivation is:
        right = cross(forward, up)   [standard right-hand rule for a camera]
        left  = -right
    """
    theta = math.radians(angle_deg)
    radial_out = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=np.float64)
    forward = -radial_out                                        # camera looks inward
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    # FIX: was cross(up, forward) which gives the left vector, not right
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    left = -right
    return (
        forward.astype(np.float32),
        right.astype(np.float32),
        left.astype(np.float32),
        up.astype(np.float32),
    )


def baseline_shift_for_pivot(
    rig_pivot: str,
    baseline_m: float,
    coordinate_system: str,
) -> np.ndarray:
    """Return the per-point offset (meters) to compensate for rig pivot position.

    All ZED point clouds are expressed in the LEFT lens optical frame. If the
    physical rig does not rotate around the left lens, every point must be
    shifted before the world rotation is applied so that the effective pivot
    matches the real rotation axis.

    rig_pivot == "left":  no shift needed (SDK frame already matches pivot)
    rig_pivot == "mid":   shift points +half_baseline along camera +X
    rig_pivot == "right": shift points +full_baseline along camera +X

    The camera +X axis direction depends on the ZED coordinate system:
        IMAGE:                  +X is right  → shift along local X
        RIGHT_HANDED_Z_UP_X_FWD: +Y is left → shift along local -Y (i.e. right)

    Returns a shape-(3,) float32 offset in CAMERA space. The caller adds this
    to every point before applying the world rotation.
    """
    if rig_pivot == "left" or baseline_m <= 0.0:
        return np.zeros(3, dtype=np.float32)

    shift_magnitude = baseline_m / 2.0 if rig_pivot == "mid" else baseline_m

    if coordinate_system == "IMAGE":
        # camera +X is rightward along the baseline; pivot is to the right of
        # the left lens, so we shift points in the +X direction
        return np.array([shift_magnitude, 0.0, 0.0], dtype=np.float32)

    if coordinate_system == "RIGHT_HANDED_Z_UP_X_FWD":
        # camera +Y is leftward; right is -Y; shift points toward right (+Y sign flip)
        return np.array([0.0, -shift_magnitude, 0.0], dtype=np.float32)

    raise ValueError(f"Unsupported coordinate_system: {coordinate_system}")


def transform_camera_to_world(
    points: np.ndarray,
    angle_deg: float,
    radius_m: float,
    height_m: float,
    coordinate_system: str,
    rig_pivot: str,
    baseline_m: float,
) -> np.ndarray:
    """Transform camera-local ZED points into scanner/world coordinates.

    The object stays fixed at the scanner center while the ZED camera moves
    around it on a circular path.

    FIX 1: Corrected right/left cross-product direction (see build_camera_axes).
    FIX 2: Added baseline compensation so the effective rotation center matches
           the physical rig pivot regardless of which lens it turns around.

    Supported ZED capture coordinate systems:
        IMAGE:                  +X right, +Y down, +Z forward
        RIGHT_HANDED_Z_UP_X_FWD: +X forward, +Y left, +Z up

    Parameters
    ----------
    points:
        (N, 3) float32 array in camera-local coordinates.
    angle_deg:
        Camera position angle around the scanner center, in degrees.
    radius_m:
        Distance from the scanner center to the rig pivot point, in meters.
        Must be consistent with rig_pivot: if rig_pivot is "left" this is the
        distance to the left lens; if "mid" it is to the midpoint, etc.
    height_m:
        Camera height offset from the world reference plane.
    coordinate_system:
        ZED SDK coordinate system used during capture.
    rig_pivot:
        Which lens center the physical rig rotates around ("left", "mid", "right").
    baseline_m:
        Left-to-right lens separation in meters, from calibration. Pass 0.0
        if not available (disables baseline compensation).
    """
    forward, right, left, up = build_camera_axes(angle_deg)
    down = -up

    # Camera pivot position in world space
    theta = math.radians(angle_deg)
    camera_position = np.array(
        [radius_m * math.cos(theta), radius_m * math.sin(theta), height_m],
        dtype=np.float32,
    )

    # FIX: shift points from left-lens frame to pivot frame before rotation
    pivot_offset = baseline_shift_for_pivot(rig_pivot, baseline_m, coordinate_system)
    shifted = points + pivot_offset  # broadcast over N points

    if coordinate_system == "IMAGE":
        return (
            camera_position
            + shifted[:, 0:1] * right
            + shifted[:, 1:2] * down
            + shifted[:, 2:3] * forward
        )
    if coordinate_system == "RIGHT_HANDED_Z_UP_X_FWD":
        return (
            camera_position
            + shifted[:, 0:1] * forward
            + shifted[:, 1:2] * left
            + shifted[:, 2:3] * up
        )
    raise ValueError(f"Unsupported coordinate_system in capture metadata: {coordinate_system}")


def camera_forward_depth(points: np.ndarray, coordinate_system: str) -> np.ndarray:
    """Return camera-forward depth for the selected ZED capture coordinate system."""
    if coordinate_system == "IMAGE":
        return points[:, 2]
    if coordinate_system == "RIGHT_HANDED_Z_UP_X_FWD":
        return points[:, 0]
    raise ValueError(f"Unsupported coordinate_system in capture metadata: {coordinate_system}")


def crop_world_region(
    points: np.ndarray,
    colors: np.ndarray,
    min_world_z: float | None,
    max_world_z: float | None,
    max_world_radius_m: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop transformed world points to the scanner region containing the object.

    min_world_z removes table/base points below the object. max_world_z removes
    high background clutter. max_world_radius_m removes side/background points
    that are too far from the scanner center in the XY plane.
    """
    keep = np.ones(points.shape[0], dtype=bool)
    if min_world_z is not None:
        keep &= points[:, 2] >= min_world_z
    if max_world_z is not None:
        keep &= points[:, 2] <= max_world_z
    if max_world_radius_m is not None:
        keep &= np.linalg.norm(points[:, :2], axis=1) <= max_world_radius_m
    return points[keep], colors[keep]


def voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a point cloud by keeping one point per voxel grid cell."""
    if voxel_m <= 0:
        return points, colors
    keys = np.floor(points / voxel_m).astype(np.int64)
    _, unique_indices = np.unique(keys, axis=0, return_index=True)
    unique_indices.sort()
    return points[unique_indices], colors[unique_indices]


def random_subsample(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly subsample a point cloud to at most max_points points.

    FIX: The original code used a regular stride (points[::stride]) which
    samples in image row order and produces horizontal banding artifacts.
    Random sampling gives uniform spatial coverage.
    """
    if points.shape[0] <= max_points:
        return points, colors
    idx = np.random.choice(points.shape[0], max_points, replace=False)
    idx.sort()
    return points[idx], colors[idx]


def make_o3d_cloud(o3d, points: np.ndarray, colors: np.ndarray, voxel_m: float):
    """Build an Open3D cloud with estimated normals for point-to-plane ICP."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    if voxel_m > 0:
        cloud = cloud.voxel_down_sample(voxel_m)
    normal_radius = max(voxel_m * 3.0, 0.01)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )
    return cloud


def refine_with_point_to_plane_icp(
    o3d,
    source_points: np.ndarray,
    source_colors: np.ndarray,
    target_points: np.ndarray,
    target_colors: np.ndarray,
    voxel_m: float,
    distance_m: float,
    iterations: int,
):
    """Refine one mechanically placed frame against the accumulated cloud.

    ICP starts from identity because the angle/radius/height transform has
    already placed source_points in world coordinates. The returned points are
    the full-resolution source points after the ICP correction.
    """
    source = make_o3d_cloud(o3d, source_points, source_colors, voxel_m)
    target = make_o3d_cloud(o3d, target_points, target_colors, voxel_m)
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_m,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations),
    )
    rotation = result.transformation[:3, :3]
    translation = result.transformation[:3, 3]
    transformed_points = source_points.astype(np.float64) @ rotation.T + translation
    return transformed_points.astype(np.float32), result


def icp_transform_summary(transformation: np.ndarray) -> tuple[float, float]:
    """Return ICP correction size as translation meters and rotation degrees."""
    translation_m = float(np.linalg.norm(transformation[:3, 3]))
    rotation = transformation[:3, :3]
    cos_angle = (float(np.trace(rotation)) - 1.0) / 2.0
    cos_angle = max(-1.0, min(1.0, cos_angle))
    rotation_deg = math.degrees(math.acos(cos_angle))
    return translation_m, rotation_deg


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write colored XYZ points to a binary little-endian PLY file.

    FIX: The original ASCII PLY writer looped over every point in Python, which
    is extremely slow for large clouds (9M+ iterations for a full 36-frame scan).
    Binary PLY writes the entire arrays at once via numpy, which is ~100x faster
    and produces a smaller file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = points.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    # Pack each vertex as: float32 x, float32 y, float32 z, uint8 r, uint8 g, uint8 b
    # Structured array lets numpy write this in one call with no Python loop.
    dtype = np.dtype([
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("r", np.uint8),
        ("g", np.uint8),
        ("b", np.uint8),
    ])
    records = np.empty(n, dtype=dtype)
    records["x"] = points[:, 0]
    records["y"] = points[:, 1]
    records["z"] = points[:, 2]
    records["r"] = colors[:, 0]
    records["g"] = colors[:, 1]
    records["b"] = colors[:, 2]

    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(records.tobytes())


def load_capture(meta_path: Path) -> tuple[dict, np.ndarray, np.ndarray] | None:
    """Load one capture metadata file and its matching compressed point arrays.

    FIX: Added explicit check for required metadata keys so missing fields give
    a clear message instead of a bare KeyError.
    """
    npz_path = meta_path.with_suffix(".npz")
    if not npz_path.exists():
        print(f"Skipping {meta_path.name}: missing {npz_path.name}")
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    required_keys = ("angle_deg", "radius_m", "height_m")
    for key in required_keys:
        if key not in meta:
            print(f"Skipping {meta_path.name}: missing required key '{key}' in metadata")
            return None

    data = np.load(npz_path)
    if "points" not in data or "colors" not in data:
        print(f"Skipping {meta_path.name}: .npz missing 'points' or 'colors' array")
        return None

    return meta, data["points"], data["colors"]


def build_icp_target(
    frames: list[tuple[np.ndarray, np.ndarray]],
    icp_voxel_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a downsampled ICP target from all accumulated frames.

    FIX: The original code concatenated all raw frames for each ICP call, making
    the target grow to O(N_frames * N_points) — very slow and memory-heavy for
    late frames. This helper voxel-downsamples the accumulated cloud at a coarser
    resolution before passing it to ICP, keeping target size bounded.
    """
    all_pts = np.concatenate([f[0] for f in frames], axis=0)
    all_col = np.concatenate([f[1] for f in frames], axis=0)
    # Use 2× the ICP voxel size for a coarser but still representative target
    return voxel_downsample(all_pts, all_col, icp_voxel_m * 2.0)


def main() -> None:
    """Merge fixed-object, orbiting-camera captures into one colored PLY."""
    args = parse_args()

    o3d = None
    if args.icp:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError(
                "Open3D is required for --icp. Install it with: python -m pip install open3d"
            ) from exc

    frames: list[tuple[np.ndarray, np.ndarray]] = []

    for meta_path in sorted(args.capture_dir.glob("angle_*.json")):
        loaded = load_capture(meta_path)
        if loaded is None:
            continue

        meta, points, colors = loaded
        radius_m = float(meta["radius_m"])
        angle_deg = float(meta["angle_deg"])
        height_m = float(meta["height_m"])

        # FIX: default was "RIGHT_HANDED_Z_UP_X_FWD" but capture script defaults
        # to "IMAGE". An old capture file without this key would be transformed
        # using the wrong coordinate system, silently corrupting every point.
        coordinate_system = meta.get("coordinate_system", "IMAGE")

        # baseline_m may be absent in captures made before it was recorded;
        # fall back to 0.0 which disables baseline compensation gracefully.
        baseline_m = float(meta.get("baseline_m", 0.0))
        if baseline_m <= 0.0 and args.rig_pivot != "left":
            print(
                f"Warning {meta_path.name}: rig_pivot='{args.rig_pivot}' but "
                f"baseline_m is not in metadata — baseline compensation skipped."
            )

        # ── depth crop in camera space ─────────────────────────────────────
        if args.max_depth_over_radius_m > 0:
            max_camera_depth_m = radius_m + args.max_depth_over_radius_m
            keep_depth = camera_forward_depth(points, coordinate_system) <= max_camera_depth_m
            points = points[keep_depth]
            colors = colors[keep_depth]

        # FIX: random subsample instead of stride to avoid spatial banding
        points, colors = random_subsample(points, colors, args.max_points_per_frame)

        if points.shape[0] == 0:
            print(f"Skipping {meta_path.name}: no points after depth crop")
            continue

        # ── transform to world space ───────────────────────────────────────
        world_points = transform_camera_to_world(
            points,
            angle_deg=angle_deg,
            radius_m=radius_m,
            height_m=height_m,
            coordinate_system=coordinate_system,
            rig_pivot=args.rig_pivot,
            baseline_m=baseline_m,
        ).astype(np.float32)
        colors = colors.astype(np.uint8)

        # ── world region crop ─────────────────────────────────────────────
        before_crop = world_points.shape[0]
        world_points, colors = crop_world_region(
            world_points,
            colors,
            min_world_z=args.min_world_z,
            max_world_z=args.max_world_z,
            max_world_radius_m=args.max_world_radius_m,
        )
        if world_points.shape[0] == 0:
            print(f"Skipping {meta_path.name}: world crop removed all points")
            continue
        if world_points.shape[0] != before_crop:
            print(
                f"World crop {meta_path.name}: "
                f"{before_crop} → {world_points.shape[0]} points"
            )

        # ── optional ICP refinement ───────────────────────────────────────
        if args.icp and frames:
            # FIX: build a bounded downsampled target instead of raw concatenation
            target_points, target_colors = build_icp_target(frames, args.icp_voxel_m)

            icp_points, icp_result = refine_with_point_to_plane_icp(
                o3d,
                world_points,
                colors,
                target_points,
                target_colors,
                voxel_m=args.icp_voxel_m,
                distance_m=args.icp_distance_m,
                iterations=args.icp_iterations,
            )
            icp_translation_m, icp_rotation_deg = icp_transform_summary(
                icp_result.transformation
            )
            icp_ok = (
                icp_result.fitness >= args.icp_min_fitness
                and icp_translation_m <= args.icp_max_translation_m
                and icp_rotation_deg <= args.icp_max_rotation_deg
            )
            status = "accepted" if icp_ok else "rejected"
            print(
                f"ICP {meta_path.name}: {status}, "
                f"fitness={icp_result.fitness:.3f}, "
                f"rmse={icp_result.inlier_rmse:.4f}, "
                f"move={icp_translation_m:.4f}m, "
                f"rot={icp_rotation_deg:.2f}deg"
            )
            if icp_ok:
                world_points = icp_points

        frames.append((world_points, colors))
        print(f"Loaded {meta_path.name}: {world_points.shape[0]} points at angle {angle_deg:g}°")

    if not frames:
        raise RuntimeError(f"No valid captures found in {args.capture_dir}")

    # ── merge and downsample ──────────────────────────────────────────────
    merged_points = np.concatenate([f[0] for f in frames], axis=0)
    merged_colors = np.concatenate([f[1] for f in frames], axis=0)
    merged_points, merged_colors = voxel_downsample(
        merged_points,
        merged_colors,
        args.voxel_m,
    )

    # FIX: binary PLY is ~100x faster and smaller than ASCII for large clouds
    write_binary_ply(args.out, merged_points, merged_colors)
    print(f"Saved merged cloud: {merged_points.shape[0]} points → {args.out}")


if __name__ == "__main__":
    main()
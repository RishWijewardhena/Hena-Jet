"""Volumetric TSDF fusion of registered captures.

Merging per-view clouds concatenates every view's noise: a surface seen from
37 angles becomes 37 slightly different surfaces stacked on top of each other.
TSDF integration instead averages the views into one signed-distance field, so
independent per-view error cancels rather than accumulating, and the extracted
surface is a single sheet.

This consumes the poses the reconstruction has already solved; it does not
change registration. Averaging only cancels *random* error - a systematic scale
or calibration error is present in every view and survives fusion unchanged.

Depth comes from the raw ``rgbd/<stem>/`` capture when the scan recorded one,
and otherwise from reprojecting ``frame_*.ply``. That reprojection is exact:
the PLY holds a pinhole back-projection of pixel centres, so projecting it back
lands on integer pixels and recovers the original depth bit-for-bit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Voxel and truncation defaults, chosen by sweep on a 1280x800 close-range scan.
# sdf_trunc sets how far the field is carried either side of a surface, so the
# extracted shell is about twice that thick; it must stay several voxels wide
# for the field to interpolate, and small relative to real surface separation.
DEFAULT_VOXEL_M = 0.001
DEFAULT_SDF_TRUNC_M = 0.003
MAX_REPAIR_HOLE_M = 0.003
TAUBIN_SMOOTHING_ITERATIONS = 3
MIN_COMPONENT_TRIANGLES = 32
MIN_COMPONENT_FRACTION = 0.0005


@dataclass
class FusionFrame:
    """One registered capture as TSDF integration needs it."""

    depth_m: np.ndarray
    color_rgb: np.ndarray
    pose: np.ndarray


def _require_intrinsics(intrinsics: dict) -> tuple[int, int, float, float, float, float]:
    missing = [k for k in ("width", "height", "fx", "fy", "cx", "cy") if k not in intrinsics]
    if missing:
        raise ValueError(f"intrinsics.json is missing required keys: {', '.join(missing)}")
    width, height = int(intrinsics["width"]), int(intrinsics["height"])
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    if width <= 0 or height <= 0 or fx <= 0 or fy <= 0:
        raise ValueError("Intrinsics must have positive dimensions and focal lengths")
    return width, height, fx, fy, cx, cy


def depth_from_ply(ply_path: Path, intrinsics: dict) -> tuple[np.ndarray, np.ndarray]:
    """Recover the depth and colour images a camera-frame PLY was built from.

    ``pointcloud_export.backproject_to_points`` projects pixel centres through a
    pinhole model with no resampling, so the inverse is exact. Points must be in
    the camera frame: passing an already-transformed cloud silently produces
    meaningless pixel coordinates, so this rejects anything that misses the grid.
    """
    import open3d as o3d

    width, height, fx, fy, cx, cy = _require_intrinsics(intrinsics)
    cloud = o3d.io.read_point_cloud(str(ply_path))
    if cloud.is_empty():
        raise RuntimeError(f"Open3D could not read points from {ply_path}")
    points = np.asarray(cloud.points)
    z = points[:, 2]
    if not np.all(z > 0):
        raise ValueError(f"{ply_path.name} contains non-positive depths; not a camera-frame cloud")

    u = points[:, 0] * fx / z + cx
    v = points[:, 1] * fy / z + cy
    ui, vi = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
    offgrid = max(np.abs(u - ui).max(), np.abs(v - vi).max())
    if offgrid > 1e-6:
        raise ValueError(
            f"{ply_path.name} does not reproject onto the pixel grid "
            f"(max offset {offgrid:.3g} px); it is not a camera-frame cloud "
            "for these intrinsics"
        )
    if not ((ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)).all():
        raise ValueError(f"{ply_path.name} reprojects outside {width}x{height}")

    depth = np.zeros((height, width), dtype=np.float32)
    color = np.zeros((height, width, 3), dtype=np.uint8)
    # Nearer samples win, so a duplicated pixel keeps the visible surface.
    order = np.argsort(-z)
    depth[vi[order], ui[order]] = z[order].astype(np.float32)
    if cloud.has_colors():
        rgb = (np.clip(np.asarray(cloud.colors), 0.0, 1.0) * 255.0).astype(np.uint8)
        color[vi[order], ui[order]] = rgb[order]
    return depth, color


def depth_from_rgbd(rgbd_dir: Path, depth_source: str) -> tuple[np.ndarray, np.ndarray]:
    """Read the ungated depth and colour a capture saved alongside its PLY."""
    import cv2

    name = "sensor_depth_00.npy" if depth_source == "sensor" else "depth_output_full.npy"
    depth_path = rgbd_dir / name
    if not depth_path.is_file():
        raise FileNotFoundError(f"No {name} in {rgbd_dir}")
    depth = np.load(depth_path).astype(np.float32)
    bgr = cv2.imread(str(rgbd_dir / "rgb_00.png"), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot read rgb_00.png in {rgbd_dir}")
    color = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if color.shape[:2] != depth.shape:
        raise RuntimeError(f"RGB and depth dimensions differ in {rgbd_dir}")
    return depth, color


def load_fusion_frame(
    ply_path: Path,
    pose: np.ndarray,
    intrinsics: dict,
    *,
    input_dir: Path,
    depth_source: str = "sensor",
    prefer_rgbd: bool = True,
) -> FusionFrame:
    """Build one integration frame, preferring ungated raw depth when saved."""
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"Invalid 4x4 pose for {ply_path.name}")

    rgbd_dir = input_dir / "rgbd" / ply_path.stem
    if prefer_rgbd and rgbd_dir.is_dir():
        depth, color = depth_from_rgbd(rgbd_dir, depth_source)
    else:
        depth, color = depth_from_ply(ply_path, intrinsics)
    return FusionFrame(depth_m=depth, color_rgb=color, pose=pose)



def _mask_depth_to_crop(depth, pose, intrinsics, crop_bounds):
    """Zero depth whose points fall outside the crop, in the reference frame.

    ``ScalableTSDFVolume`` allocates wherever it sees a surface, so integrating
    the full frustum builds the whole room at the working voxel size. Masking
    first bounds memory and makes the fused surface match the region the merge
    path keeps.
    """
    from pointcloud_processing import CylinderCrop

    if crop_bounds is None:
        return depth
    width, height, fx, fy, cx, cy = _require_intrinsics(intrinsics)
    valid = depth > 0
    if not np.any(valid):
        return depth
    rows, cols = np.nonzero(valid)
    z = depth[rows, cols].astype(np.float64)
    camera = np.column_stack(((cols - cx) / fx * z, (rows - cy) / fy * z, z))
    reference = camera @ np.asarray(pose, float)[:3, :3].T + np.asarray(pose, float)[:3, 3]

    if isinstance(crop_bounds, CylinderCrop):
        keep = crop_bounds.mask(reference)
    else:
        bounds = np.asarray(crop_bounds, dtype=float)
        if bounds.shape != (6,) or not np.isfinite(bounds).all():
            raise ValueError("crop_bounds must contain six finite values")
        keep = np.all((reference >= bounds[:3]) & (reference <= bounds[3:]), axis=1)

    masked = np.zeros_like(depth)
    masked[rows[keep], cols[keep]] = depth[rows[keep], cols[keep]]
    return masked


def integrate(
    frames: Sequence[FusionFrame],
    intrinsics: dict,
    *,
    voxel_length_m: float = DEFAULT_VOXEL_M,
    sdf_trunc_m: float = DEFAULT_SDF_TRUNC_M,
    depth_min_m: float = 0.0,
    depth_max_m: float = 1.0,
    crop_bounds=None,
):
    """Average the registered views into one signed-distance field."""
    import open3d as o3d

    if not frames:
        raise ValueError("TSDF fusion needs at least one frame")
    if voxel_length_m <= 0 or sdf_trunc_m <= 0:
        raise ValueError("Voxel length and SDF truncation must be positive")
    if sdf_trunc_m < voxel_length_m:
        raise ValueError(
            f"sdf_trunc ({sdf_trunc_m}) below voxel length ({voxel_length_m}); "
            "the field cannot interpolate across a surface"
        )
    if not np.isfinite([depth_min_m, depth_max_m]).all() or depth_max_m <= depth_min_m:
        raise ValueError("Depth range must satisfy 0 <= min < max")

    width, height, fx, fy, cx, cy = _require_intrinsics(intrinsics)
    pinhole = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_length_m),
        sdf_trunc=float(sdf_trunc_m),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    integrated = 0
    for frame in frames:
        if frame.depth_m.shape != (height, width):
            raise ValueError(
                f"Depth is {frame.depth_m.shape}, intrinsics describe {(height, width)}"
            )
        depth = np.where(
            np.isfinite(frame.depth_m)
            & (frame.depth_m > depth_min_m)
            & (frame.depth_m <= depth_max_m),
            frame.depth_m,
            0.0,
        ).astype(np.float32)
        depth = _mask_depth_to_crop(depth, frame.pose, intrinsics, crop_bounds)
        if not np.any(depth > 0):
            continue
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(frame.color_rgb)),
            o3d.geometry.Image(np.ascontiguousarray(depth)),
            depth_scale=1.0,
            depth_trunc=float(depth_max_m),
            convert_rgb_to_intensity=False,
        )
        # Poses map camera into the reference frame; integrate() wants the
        # inverse. Passing the pose itself mirrors the model about the origin.
        volume.integrate(rgbd, pinhole, np.linalg.inv(frame.pose))
        integrated += 1

    if integrated == 0:
        raise RuntimeError("No frame carried depth inside the fusion range")
    logger.info("Integrated %d/%d frames into the TSDF volume", integrated, len(frames))
    return volume


def _apply_crop(geometry, crop_bounds, *, is_mesh: bool):
    """Restrict extracted geometry to the same region the merge path crops to."""
    import open3d as o3d

    from pointcloud_processing import CylinderCrop

    if crop_bounds is None:
        return geometry
    if isinstance(crop_bounds, CylinderCrop):
        points = np.asarray(geometry.vertices if is_mesh else geometry.points)
        keep = np.flatnonzero(crop_bounds.mask(points))
        if is_mesh:
            return geometry.select_by_index(keep.tolist())
        return geometry.select_by_index(keep.tolist())
    bounds = np.asarray(crop_bounds, dtype=float)
    if bounds.shape != (6,) or not np.isfinite(bounds).all():
        raise ValueError("crop_bounds must contain six finite values")
    box = o3d.geometry.AxisAlignedBoundingBox(min_bound=bounds[:3], max_bound=bounds[3:])
    return geometry.crop(box)


def extract(volume, *, crop_bounds=None) -> tuple:
    """Pull the fused surface out as both a mesh and a point cloud."""
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    cloud = volume.extract_point_cloud()
    mesh = _apply_crop(mesh, crop_bounds, is_mesh=True)
    cloud = _apply_crop(cloud, crop_bounds, is_mesh=False)
    if len(mesh.vertices) == 0 and len(cloud.points) == 0:
        raise RuntimeError("TSDF extraction produced no geometry inside the crop")
    return mesh, cloud


def _remove_small_components(mesh) -> int:
    """Remove detached triangle islands too small to be meaningful anatomy."""
    if len(mesh.triangles) == 0:
        return 0
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    if len(counts) <= 1:
        return 0
    minimum = max(
        MIN_COMPONENT_TRIANGLES,
        int(np.ceil(float(counts.max()) * MIN_COMPONENT_FRACTION)),
    )
    remove = counts[labels] < minimum
    removed_components = int(np.count_nonzero(counts < minimum))
    if np.any(remove):
        mesh.remove_triangles_by_mask(remove.tolist())
        mesh.remove_unreferenced_vertices()
    return removed_components


def _small_boundary_loop_count(mesh, maximum_extent_m: float) -> int:
    """Count boundary loops whose axis-aligned extent permits hole repair."""
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if len(triangles) == 0:
        return 0
    edges = np.concatenate((
        triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)],
    ))
    edges.sort(axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique_edges[counts == 1]
    if len(boundary) == 0:
        return 0

    adjacency: dict[int, set[int]] = {}
    for left, right in boundary:
        adjacency.setdefault(int(left), set()).add(int(right))
        adjacency.setdefault(int(right), set()).add(int(left))

    vertices = np.asarray(mesh.vertices)
    unvisited = set(adjacency)
    small_loops = 0
    while unvisited:
        start = unvisited.pop()
        component = {start}
        pending = [start]
        while pending:
            vertex = pending.pop()
            for neighbor in adjacency[vertex]:
                if neighbor not in component:
                    component.add(neighbor)
                    unvisited.discard(neighbor)
                    pending.append(neighbor)
        extent = np.ptp(vertices[list(component)], axis=0)
        if float(np.max(extent)) <= maximum_extent_m:
            small_loops += 1
    return small_loops


def clean_mesh(mesh):
    """Repair tiny TSDF defects and lightly smooth a mesh without shrinkage."""
    import open3d as o3d

    raw_vertices = len(mesh.vertices)
    raw_triangles = len(mesh.triangles)
    mesh = o3d.geometry.TriangleMesh(mesh)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    removed_components = _remove_small_components(mesh)

    small_holes_before = _small_boundary_loop_count(mesh, MAX_REPAIR_HOLE_M)
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    mesh = tensor_mesh.fill_holes(MAX_REPAIR_HOLE_M).to_legacy()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    small_holes_after = _small_boundary_loop_count(mesh, MAX_REPAIR_HOLE_M)

    mesh = mesh.filter_smooth_taubin(
        number_of_iterations=TAUBIN_SMOOTHING_ITERATIONS,
    )
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh, {
        "raw_vertices": raw_vertices,
        "raw_triangles": raw_triangles,
        "cleaned_vertices": len(mesh.vertices),
        "cleaned_triangles": len(mesh.triangles),
        "removed_components": removed_components,
        "repaired_holes": max(0, small_holes_before - small_holes_after),
        "max_repair_hole_m": MAX_REPAIR_HOLE_M,
        "taubin_iterations": TAUBIN_SMOOTHING_ITERATIONS,
    }


def fuse_captures(
    ply_paths: Sequence[Path],
    poses: Sequence[np.ndarray],
    intrinsics: dict,
    *,
    input_dir: Path,
    output_dir: Path,
    voxel_length_m: float = DEFAULT_VOXEL_M,
    sdf_trunc_m: float = DEFAULT_SDF_TRUNC_M,
    depth_min_m: float = 0.0,
    depth_max_m: float = 1.0,
    depth_source: str = "sensor",
    crop_bounds=None,
) -> dict:
    """Fuse registered captures and write the mesh and cloud beside the merge."""
    import open3d as o3d

    if len(ply_paths) != len(poses):
        raise ValueError("Each capture needs exactly one pose")

    used_rgbd = 0
    frames = []
    for path, pose in zip(ply_paths, poses):
        frame = load_fusion_frame(
            Path(path), pose, intrinsics,
            input_dir=input_dir, depth_source=depth_source,
        )
        used_rgbd += int((input_dir / "rgbd" / Path(path).stem).is_dir())
        frames.append(frame)

    volume = integrate(
        frames, intrinsics,
        voxel_length_m=voxel_length_m, sdf_trunc_m=sdf_trunc_m,
        depth_min_m=depth_min_m, depth_max_m=depth_max_m,
        crop_bounds=crop_bounds,
    )
    mesh, cloud = extract(volume, crop_bounds=crop_bounds)
    cleaned_mesh, cleanup_stats = clean_mesh(mesh)

    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = output_dir / "tsdf_mesh.ply"
    cleaned_mesh_path = output_dir / "tsdf_mesh_cleaned.ply"
    cloud_path = output_dir / "tsdf_cloud.ply"
    if not o3d.io.write_triangle_mesh(str(mesh_path), mesh):
        raise RuntimeError(f"Could not write {mesh_path}")
    if not o3d.io.write_triangle_mesh(str(cleaned_mesh_path), cleaned_mesh):
        raise RuntimeError(f"Could not write {cleaned_mesh_path}")
    if not o3d.io.write_point_cloud(str(cloud_path), cloud):
        raise RuntimeError(f"Could not write {cloud_path}")

    stats = {
        "frames": len(frames),
        "depth_from_rgbd": used_rgbd,
        "depth_from_ply_reprojection": len(frames) - used_rgbd,
        "depth_source": depth_source,
        "voxel_length_m": float(voxel_length_m),
        "sdf_trunc_m": float(sdf_trunc_m),
        "depth_range_m": [float(depth_min_m), float(depth_max_m)],
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),
        "cloud_points": len(cloud.points),
        "mesh_path": str(mesh_path),
        "cleaned_mesh_path": str(cleaned_mesh_path),
        "cloud_path": str(cloud_path),
        "mesh_cleanup": cleanup_stats,
    }
    logger.info(
        "TSDF fused %d frames -> %d mesh vertices, %d cloud points",
        len(frames), len(mesh.vertices), len(cloud.points),
    )
    return stats

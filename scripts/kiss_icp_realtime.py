# #!/usr/bin/env python3
# """Real-time ZED + KISS-ICP mapping with live Open3D visualisation.

# The camera streams continuously. Every frame is:
#     1. Grabbed from the ZED SDK
#     2. Depth-cropped to [min_depth_m, max_depth_m]
#     3. Registered against the previous frame with KISS-ICP
#     4. Accumulated into a global map
#     5. Displayed live in an Open3D viewer

# Usage
# -----
# python scripts/kiss_icp_realtime.py \
#     --max-depth-m 0.35 \
#     --min-depth-m 0.05 \
#     --resolution HD720 \
#     --coordinate-system RIGHT_HANDED_Z_UP_X_FWD \
#     --voxel-m 0.002 \
#     --out outputs/kiss_icp_map.ply
# """

# from __future__ import annotations

# import argparse
# import time
# from pathlib import Path

# import numpy as np
# import pyzed.sl as sl

# # ── KISS-ICP import ────────────────────────────────────────────────────────────
# try:
#     from kiss_icp.kiss_icp import KissICP
#     from kiss_icp.config import KISSConfig
# except ImportError as exc:
#     raise SystemExit(
#         "KISS-ICP is required. Install it with:\n"
#         "  pip install kiss-icp"
#     ) from exc

# # ── Open3D import ──────────────────────────────────────────────────────────────
# try:
#     import open3d as o3d
# except ImportError as exc:
#     raise SystemExit(
#         "Open3D is required. Install it with:\n"
#         "  pip install open3d"
#     ) from exc


# # ──────────────────────────────────────────────────────────────────────────────
# # CLI
# # ──────────────────────────────────────────────────────────────────────────────

# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Real-time ZED + KISS-ICP mapping with live viewer."
#     )
#     parser.add_argument("--max-depth-m", type=float, default=0.35,
#                         help="Far depth cutoff in metres (default: 0.35)")
#     parser.add_argument("--min-depth-m", type=float, default=0.05,
#                         help="Near depth cutoff in metres (default: 0.05)")
#     parser.add_argument(
#         "--resolution",
#         choices=["HD2K", "HD1080", "HD720", "VGA"],
#         default="HD720",
#     )
#     parser.add_argument(
#         "--coordinate-system",
#         choices=["IMAGE", "RIGHT_HANDED_Z_UP_X_FWD"],
#         default="RIGHT_HANDED_Z_UP_X_FWD",
#     )
#     parser.add_argument("--voxel-m", type=float, default=0.002,
#                         help="Map voxel size in metres (default: 0.002)")
#     parser.add_argument("--max-map-points", type=int, default=5_000_000,
#                         help="Safety cap on accumulated map points")
#     parser.add_argument("--max-points-per-frame", type=int, default=80_000,
#                         help="Subsample each frame to this many points before KISS-ICP. "
#                              "Reduces CPU load and prevents map compaction every few frames. "
#                              "KISS-ICP voxelises internally so extra points don't help. "
#                              "Default 80000 is good for HD720; use 50000 for HD1080.")
#     parser.add_argument("--warmup", type=int, default=20,
#                         help="Frames to skip before mapping starts")
#     parser.add_argument("--out", type=Path, default=Path("outputs/kiss_icp_map.ply"),
#                         help="Output PLY path when you press Q to quit")
#     parser.add_argument("--no-viz", action="store_true",
#                         help="Disable Open3D live viewer (headless mode)")
#     # KISS-ICP tuning
#     parser.add_argument("--kiss-voxel-m", type=float, default=0.01,
#                         help="KISS-ICP internal voxel size (default: 0.01)")
#     parser.add_argument("--kiss-max-range-m", type=float, default=0.35,
#                         help="KISS-ICP max range — should match --max-depth-m")
#     parser.add_argument("--kiss-min-range-m", type=float, default=0.05,
#                         help="KISS-ICP min range — should match --min-depth-m")
#     return parser.parse_args()


# # ──────────────────────────────────────────────────────────────────────────────
# # ZED helpers
# # ──────────────────────────────────────────────────────────────────────────────

# def resolution_enum(name: str) -> sl.RESOLUTION:
#     return {
#         "HD2K": sl.RESOLUTION.HD2K,
#         "HD1080": sl.RESOLUTION.HD1080,
#         "HD720": sl.RESOLUTION.HD720,
#         "VGA": sl.RESOLUTION.VGA,
#     }[name]


# def coordinate_system_enum(name: str) -> sl.COORDINATE_SYSTEM:
#     return {
#         "IMAGE": sl.COORDINATE_SYSTEM.IMAGE,
#         "RIGHT_HANDED_Z_UP_X_FWD": sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD,
#     }[name]


# def extract_points_and_colors(
#     point_cloud: sl.Mat,
#     min_depth_m: float,
#     max_depth_m: float,
#     coordinate_system: str,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """Extract filtered (N,3) XYZ float32 and (N,3) RGB uint8 from a ZED Mat.

#     Depth axis depends on coordinate system:
#         IMAGE:                  forward = Z axis (col 2)
#         RIGHT_HANDED_Z_UP_X_FWD: forward = X axis (col 0)
#     """
#     cloud = point_cloud.get_data()          # (H, W, 4) float32 XYZRGBA
#     xyz = cloud[:, :, :3].reshape(-1, 3)
#     rgba_raw = cloud[:, :, 3].reshape(-1)

#     # Finite mask — rejects NaN and Inf
#     finite = np.isfinite(xyz).all(axis=1)

#     # Zero-magnitude mask — KISS-ICP's Sophus SO3::exp crashes on zero/near-zero
#     # vectors. ZED emits exact (0,0,0) for pixels with no depth solution.
#     magnitude = np.linalg.norm(xyz, axis=1)
#     nonzero = magnitude > 1e-6

#     # Forward-depth mask
#     if coordinate_system == "IMAGE":
#         forward_depth = xyz[:, 2]
#     else:  # RIGHT_HANDED_Z_UP_X_FWD
#         forward_depth = xyz[:, 0]

#     in_range = (forward_depth >= min_depth_m) & (forward_depth <= max_depth_m)
#     valid = finite & nonzero & in_range

#     points = xyz[valid].astype(np.float32)

#     # BGRA → RGB  (ZED packs as BGRA in the float channel)
#     rgba_uint32 = rgba_raw[valid].view(np.uint32)
#     b = (rgba_uint32 & 0xFF).astype(np.uint8)
#     g = ((rgba_uint32 >> 8) & 0xFF).astype(np.uint8)
#     r = ((rgba_uint32 >> 16) & 0xFF).astype(np.uint8)
#     colors = np.stack([r, g, b], axis=1)

#     return points, colors


# # ──────────────────────────────────────────────────────────────────────────────
# # Map helpers
# # ──────────────────────────────────────────────────────────────────────────────

# def voxel_downsample_numpy(
#     points: np.ndarray,
#     colors: np.ndarray,
#     voxel_m: float,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """Fast numpy voxel downsample — keeps one point per voxel cell."""
#     if voxel_m <= 0 or points.shape[0] == 0:
#         return points, colors
#     keys = np.floor(points / voxel_m).astype(np.int64)
#     _, idx = np.unique(keys, axis=0, return_index=True)
#     idx.sort()
#     return points[idx], colors[idx]


# def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
#     """Apply a 4×4 SE3 pose matrix to an (N,3) point array."""
#     R = pose[:3, :3].astype(np.float32)
#     t = pose[:3, 3].astype(np.float32)
#     return points @ R.T + t


# # ──────────────────────────────────────────────────────────────────────────────
# # Binary PLY writer
# # ──────────────────────────────────────────────────────────────────────────────

# def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
#     """Write a colored point cloud to a binary little-endian PLY file."""
#     path.parent.mkdir(parents=True, exist_ok=True)
#     n = points.shape[0]
#     header = (
#         "ply\n"
#         "format binary_little_endian 1.0\n"
#         f"element vertex {n}\n"
#         "property float x\n"
#         "property float y\n"
#         "property float z\n"
#         "property uchar red\n"
#         "property uchar green\n"
#         "property uchar blue\n"
#         "end_header\n"
#     )
#     dtype = np.dtype([
#         ("x", np.float32), ("y", np.float32), ("z", np.float32),
#         ("r", np.uint8),   ("g", np.uint8),   ("b", np.uint8),
#     ])
#     records = np.empty(n, dtype=dtype)
#     records["x"] = points[:, 0]
#     records["y"] = points[:, 1]
#     records["z"] = points[:, 2]
#     records["r"] = colors[:, 0]
#     records["g"] = colors[:, 1]
#     records["b"] = colors[:, 2]
#     with path.open("wb") as f:
#         f.write(header.encode("ascii"))
#         f.write(records.tobytes())


# # ──────────────────────────────────────────────────────────────────────────────
# # Open3D live visualiser
# # ──────────────────────────────────────────────────────────────────────────────

# class LiveVisualiser:
#     """Thin wrapper around an Open3D non-blocking visualiser window.

#     FIX: The original code called vis.add_geometry() on an empty PointCloud
#     in __init__, before any frames were captured. Open3D internally computes
#     an axis-aligned bounding box when adding geometry; on an empty cloud this
#     triggers a Sophus SO3::exp assertion deep in Open3D's C++ layer and causes
#     an immediate core dump — even before KISS-ICP processes a single frame.

#     Fix: do NOT add geometry in __init__. Instead, add it lazily on the first
#     update() call when we have real points. This is the only safe approach.
#     """

#     UPDATE_EVERY_N_FRAMES = 5   # refresh display every N frames to save GPU

#     def __init__(self) -> None:
#         self.vis = o3d.visualization.Visualizer()
#         self.vis.create_window(
#             window_name="KISS-ICP Real-time Map  [Q = quit & save]",
#             width=1280,
#             height=720,
#         )
#         self.cloud = o3d.geometry.PointCloud()
#         # FIX: do NOT add empty geometry here — deferred to first update()
#         self._geometry_added = False
#         self._frame_count = 0

#         # render options
#         opt = self.vis.get_render_option()
#         opt.background_color = np.array([0.05, 0.05, 0.05])
#         opt.point_size = 1.5

#     def update(self, points: np.ndarray, colors: np.ndarray) -> bool:
#         """Push new map data to the viewer. Returns False if window was closed."""
#         self._frame_count += 1
#         if self._frame_count % self.UPDATE_EVERY_N_FRAMES != 0:
#             self.vis.poll_events()
#             return True

#         self.cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
#         self.cloud.colors = o3d.utility.Vector3dVector(
#             colors.astype(np.float64) / 255.0
#         )

#         if not self._geometry_added:
#             # FIX: add geometry only when we have real points — avoids the
#             # empty-cloud bounding-box crash in Open3D/Sophus
#             self.vis.add_geometry(self.cloud)
#             self._geometry_added = True
#             self.vis.reset_view_point(True)
#         else:
#             self.vis.update_geometry(self.cloud)

#         self.vis.poll_events()
#         self.vis.update_renderer()
#         return True

#     def is_open(self) -> bool:
#         return self.vis.poll_events()

#     def destroy(self) -> None:
#         self.vis.destroy_window()


# # ──────────────────────────────────────────────────────────────────────────────
# # KISS-ICP setup
# # ──────────────────────────────────────────────────────────────────────────────

# def _set_if_exists(obj: object, field: str, value) -> bool:
#     """Set obj.field = value only if the field exists. Returns True on success."""
#     if hasattr(obj, field):
#         setattr(obj, field, value)
#         return True
#     return False


# def build_kiss_icp(args: argparse.Namespace) -> KissICP:
#     """Construct a KissICP instance tuned for close-range ZED scanning.

#     KISSConfig's internal structure changed between versions:
#         v0.x : flat fields directly on KISSConfig  (voxel_size, max_range, …)
#         v1.0 : nested sub-models                   (config.mapping.voxel_size, …)
#         v1.1+: nested sub-models with different names or removed fields

#     Rather than hard-coding one version's layout, this function inspects the
#     actual config object at runtime and sets whatever fields exist.  Fields
#     that have been removed in a newer version are simply skipped with a warning
#     so the script keeps running with KISS-ICP defaults for those parameters.
#     """
#     config = KISSConfig()

#     # Print what the installed version actually exposes so debugging is easy
#     print(f"KISSConfig fields: {list(KISSConfig.model_fields.keys())}")

#     # ── helper: try nested then flat ─────────────────────────────────────────
#     def apply(nested_path: str, flat_name: str, value) -> None:
#         """
#         Try config.<sub>.<field> first (v1.x nested style).
#         Fall back to config.<flat_name> (v0.x flat style).
#         Warn if neither exists.
#         """
#         parts = nested_path.split(".")           # e.g. ["mapping", "voxel_size"]
#         sub_name, field_name = parts[0], parts[1]

#         sub = getattr(config, sub_name, None)
#         if sub is not None and hasattr(sub, field_name):
#             setattr(sub, field_name, value)
#             return

#         if _set_if_exists(config, flat_name, value):
#             return

#         print(f"  Warning: config field '{nested_path}' / '{flat_name}' "
#               f"not found in this KISS-ICP version — using default.")

#     # ── apply parameters ──────────────────────────────────────────────────────
#     # Confirmed fields in this version:
#     #   data, registration, mapping, adaptive_threshold
#     apply("mapping.voxel_size",              "voxel_size",         args.kiss_voxel_m)
#     apply("data.max_range",                  "max_range",          args.kiss_max_range_m)
#     apply("data.min_range",                  "min_range",          args.kiss_min_range_m)
#     apply("data.deskew",                     "deskew",             False)  # ZED captures both eyes simultaneously
#     apply("registration.max_num_iterations", "max_num_iterations", 500)
#     apply("registration.convergence_criterion", "convergence_criterion", 0.0001)

#     # ── CRITICAL for close-range scanning ─────────────────────────────────────
#     # Default min_motion_th=0.1 means KISS-ICP skips registration unless the
#     # camera moves >10cm between frames. At 0.35m max depth this effectively
#     # disables odometry entirely — pose stays at identity every frame.
#     # Set to a very small value so every frame is registered regardless of motion.
#     apply("adaptive_threshold.min_motion_th",  "min_motion_th",   0.001)
#     # initial_threshold: maximum correspondence distance for the first frame.
#     # Default 2.0m is for outdoor LiDAR. For close-range ZED at 0.35m we want
#     # ~0.1m — large enough to find correspondences at 2-5fps hand motion speeds,
#     # small enough to reject background noise matches.
#     apply("adaptive_threshold.initial_threshold", "initial_threshold", 0.1)
#     # convergence_criterion — also present in registration in this version
#     apply("odometry.convergence_criterion",    "convergence_criterion", 0.0001)

#     return KissICP(config=config)


# # ──────────────────────────────────────────────────────────────────────────────
# # Main loop
# # ──────────────────────────────────────────────────────────────────────────────

# def main() -> None:
#     args = parse_args()

#     # ── open ZED ──────────────────────────────────────────────────────────────
#     init = sl.InitParameters()
#     init.camera_resolution = resolution_enum(args.resolution)
#     init.depth_mode = sl.DEPTH_MODE.NEURAL
#     init.coordinate_units = sl.UNIT.METER
#     init.coordinate_system = coordinate_system_enum(args.coordinate_system)
#     init.depth_minimum_distance = args.min_depth_m
#     init.depth_maximum_distance = args.max_depth_m

#     zed = sl.Camera()
#     status = zed.open(init)
#     if status != sl.ERROR_CODE.SUCCESS:
#         raise RuntimeError(f"Could not open ZED camera: {status}")

#     runtime = sl.RuntimeParameters()
#     point_cloud_mat = sl.Mat()

#     # ── warmup ────────────────────────────────────────────────────────────────
#     print(f"Warming up ({args.warmup} frames)…")
#     for i in range(args.warmup):
#         if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
#             raise RuntimeError(f"Warmup frame {i + 1} failed")
#     print("Warmup done. Starting mapping. Press Q in the viewer to stop.")

#     # ── KISS-ICP ──────────────────────────────────────────────────────────────
#     kiss = build_kiss_icp(args)

#     # ── global map accumulators ───────────────────────────────────────────────
#     map_points: list[np.ndarray] = []
#     map_colors: list[np.ndarray] = []
#     map_point_count = 0
#     all_poses: list[np.ndarray] = []   # manual trajectory for v1.1+ compatibility

#     # ── visualiser ────────────────────────────────────────────────────────────
#     viz: LiveVisualiser | None = None
#     if not args.no_viz:
#         viz = LiveVisualiser()

#     # ── timing ────────────────────────────────────────────────────────────────
#     frame_idx = 0
#     t_start = time.perf_counter()

#     try:
#         while True:
#             # ── check if viewer was closed ─────────────────────────────────
#             if viz is not None and not viz.is_open():
#                 print("Viewer closed — stopping.")
#                 break

#             # ── grab frame ────────────────────────────────────────────────
#             grab_status = zed.grab(runtime)
#             if grab_status != sl.ERROR_CODE.SUCCESS:
#                 print(f"Frame grab failed: {grab_status} — skipping")
#                 continue

#             zed.retrieve_measure(point_cloud_mat, sl.MEASURE.XYZRGBA)

#             # ── extract points ────────────────────────────────────────────
#             frame_pts, frame_col = extract_points_and_colors(
#                 point_cloud_mat,
#                 min_depth_m=args.min_depth_m,
#                 max_depth_m=args.max_depth_m,
#                 coordinate_system=args.coordinate_system,
#             )

#             # KISS-ICP needs enough geometry to find a reliable rotation.
#             # Too few points and Sophus will produce degenerate transforms.
#             if frame_pts.shape[0] < 1000:
#                 print(f"Frame {frame_idx}: too few points ({frame_pts.shape[0]}) — skipping")
#                 frame_idx += 1
#                 continue

#             # Diagnostic on first good frame — confirm points look sane
#             if frame_idx == 0:
#                 print(f"First frame: {frame_pts.shape[0]} points")
#                 print(f"  XYZ min:  {frame_pts.min(axis=0)}")
#                 print(f"  XYZ max:  {frame_pts.max(axis=0)}")
#                 print(f"  XYZ mean: {frame_pts.mean(axis=0)}")

#             # ── per-frame subsampling ─────────────────────────────────────
#             # Limit points fed to KISS-ICP. The algorithm voxelises internally
#             # so extra points beyond ~80k don't improve registration but do
#             # slow down each frame significantly (HD1080 gives 200k+ points).
#             if frame_pts.shape[0] > args.max_points_per_frame:
#                 idx = np.random.choice(
#                     frame_pts.shape[0], args.max_points_per_frame, replace=False
#                 )
#                 idx.sort()
#                 frame_pts = frame_pts[idx]
#                 frame_col = frame_col[idx]

#             # ── KISS-ICP register ─────────────────────────────────────────
#             # Pass an EMPTY timestamps array (shape (0,)) to disable motion
#             # deskewing entirely. This is correct for ZED stereo — both images
#             # are captured at the same instant, so there is no within-frame
#             # motion to compensate.
#             #
#             # Do NOT pass all-zeros of length N — that triggers a degenerate
#             # division-by-zero inside KISS-ICP's deskew path which causes
#             # Sophus SO3::exp to abort with SIGABRT (uncatchable by Python).
#             try:
#                 kiss.register_frame(
#                     frame_pts.astype(np.float64),
#                     timestamps=np.array([], dtype=np.float64),
#                 )
#             except Exception as exc:
#                 print(f"\nFrame {frame_idx}: KISS-ICP register failed: {exc} — skipping")
#                 frame_idx += 1
#                 continue

#             # ── get latest pose ───────────────────────────────────────────
#             # kiss.last_pose is the current (4,4) SE3 pose in world frame.
#             pose = kiss.last_pose.astype(np.float64)

#             # Store pose for trajectory (needed in v1.1+ which dropped kiss.poses)
#             all_poses.append(pose.copy())

#             # ── transform frame to world ──────────────────────────────────
#             world_pts = transform_points(frame_pts, pose)

#             # ── accumulate into map ───────────────────────────────────────
#             if map_point_count + world_pts.shape[0] <= args.max_map_points:
#                 map_points.append(world_pts)
#                 map_colors.append(frame_col)
#                 map_point_count += world_pts.shape[0]
#             else:
#                 # Map is full — voxel-downsample the whole thing to make room
#                 all_pts = np.concatenate(map_points, axis=0)
#                 all_col = np.concatenate(map_colors, axis=0)
#                 all_pts, all_col = voxel_downsample_numpy(all_pts, all_col, args.voxel_m)
#                 map_points = [all_pts, world_pts]
#                 map_colors = [all_col, frame_col]
#                 map_point_count = all_pts.shape[0] + world_pts.shape[0]
#                 print(f"Frame {frame_idx}: map compacted to {map_point_count} points")

#             # ── fps display ───────────────────────────────────────────────
#             frame_idx += 1
#             elapsed = time.perf_counter() - t_start
#             fps = frame_idx / elapsed if elapsed > 0 else 0.0

#             t_xyz = pose[:3, 3]
#             print(
#                 f"Frame {frame_idx:5d} | "
#                 f"pts={frame_pts.shape[0]:6d} | "
#                 f"map={map_point_count:7d} | "
#                 f"pos=({t_xyz[0]:.3f}, {t_xyz[1]:.3f}, {t_xyz[2]:.3f}) | "
#                 f"fps={fps:.1f}",
#                 end="\r",
#             )

#             # ── update viewer ─────────────────────────────────────────────
#             if viz is not None:
#                 display_pts = np.concatenate(map_points, axis=0)
#                 display_col = np.concatenate(map_colors, axis=0)
#                 viz.update(display_pts, display_col)

#     except KeyboardInterrupt:
#         print("\nKeyboard interrupt — saving map…")
#     finally:
#         zed.close()
#         if viz is not None:
#             viz.destroy()

#     # ── final downsample and save ─────────────────────────────────────────────
#     print(f"\nTotal frames processed: {frame_idx}")
#     if not map_points:
#         print("No points accumulated — nothing saved.")
#         return

#     print("Downsampling final map…")
#     final_pts = np.concatenate(map_points, axis=0)
#     final_col = np.concatenate(map_colors, axis=0)
#     final_pts, final_col = voxel_downsample_numpy(final_pts, final_col, args.voxel_m)
#     print(f"Final map: {final_pts.shape[0]} points")

#     write_binary_ply(args.out, final_pts, final_col)
#     print(f"Saved → {args.out}")

#     # ── also save trajectory ──────────────────────────────────────────────────
#     # Handle both v0.x (kiss.poses list) and v1.1+ (no stored trajectory).
#     # In v1.1+ we accumulated poses manually into all_poses during the loop.
#     trajectory = None
#     if hasattr(kiss, 'poses') and len(kiss.poses) > 0:
#         trajectory = np.stack(kiss.poses, axis=0)
#     elif all_poses:
#         trajectory = np.stack(all_poses, axis=0)

#     if trajectory is not None:
#         traj_path = args.out.with_stem(args.out.stem + '_trajectory').with_suffix('.npy')
#         np.save(traj_path, trajectory)
#         print(f'Trajectory saved → {traj_path}  ({len(trajectory)} poses)')


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""Real-time ZED + KISS-ICP mapping with live Open3D visualisation.

The camera streams continuously. Every frame is:
    1. Grabbed from the ZED SDK
    2. Depth-cropped to [min_depth_m, max_depth_m]
    3. Registered against the previous frame with KISS-ICP
    4. Accumulated into a global map
    5. Displayed live in an Open3D viewer

Usage
-----
python scripts/kiss_icp_realtime.py \
    --max-depth-m 0.35 \
    --min-depth-m 0.05 \
    --resolution HD720 \
    --coordinate-system RIGHT_HANDED_Z_UP_X_FWD \
    --voxel-m 0.002 \
    --out outputs/kiss_icp_map.ply
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyzed.sl as sl

# ── KISS-ICP import ────────────────────────────────────────────────────────────
try:
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig
except ImportError as exc:
    raise SystemExit(
        "KISS-ICP is required. Install it with:\n"
        "  pip install kiss-icp"
    ) from exc

# ── Open3D import ──────────────────────────────────────────────────────────────
try:
    import open3d as o3d
except ImportError as exc:
    raise SystemExit(
        "Open3D is required. Install it with:\n"
        "  pip install open3d"
    ) from exc


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time ZED + KISS-ICP mapping with live viewer."
    )
    parser.add_argument("--max-depth-m", type=float, default=0.35,
                        help="Far depth cutoff in metres (default: 0.35)")
    parser.add_argument("--min-depth-m", type=float, default=0.05,
                        help="Near depth cutoff in metres (default: 0.05)")
    parser.add_argument(
        "--resolution",
        choices=["HD2K", "HD1080", "HD720", "VGA"],
        default="HD720",
    )
    parser.add_argument(
        "--coordinate-system",
        choices=["IMAGE", "RIGHT_HANDED_Z_UP_X_FWD"],
        default="RIGHT_HANDED_Z_UP_X_FWD",
    )
    parser.add_argument("--voxel-m", type=float, default=0.002,
                        help="Map voxel size in metres (default: 0.002)")
    parser.add_argument("--max-map-points", type=int, default=5_000_000,
                        help="Safety cap on accumulated map points")
    parser.add_argument("--max-points-per-frame", type=int, default=80_000,
                        help="Subsample each frame to this many points before KISS-ICP. "
                             "Reduces CPU load and prevents map compaction every few frames. "
                             "KISS-ICP voxelises internally so extra points don't help. "
                             "Default 80000 is good for HD720; use 50000 for HD1080.")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Frames to skip before mapping starts")
    parser.add_argument("--out", type=Path, default=Path("outputs/kiss_icp_map.ply"),
                        help="Output PLY path when you press Q to quit")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable Open3D live viewer (headless mode)")
    parser.add_argument("--max-jump-m", type=float, default=0.15,
                        help="Discard frames where pose jumps more than this many metres "
                             "in one step — guards against tracking loss after dropouts. "
                             "Default 0.15m. Increase if you move the camera fast.")
    # KISS-ICP tuning
    parser.add_argument("--kiss-voxel-m", type=float, default=0.01,
                        help="KISS-ICP internal voxel size (default: 0.01)")
    parser.add_argument("--kiss-max-range-m", type=float, default=0.35,
                        help="KISS-ICP max range — should match --max-depth-m")
    parser.add_argument("--kiss-min-range-m", type=float, default=0.05,
                        help="KISS-ICP min range — should match --min-depth-m")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# ZED helpers
# ──────────────────────────────────────────────────────────────────────────────

def resolution_enum(name: str) -> sl.RESOLUTION:
    return {
        "HD2K": sl.RESOLUTION.HD2K,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "VGA": sl.RESOLUTION.VGA,
    }[name]


def coordinate_system_enum(name: str) -> sl.COORDINATE_SYSTEM:
    return {
        "IMAGE": sl.COORDINATE_SYSTEM.IMAGE,
        "RIGHT_HANDED_Z_UP_X_FWD": sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD,
    }[name]


def extract_points_and_colors(
    point_cloud: sl.Mat,
    min_depth_m: float,
    max_depth_m: float,
    coordinate_system: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract filtered (N,3) XYZ float32 and (N,3) RGB uint8 from a ZED Mat.

    Depth axis depends on coordinate system:
        IMAGE:                  forward = Z axis (col 2)
        RIGHT_HANDED_Z_UP_X_FWD: forward = X axis (col 0)
    """
    cloud = point_cloud.get_data()          # (H, W, 4) float32 XYZRGBA
    xyz = cloud[:, :, :3].reshape(-1, 3)
    rgba_raw = cloud[:, :, 3].reshape(-1)

    # Finite mask — rejects NaN and Inf
    finite = np.isfinite(xyz).all(axis=1)

    # Zero-magnitude mask — KISS-ICP's Sophus SO3::exp crashes on zero/near-zero
    # vectors. ZED emits exact (0,0,0) for pixels with no depth solution.
    magnitude = np.linalg.norm(xyz, axis=1)
    nonzero = magnitude > 1e-6

    # Forward-depth mask
    if coordinate_system == "IMAGE":
        forward_depth = xyz[:, 2]
    else:  # RIGHT_HANDED_Z_UP_X_FWD
        forward_depth = xyz[:, 0]

    in_range = (forward_depth >= min_depth_m) & (forward_depth <= max_depth_m)
    valid = finite & nonzero & in_range

    points = xyz[valid].astype(np.float32)

    # BGRA → RGB  (ZED packs as BGRA in the float channel)
    rgba_uint32 = rgba_raw[valid].view(np.uint32)
    b = (rgba_uint32 & 0xFF).astype(np.uint8)
    g = ((rgba_uint32 >> 8) & 0xFF).astype(np.uint8)
    r = ((rgba_uint32 >> 16) & 0xFF).astype(np.uint8)
    colors = np.stack([r, g, b], axis=1)

    return points, colors


# ──────────────────────────────────────────────────────────────────────────────
# Map helpers
# ──────────────────────────────────────────────────────────────────────────────

def voxel_downsample_numpy(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast numpy voxel downsample — keeps one point per voxel cell."""
    if voxel_m <= 0 or points.shape[0] == 0:
        return points, colors
    keys = np.floor(points / voxel_m).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx.sort()
    return points[idx], colors[idx]


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Apply a 4×4 SE3 pose matrix to an (N,3) point array."""
    R = pose[:3, :3].astype(np.float32)
    t = pose[:3, 3].astype(np.float32)
    return points @ R.T + t


# ──────────────────────────────────────────────────────────────────────────────
# Binary PLY writer
# ──────────────────────────────────────────────────────────────────────────────

def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a colored point cloud to a binary little-endian PLY file."""
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
    dtype = np.dtype([
        ("x", np.float32), ("y", np.float32), ("z", np.float32),
        ("r", np.uint8),   ("g", np.uint8),   ("b", np.uint8),
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


# ──────────────────────────────────────────────────────────────────────────────
# Open3D live visualiser
# ──────────────────────────────────────────────────────────────────────────────

class LiveVisualiser:
    """Thin wrapper around an Open3D non-blocking visualiser window.

    FIX: The original code called vis.add_geometry() on an empty PointCloud
    in __init__, before any frames were captured. Open3D internally computes
    an axis-aligned bounding box when adding geometry; on an empty cloud this
    triggers a Sophus SO3::exp assertion deep in Open3D's C++ layer and causes
    an immediate core dump — even before KISS-ICP processes a single frame.

    Fix: do NOT add geometry in __init__. Instead, add it lazily on the first
    update() call when we have real points. This is the only safe approach.
    """

    UPDATE_EVERY_N_FRAMES = 5   # refresh display every N frames to save GPU

    def __init__(self) -> None:
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name="KISS-ICP Real-time Map  [Q = quit & save]",
            width=1280,
            height=720,
        )
        self.cloud = o3d.geometry.PointCloud()
        # FIX: do NOT add empty geometry here — deferred to first update()
        self._geometry_added = False
        self._frame_count = 0

        # render options
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.05])
        opt.point_size = 1.5

    def update(self, points: np.ndarray, colors: np.ndarray) -> bool:
        """Push new map data to the viewer. Returns False if window was closed."""
        self._frame_count += 1
        if self._frame_count % self.UPDATE_EVERY_N_FRAMES != 0:
            self.vis.poll_events()
            return True

        self.cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        self.cloud.colors = o3d.utility.Vector3dVector(
            colors.astype(np.float64) / 255.0
        )

        if not self._geometry_added:
            # FIX: add geometry only when we have real points — avoids the
            # empty-cloud bounding-box crash in Open3D/Sophus
            self.vis.add_geometry(self.cloud)
            self._geometry_added = True
            self.vis.reset_view_point(True)
        else:
            self.vis.update_geometry(self.cloud)

        self.vis.poll_events()
        self.vis.update_renderer()
        return True

    def is_open(self) -> bool:
        return self.vis.poll_events()

    def destroy(self) -> None:
        self.vis.destroy_window()


# ──────────────────────────────────────────────────────────────────────────────
# KISS-ICP setup
# ──────────────────────────────────────────────────────────────────────────────

def _set_if_exists(obj: object, field: str, value) -> bool:
    """Set obj.field = value only if the field exists. Returns True on success."""
    if hasattr(obj, field):
        setattr(obj, field, value)
        return True
    return False


def build_kiss_icp(args: argparse.Namespace) -> KissICP:
    """Construct a KissICP instance tuned for close-range ZED scanning.

    KISSConfig's internal structure changed between versions:
        v0.x : flat fields directly on KISSConfig  (voxel_size, max_range, …)
        v1.0 : nested sub-models                   (config.mapping.voxel_size, …)
        v1.1+: nested sub-models with different names or removed fields

    Rather than hard-coding one version's layout, this function inspects the
    actual config object at runtime and sets whatever fields exist.  Fields
    that have been removed in a newer version are simply skipped with a warning
    so the script keeps running with KISS-ICP defaults for those parameters.
    """
    config = KISSConfig()

    # Print what the installed version actually exposes so debugging is easy
    print(f"KISSConfig fields: {list(KISSConfig.model_fields.keys())}")

    # ── helper: try nested then flat ─────────────────────────────────────────
    def apply(nested_path: str, flat_name: str, value) -> None:
        """
        Try config.<sub>.<field> first (v1.x nested style).
        Fall back to config.<flat_name> (v0.x flat style).
        Warn if neither exists.
        """
        parts = nested_path.split(".")           # e.g. ["mapping", "voxel_size"]
        sub_name, field_name = parts[0], parts[1]

        sub = getattr(config, sub_name, None)
        if sub is not None and hasattr(sub, field_name):
            setattr(sub, field_name, value)
            return

        if _set_if_exists(config, flat_name, value):
            return

        print(f"  Warning: config field '{nested_path}' / '{flat_name}' "
              f"not found in this KISS-ICP version — using default.")

    # ── apply parameters ──────────────────────────────────────────────────────
    # Confirmed fields in this version:
    #   data, registration, mapping, adaptive_threshold
    apply("mapping.voxel_size",              "voxel_size",         args.kiss_voxel_m)
    apply("data.max_range",                  "max_range",          args.kiss_max_range_m)
    apply("data.min_range",                  "min_range",          args.kiss_min_range_m)
    apply("data.deskew",                     "deskew",             False)  # ZED captures both eyes simultaneously
    apply("registration.max_num_iterations", "max_num_iterations", 500)
    apply("registration.convergence_criterion", "convergence_criterion", 0.0001)

    # ── CRITICAL for close-range scanning ─────────────────────────────────────
    # Default min_motion_th=0.1 means KISS-ICP skips registration unless the
    # camera moves >10cm between frames. At 0.35m max depth this effectively
    # disables odometry entirely — pose stays at identity every frame.
    # Set to a very small value so every frame is registered regardless of motion.
    apply("adaptive_threshold.min_motion_th",  "min_motion_th",   0.001)
    # initial_threshold: maximum correspondence distance for the first frame.
    # Default 2.0m is for outdoor LiDAR. For close-range ZED at 0.35m we want
    # ~0.1m — large enough to find correspondences at 2-5fps hand motion speeds,
    # small enough to reject background noise matches.
    apply("adaptive_threshold.initial_threshold", "initial_threshold", 0.1)
    return KissICP(config=config)


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── open ZED ──────────────────────────────────────────────────────────────
    init = sl.InitParameters()
    init.camera_resolution = resolution_enum(args.resolution)
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = coordinate_system_enum(args.coordinate_system)
    init.depth_minimum_distance = args.min_depth_m
    init.depth_maximum_distance = args.max_depth_m

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open ZED camera: {status}")

    runtime = sl.RuntimeParameters()
    point_cloud_mat = sl.Mat()

    # ── warmup ────────────────────────────────────────────────────────────────
    print(f"Warming up ({args.warmup} frames)…")
    for i in range(args.warmup):
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Warmup frame {i + 1} failed")
    print("Warmup done. Starting mapping. Press Q in the viewer to stop.")

    # ── KISS-ICP ──────────────────────────────────────────────────────────────
    kiss = build_kiss_icp(args)

    # ── global map accumulators ───────────────────────────────────────────────
    map_points: list[np.ndarray] = []
    map_colors: list[np.ndarray] = []
    map_point_count = 0
    all_poses: list[np.ndarray] = []   # manual trajectory for v1.1+ compatibility

    # ── visualiser ────────────────────────────────────────────────────────────
    viz: LiveVisualiser | None = None
    if not args.no_viz:
        viz = LiveVisualiser()

    # ── timing ────────────────────────────────────────────────────────────────
    frame_idx = 0
    skipped_frames = 0
    t_start = time.perf_counter()

    try:
        while True:
            # ── check if viewer was closed ─────────────────────────────────
            if viz is not None and not viz.is_open():
                print("Viewer closed — stopping.")
                break

            # ── grab frame ────────────────────────────────────────────────
            grab_status = zed.grab(runtime)
            if grab_status != sl.ERROR_CODE.SUCCESS:
                print(f"Frame grab failed: {grab_status} — skipping")
                continue

            zed.retrieve_measure(point_cloud_mat, sl.MEASURE.XYZRGBA)

            # ── extract points ────────────────────────────────────────────
            frame_pts, frame_col = extract_points_and_colors(
                point_cloud_mat,
                min_depth_m=args.min_depth_m,
                max_depth_m=args.max_depth_m,
                coordinate_system=args.coordinate_system,
            )

            # ── sparse / empty frame guard ────────────────────────────────
            # Below 500 points KISS-ICP can produce degenerate transforms.
            # All sparse/empty frames are silently skipped and counted.
            # A summary line prints every 30 skipped frames so you know
            # the camera is out of range without flooding the terminal.
            if frame_pts.shape[0] < 500:
                skipped_frames += 1
                if skipped_frames % 30 == 0:
                    print(f"  [{skipped_frames} low-point frames skipped so far "
                          f"— move camera closer to object]")
                frame_idx += 1
                continue
            # Reset skip counter when a good frame arrives
            if skipped_frames > 0:
                if skipped_frames >= 5:
                    print(f"  [recovered after {skipped_frames} skipped frames]")
                skipped_frames = 0

            # Diagnostic on first good frame — confirm points look sane
            if frame_idx == 0:
                print(f"First frame: {frame_pts.shape[0]} points")
                print(f"  XYZ min:  {frame_pts.min(axis=0)}")
                print(f"  XYZ max:  {frame_pts.max(axis=0)}")
                print(f"  XYZ mean: {frame_pts.mean(axis=0)}")

            # ── per-frame subsampling ─────────────────────────────────────
            # Limit points fed to KISS-ICP. The algorithm voxelises internally
            # so extra points beyond ~80k don't improve registration but do
            # slow down each frame significantly (HD1080 gives 200k+ points).
            if frame_pts.shape[0] > args.max_points_per_frame:
                idx = np.random.choice(
                    frame_pts.shape[0], args.max_points_per_frame, replace=False
                )
                idx.sort()
                frame_pts = frame_pts[idx]
                frame_col = frame_col[idx]

            # ── KISS-ICP register ─────────────────────────────────────────
            # Pass an EMPTY timestamps array (shape (0,)) to disable motion
            # deskewing entirely. This is correct for ZED stereo — both images
            # are captured at the same instant, so there is no within-frame
            # motion to compensate.
            #
            # Do NOT pass all-zeros of length N — that triggers a degenerate
            # division-by-zero inside KISS-ICP's deskew path which causes
            # Sophus SO3::exp to abort with SIGABRT (uncatchable by Python).
            try:
                kiss.register_frame(
                    frame_pts.astype(np.float64),
                    timestamps=np.array([], dtype=np.float64),
                )
            except Exception as exc:
                print(f"\nFrame {frame_idx}: KISS-ICP register failed: {exc} — skipping")
                frame_idx += 1
                continue

            # ── get latest pose ───────────────────────────────────────────
            # kiss.last_pose is the current (4,4) SE3 pose in world frame.
            pose = kiss.last_pose.astype(np.float64)

            # ── tracking loss detection ──────────────────────────────────
            # If the pose jumps more than max_jump_m in one frame, KISS-ICP
            # has likely lost tracking (e.g. after a long dropout gap).
            # Discard the frame rather than corrupting the map with a bad pose.
            max_jump_m = args.max_jump_m
            if all_poses:
                prev_t = all_poses[-1][:3, 3]
                curr_t = pose[:3, 3]
                jump_m = float(np.linalg.norm(curr_t - prev_t))
                if jump_m > max_jump_m:
                    print(f"\nFrame {frame_idx}: tracking jump {jump_m:.3f}m > {max_jump_m}m — discarding")
                    frame_idx += 1
                    continue

            # Store pose for trajectory (needed in v1.1+ which dropped kiss.poses)
            all_poses.append(pose.copy())

            # ── transform frame to world ──────────────────────────────────
            world_pts = transform_points(frame_pts, pose)

            # ── accumulate into map ───────────────────────────────────────
            if map_point_count + world_pts.shape[0] <= args.max_map_points:
                map_points.append(world_pts)
                map_colors.append(frame_col)
                map_point_count += world_pts.shape[0]
            else:
                # Map is full — voxel-downsample the whole thing to make room
                all_pts = np.concatenate(map_points, axis=0)
                all_col = np.concatenate(map_colors, axis=0)
                all_pts, all_col = voxel_downsample_numpy(all_pts, all_col, args.voxel_m)
                map_points = [all_pts, world_pts]
                map_colors = [all_col, frame_col]
                map_point_count = all_pts.shape[0] + world_pts.shape[0]
                print(f"Frame {frame_idx}: map compacted to {map_point_count} points")

            # ── fps display ───────────────────────────────────────────────
            frame_idx += 1
            elapsed = time.perf_counter() - t_start
            fps = frame_idx / elapsed if elapsed > 0 else 0.0

            t_xyz = pose[:3, 3]
            print(
                f"Frame {frame_idx:5d} | "
                f"pts={frame_pts.shape[0]:6d} | "
                f"map={map_point_count:7d} | "
                f"pos=({t_xyz[0]:.3f}, {t_xyz[1]:.3f}, {t_xyz[2]:.3f}) | "
                f"fps={fps:.1f}",
                end="\r",
            )

            # ── update viewer ─────────────────────────────────────────────
            if viz is not None:
                display_pts = np.concatenate(map_points, axis=0)
                display_col = np.concatenate(map_colors, axis=0)
                viz.update(display_pts, display_col)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt — saving map…")
    finally:
        zed.close()
        if viz is not None:
            viz.destroy()

    # ── final downsample and save ─────────────────────────────────────────────
    print(f"\nTotal frames processed: {frame_idx}")
    if not map_points:
        print("No points accumulated — nothing saved.")
        return

    print("Downsampling final map…")
    final_pts = np.concatenate(map_points, axis=0)
    final_col = np.concatenate(map_colors, axis=0)
    final_pts, final_col = voxel_downsample_numpy(final_pts, final_col, args.voxel_m)
    print(f"Final map: {final_pts.shape[0]} points")

    write_binary_ply(args.out, final_pts, final_col)
    print(f"Saved → {args.out}")

    # ── also save trajectory ──────────────────────────────────────────────────
    # Handle both v0.x (kiss.poses list) and v1.1+ (no stored trajectory).
    # In v1.1+ we accumulated poses manually into all_poses during the loop.
    trajectory = None
    if hasattr(kiss, 'poses') and len(kiss.poses) > 0:
        trajectory = np.stack(kiss.poses, axis=0)
    elif all_poses:
        trajectory = np.stack(all_poses, axis=0)

    if trajectory is not None:
        traj_path = args.out.with_stem(args.out.stem + '_trajectory').with_suffix('.npy')
        np.save(traj_path, trajectory)
        print(f'Trajectory saved → {traj_path}  ({len(trajectory)} poses)')


if __name__ == "__main__":
    main()

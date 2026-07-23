# Orbit-Prior Pose-Graph Fusion

## Objective

Reconstruct the stationary scan object from `capture_zed_angle.py` RGB-D
captures while the ZED Mini moves on a measured circular rail. Use the saved
motor angle and measured optical-center radius as the primary camera pose.
Point-cloud registration may refine that pose, but it must not replace the
mechanical orbit with an unconstrained estimate.

## Coordinate model

- Captures use ZED/Open3D IMAGE coordinates: camera `+X` right, `+Y` down,
  `+Z` forward.
- World `+Z` is up.
- The object/pivot has one fixed world center. A centroid is not estimated
  independently for every frame because background geometry would move it.
- A node pose is camera-to-world.
- A graph edge `i -> j` transforms points from camera `i` coordinates into
  camera `j` coordinates:

  `T_j_i = inverse(T_world_j) @ T_world_i`

- Open3D TSDF integration receives world-to-camera:

  `T_camera_world = inverse(T_world_camera)`

## Pipeline

1. Load and sort matching `angle_*.json` and `angle_*.npz` captures. Accept
   either the legacy flat directory or ordered `pass_*_height_*mm` directories.
2. Build one analytic camera-to-world pose per saved motor angle using the
   radius, height, angle direction, angle offset, camera mounting yaw, and
   pivot offset.
3. Apply the same depth and world-space filters used by
   `fuse_tsdf_scan.py`.
4. Create a downsampled local camera cloud with normals for every frame.
5. Register each adjacent pair within the same height pass with point-to-plane
   ICP initialized by the selected relative pose.
6. Add a high-weight analytic edge for every adjacent capture so the optimized
   graph retains the measured orbit. Add accepted ICP results as uncertain
   auxiliary edges rather than replacing the mechanical constraints.
7. Reject an ICP result when its fitness/RMSE is poor or its correction from
   the analytic prior exceeds configured translation/rotation limits. A
   rejected result is recorded but is not added to the graph.
8. Add a first-to-last loop edge for every height pass. For multilevel data,
   connect adjacent heights using one-to-one nearest radial camera positions
   from saved VSLAM geometry; repeated relative angle labels are not assumed to
   identify the same physical viewpoint after motor runout.
9. Optimize the pose graph with node 0 fixed.
10. Rebuild the TSDF using the optimized camera-to-world node poses.
11. Apply configurable statistical outlier removal to the extracted point
    cloud. Leave the TSDF mesh unchanged to avoid creating mesh holes.
12. Save mesh, point cloud, pose graph, poses, and per-edge diagnostics.

## Command-line interface

The new entry point is `scripts/fuse_orbit_posegraph.py`. It accepts the
existing orbit, crop, depth, and TSDF calibration options plus:

- ICP voxel size and coarse/fine correspondence distances.
- Minimum fitness and maximum RMSE.
- Maximum translation and rotation correction away from the mechanical prior.
- Mechanical-orbit edge weight relative to auxiliary ICP edges.
- Constant camera mounting yaw relative to the inward radial direction.
- Statistical outlier neighbor count, standard-deviation ratio, and an option
  to disable the filter.
- Loop-closure enable/disable.
- Pose graph, pose array, and diagnostics output paths.

Defaults target a small desktop object at roughly 0.192 m camera radius. Users
must still set world crops for their physical object; ICP cannot distinguish
the mascot from static background without a region of interest.

## Failure behavior

- Missing captures, mismatched RGB-D dimensions, invalid calibration, or fewer
  than two frames fail with a clear message.
- Too-small registration clouds do not abort the scan. Their sequential edge
  falls back to the analytic orbit and records the reason.
- A rejected loop edge is omitted.
- Existing captures and the original `fuse_tsdf_scan.py` are never modified.
- Multilevel fusion fails clearly when its top-level session lacks the fixed
  object center or up vector needed for geometry-based cross-height pairing.

## Verification

- Unit tests prove analytic source-to-target transform direction.
- Unit tests prove correction magnitude and ICP acceptance thresholds.
- The script passes Python bytecode compilation and CLI help checks.
- A real reconstruction run writes all requested artifacts and diagnostics.

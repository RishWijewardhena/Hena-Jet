# Open3D TSDF Integration Explained

Reference: <https://www.open3d.org/docs/latest/tutorial/t_reconstruction_system/integration.html>

This note explains the Open3D TSDF integration in the context of the Orbbec ArUco scanner in this folder.

## What TSDF Integration Does

TSDF means **Truncated Signed Distance Function**.

Instead of directly stacking raw point clouds, Open3D builds a 3D voxel volume. Each voxel stores:
- distance to nearest observed surface
- confidence / weight
- color

For each Orbbec RGB-D frame, Open3D uses:
- RGB image
- Depth image (aligned to color and converted to meters)
- Camera intrinsics
- 6-DoF camera pose (from ArUco table board PnP)

Then it projects that depth observation into the 3D voxel volume and updates voxels near the observed surface.

The high-level flow is:

```text
RGB-D frame + camera intrinsics + camera pose
        |
        v
project depth into voxel grid
        |
        v
update TSDF values with weighted averaging
        |
        v
repeat for many frames
        |
        v
extract mesh or point cloud
```

This weighted averaging is why TSDF usually gives smoother output than simply merging raw point clouds.

## How This Maps to `orbbec_aruco_tsdf_scan.py`

The Orbbec ArUco scanner executes:

```python
rgbd = make_rgbd(color_rgb, cleaned_depth, args.max_depth_m)
volume.integrate(rgbd, open3d_intrinsic, world_to_camera)
```

The meanings are:
- `rgbd`: Orbbec RGB image + aligned, range-filtered, marker-masked depth image
- `open3d_intrinsic`: `fx, fy, cx, cy` from the Orbbec RGB camera parameter
- `world_to_camera`: 6-DoF camera pose estimated from the ArUco table board
- `volume.integrate`: fuse this frame into the Open3D TSDF volume

The most important input is `world_to_camera`. When `world_to_camera` is accurate, multi-view depth frames fuse seamlessly in global table space.

## Mesh and Point Cloud Extraction

After all frames are integrated, Open3D extracts:

```python
mesh = volume.extract_triangle_mesh()
cloud = volume.extract_point_cloud()
```

Mesh extraction uses marching cubes to create triangle faces from the zero-crossings of the TSDF volume. Point cloud extraction extracts voxel vertices with normals and colors.

Outputs:
```text
orbbec_aruco_tsdf_mesh.ply
orbbec_aruco_tsdf_cloud.ply
```

## Important Parameters

- `--voxel-length-m`: Voxel size in meters (e.g. `0.0015` for 1.5 mm).
- `--sdf-trunc-m`: Truncation distance for signed distance averaging (e.g. `0.008` for 8 mm).
- `--min-depth-m` / `--max-depth-m`: Depth bounding range.
- `--marker-mask-padding-px`: Masks pixels belonging to ArUco markers so they do not get integrated into the mesh.

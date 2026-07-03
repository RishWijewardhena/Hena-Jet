# Open3D TSDF Integration Explained

Reference: <https://www.open3d.org/docs/latest/tutorial/t_reconstruction_system/integration.html>

This note explains the Open3D TSDF integration tutorial in the context of the
ZED-M ArUco scanner in this folder.

## What TSDF Integration Does

TSDF means **Truncated Signed Distance Function**.

Instead of directly stacking raw point clouds, Open3D builds a 3D voxel volume.
Each voxel stores information similar to:

```text
distance to nearest observed surface
confidence / weight
color
```

For each ZED RGB-D frame, Open3D uses:

```text
RGB image
depth image
camera intrinsics
camera pose
```

Then it projects that depth observation into the 3D voxel volume and updates
voxels near the observed surface.

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

This weighted averaging is why TSDF usually gives smoother output than simply
merging raw point clouds.

## Activation and Integration

Open3D describes TSDF processing in two main stages.

### Activation

Open3D first decides which voxel blocks are relevant for the current RGB-D
frame. It does not allocate or process an infinite 3D grid. It activates only
the blocks that are inside the current camera view and near observed depth.

This keeps reconstruction practical for larger scenes.

### Integration

For each active voxel, Open3D projects the voxel into the depth image and
compares:

```text
voxel's expected depth from the camera
measured depth image value
```

If the voxel is close enough to the measured surface, Open3D updates the TSDF
value using a weighted average. If color integration is enabled, color is also
updated.

## How This Maps to `zed_aruco_tsdf_scan.py`

The ZED ArUco scanner does this:

```python
rgbd = make_rgbd(color_for_integration, depth, args.max_depth_m)
volume.integrate(rgbd, open3d_intrinsic, world_to_camera)
```

The meanings are:

```text
rgbd             = ZED RGB image + cleaned ZED depth image
open3d_intrinsic = fx, fy, cx, cy from the ZED left camera
world_to_camera  = camera pose estimated from the ArUco table board
volume.integrate = fuse this frame into the TSDF volume
```

The most important input is `world_to_camera`.

If `world_to_camera` is accurate, depth frames overlap correctly in the TSDF
volume. If it is wrong, Open3D will average surfaces in slightly different
places. That creates thick, rough, or doubled mesh surfaces.

## Mesh and Point Cloud Extraction

After all frames are integrated, Open3D can extract either:

```python
mesh = volume.extract_triangle_mesh()
cloud = volume.extract_point_cloud()
```

The mesh extraction uses marching cubes to create triangle faces from the TSDF
surface. The point cloud extraction uses a related surface extraction process
but skips triangle face generation.

So these outputs:

```text
hand_mesh.ply
hand_cloud.ply
```

are extracted from the fused TSDF volume. They are not direct dumps of the raw
ZED point clouds.

## Why Noise Still Appears

TSDF reduces random depth noise, but it does not know which object should be
kept. If valid depth contains the table, paper, ArUco markers, cardboard, tools,
or background, those surfaces can also be fused.

So TSDF helps with smoothing repeated observations, but it does not replace
scene cleanup.

For clean hand scans, the pipeline still needs:

```text
depth range filtering
ROI crop
marker masking
table-plane removal
world-space crop
outlier removal
mesh cleanup
```

## Important Parameters

These scanner parameters directly affect TSDF behavior:

```bash
--voxel-length-m
```

Voxel size. Smaller voxels give denser geometry, but they preserve more ZED
depth noise and pose jitter.

```bash
--sdf-trunc-m
```

Distance around the observed surface used for TSDF averaging. A larger value
smooths more but can thicken surfaces. A smaller value keeps sharper detail but
can fragment surfaces if pose or depth is noisy.

Practical starting pairs:

```text
balanced:      --voxel-length-m 0.0015 --sdf-trunc-m 0.008
more detail:   --voxel-length-m 0.0012 --sdf-trunc-m 0.007
high detail:   --voxel-length-m 0.0010 --sdf-trunc-m 0.006
noise-prone:   --voxel-length-m 0.0008 --sdf-trunc-m 0.005
```

If smaller voxels make the mesh worse, the limiting problem is probably pose
error, depth noise, or unwanted scene geometry, not TSDF resolution.

## Main Takeaway

Open3D TSDF integration fuses many posed RGB-D frames into one voxel volume
using weighted averaging. The final mesh or point cloud is extracted from that
fused volume.

For this scanner, the quality depends mostly on:

- accurate ArUco camera pose for every frame
- clean depth before fusion
- good voxel/truncation settings
- removing table/background/outlier geometry before or after TSDF

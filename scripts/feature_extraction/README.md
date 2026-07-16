# RGB-D Feature Tracking Baseline

This folder is for markerless tracking experiments before connecting the result
to TSDF fusion.

## Main Pipeline

Use this one-command pipeline for the current LightGlue scanner flow:

```bash
/home/rishmika/miniconda3/envs/hena_jet/bin/python \
  scripts/feature_extraction/run_lightglue_posegraph_pipeline.py \
  --overwrite
```

It runs:

```text
ZED RGB-D capture -> LightGlue tracking -> pose graph loop closure -> TSDF fusion
```

Defaults:

```text
depth mode: NEURAL
depth range: 0.10m..0.20m
frames: 80
device: cuda
SuperPoint keypoints: 4096
LightGlue filter threshold: 0.05
final outputs:
  outputs/feature_tracking/lightglue_pipeline_10_20cm_cloud.ply
  outputs/feature_tracking/lightglue_pipeline_10_20cm_mesh.ply
```

To process an already captured dataset without opening the camera:

```bash
/home/rishmika/miniconda3/envs/hena_jet/bin/python \
  scripts/feature_extraction/run_lightglue_posegraph_pipeline.py \
  --skip-capture
```

To preview the commands without running them:

```bash
/home/rishmika/miniconda3/envs/hena_jet/bin/python \
  scripts/feature_extraction/run_lightglue_posegraph_pipeline.py \
  --dry-run --skip-capture
```

For small or weakly textured geometry such as fingers, run the same wrapper with
more keypoints and a lower LightGlue threshold:

```bash
/home/rishmika/miniconda3/envs/hena_jet/bin/python \
  scripts/feature_extraction/run_lightglue_posegraph_pipeline.py \
  --skip-capture \
  --max-keypoints 8192 \
  --filter-threshold 0.03 \
  --min-depth-matches 12 \
  --min-inliers 8
```

This accepts more tentative RGB matches, then still uses depth RANSAC to reject
geometrically bad matches. If the result becomes unstable, raise
`--filter-threshold` back toward `0.05` or `0.10`.

## Stage 1: ORB/SIFT + Depth RANSAC

`rgbd_feature_ransac_tracker.py` reads an Open3D-style ZED dataset:

```text
dataset/
  image/*.png
  depth/*.png
  intrinsic.json
  capture_config.json
```

For each candidate frame it:

1. detects ORB or SIFT features in RGB,
2. matches features to the last accepted frame,
3. lifts matched pixels to 3D using ZED depth,
4. estimates a rigid transform with RANSAC,
5. rejects weak/jumpy frames,
6. writes accepted camera-to-world poses and a diagnostics report.

Run a small test first:

```bash
/home/rishmika/miniconda3/envs/hena_jet/bin/python \
  scripts/feature_extraction/rgbd_feature_ransac_tracker.py \
  --dataset datasets/zed_m_open3d \
  --max-frames 20 \
  --method orb
```

Try SIFT if ORB gives too few stable matches:

```bash
/home/rishmika/miniconda3/envs/hena_jet/bin/python \
  scripts/feature_extraction/rgbd_feature_ransac_tracker.py \
  --dataset datasets/zed_m_open3d \
  --max-frames 20 \
  --method sift
```

Outputs default to:

```text
outputs/feature_tracking/rgbd_feature_poses.npy
outputs/feature_tracking/rgbd_feature_report.json
outputs/feature_tracking/debug_matches/*.jpg
```

Fuse the accepted frames into a final cloud and mesh:

```bash
/home/rishmika/miniconda3/envs/hena_jet/bin/python \
  scripts/feature_extraction/fuse_feature_tracked_rgbd.py \
  --dataset datasets/zed_m_open3d
```

Fusion outputs default to:

```text
outputs/feature_tracking/feature_tracked_cloud.ply
outputs/feature_tracking/feature_tracked_mesh.ply
```

## How To Read The Result

Useful report fields:

| Field | Meaning |
|---|---|
| `raw_matches` | Feature matches after descriptor matching and ratio test. |
| `depth_valid_matches` | Matches where both pixels had usable depth. |
| `inliers` | Matches that agreed with one rigid 3D transform. |
| `inlier_ratio` | `inliers / depth_valid_matches`; higher is better. |
| `translation_m`, `rotation_deg` | Estimated motion from candidate frame to last accepted frame. |
| `reason` | `accepted` or the rejection reason. |

If many frames are rejected with `too_few_depth_valid_matches`, the scene needs
more texture, better depth, slower motion, or a less strict depth range.

If many frames are rejected with `low_inlier_ratio`, the feature matches are not
geometrically consistent. Try SIFT, better lighting, or a textured scan mat.

If frames are rejected with `translation_jump` or `rotation_jump`, the camera is
moving too quickly or the estimated transform is wrong.

## Current Limit

This script only estimates poses and diagnostics. It does not yet fuse frames
into a mesh/cloud. Use `fuse_feature_tracked_rgbd.py` for a first TSDF fusion
from those poses. The next improvement is ICP refinement before TSDF integration.

## Stage 2: SuperPoint + LightGlue + Depth RANSAC

`superpoint_lightglue_tracker.py` replaces ORB/SIFT matching with learned
SuperPoint keypoints and LightGlue matching, then uses the same ZED-depth RANSAC
geometry check.

This needs PyTorch and LightGlue. The current `hena_jet` environment has been
tested with CUDA PyTorch and LightGlue on the RTX 2050. LightGlue's official
install path is:

```bash
git clone https://github.com/cvg/LightGlue.git
cd LightGlue
python -m pip install -e .
```

On Jetson, install the correct NVIDIA-compatible PyTorch build before installing
LightGlue.

Run learned tracking on GPU:

```bash
/home/rishmika/miniconda3/envs/hena_jet/bin/python \
  scripts/feature_extraction/superpoint_lightglue_tracker.py \
  --dataset datasets/zed_m_open3d \
  --max-frames 80 \
  --device cuda \
  --debug-pairs 80 \
  --max-step-rotation-deg 30
```

Fuse the learned-tracked result:

```bash
/home/rishmika/miniconda3/envs/hena_jet/bin/python \
  scripts/feature_extraction/fuse_feature_tracked_rgbd.py \
  --dataset datasets/zed_m_open3d \
  --report outputs/feature_tracking/lightglue_feature_report.json \
  --poses outputs/feature_tracking/lightglue_feature_poses.npy \
  --cloud-out outputs/feature_tracking/lightglue_feature_tracked_cloud.ply \
  --mesh-out outputs/feature_tracking/lightglue_feature_tracked_mesh.ply
```

Official LightGlue source: <https://github.com/cvg/LightGlue>

## Stage 3: Local-Map ICP Pose Refinement

`refine_feature_poses_icp.py` uses the Stage 2 poses as the initial estimate,
then refines each frame against a rolling local depth map with point-to-plane
ICP. Corrections are accepted only when the ICP overlap is good and the
correction is small.

Run ICP refinement on the 10-20 cm LightGlue result:

```bash
/home/rishmika/miniconda3/envs/hena_jet/bin/python \
  scripts/feature_extraction/refine_feature_poses_icp.py \
  --dataset datasets/zed_m_open3d \
  --report outputs/feature_tracking/lightglue_feature_report_10_20cm.json \
  --poses outputs/feature_tracking/lightglue_feature_poses_10_20cm.npy \
  --depth-min-m 0.10 \
  --depth-max-m 0.20
```

Fuse the ICP-refined poses:

```bash
/home/rishmika/miniconda3/envs/hena_jet/bin/python \
  scripts/feature_extraction/fuse_feature_tracked_rgbd.py \
  --dataset datasets/zed_m_open3d \
  --report outputs/feature_tracking/lightglue_icp_refined_report_10_20cm.json \
  --poses outputs/feature_tracking/lightglue_icp_refined_poses_10_20cm.npy \
  --cloud-out outputs/feature_tracking/lightglue_icp_refined_cloud_10_20cm.ply \
  --mesh-out outputs/feature_tracking/lightglue_icp_refined_mesh_10_20cm.ply \
  --depth-min-m 0.10 \
  --depth-max-m 0.20 \
  --voxel-length-m 0.0015 \
  --sdf-trunc-m 0.006
```

## Stage 4: Pose Graph Loop Closure

`optimize_feature_pose_graph.py` builds an Open3D pose graph from the Stage 2
or Stage 3 poses. It adds odometry edges between neighboring frames and guarded
loop-closure edges between non-neighbor frames that are close in the current
pose estimate and pass depth-ICP validation.

Run pose graph optimization from the 10-20 cm LightGlue poses:

```bash
/home/rishmika/miniconda3/envs/hena_jet/bin/python \
  scripts/feature_extraction/optimize_feature_pose_graph.py \
  --dataset datasets/zed_m_open3d \
  --report outputs/feature_tracking/lightglue_feature_report_10_20cm.json \
  --poses outputs/feature_tracking/lightglue_feature_poses_10_20cm.npy \
  --depth-min-m 0.10 \
  --depth-max-m 0.20
```

Fuse the optimized poses:

```bash
/home/rishmika/miniconda3/envs/hena_jet/bin/python \
  scripts/feature_extraction/fuse_feature_tracked_rgbd.py \
  --dataset datasets/zed_m_open3d \
  --report outputs/feature_tracking/lightglue_posegraph_report_10_20cm.json \
  --poses outputs/feature_tracking/lightglue_posegraph_poses_10_20cm.npy \
  --cloud-out outputs/feature_tracking/lightglue_posegraph_cloud_10_20cm.ply \
  --mesh-out outputs/feature_tracking/lightglue_posegraph_mesh_10_20cm.ply \
  --depth-min-m 0.10 \
  --depth-max-m 0.20 \
  --voxel-length-m 0.0015 \
  --sdf-trunc-m 0.006
```

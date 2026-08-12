# Hena Jet

Automated mehendi drawing robot project.

## Goal

Build a robotic system that scans a human hand or arm, reconstructs the skin surface as a 3D point cloud or mesh, maps a mehendi pattern onto that surface, and draws the pattern accurately with a robot mechanism.

## Core Concept

The hand stays fixed inside a hollow cylindrical scanner while the depth camera rotates around it.

This is the preferred concept because the hand does not move during scanning, and the camera pose can be controlled mechanically using motors and encoders.

## Workflow

1. Place the hand inside the cylindrical scanner.
2. Rotate the depth camera around the stationary hand.
3. Capture RGB and depth frames at known angles.
4. Read camera angle from a rotary encoder.
5. Convert each depth frame into a point cloud.
6. Transform each point cloud using the camera pose.
7. Merge all point clouds into one hand model.
8. Clean and filter the point cloud.
9. Convert the point cloud into a mesh.
10. Map the mehendi pattern onto the hand surface.
11. Generate the drawing path.
12. Draw the mehendi path with the robot.

## Current Camera Plan

### Immediate feasibility camera

Use the already installed ZED SDK and test with the available ZED-M camera first.

The purpose of this phase is not to finalize the camera choice. The purpose is to prove that:

- A rotating-camera, fixed-hand scanner can produce usable geometry.
- Camera poses from the mechanical rig can be used to merge frames.
- The point cloud quality is good enough for path planning experiments.

### Target prototype camera

The current target camera for the next hardware step is the Orbbec Gemini 305.

Reasons:

- Suitable close working range for the scanner concept.
- USB-C connection.
- Small body for a rotating mount.
- SDK support for depth, RGB, point clouds, Python, C++, and ROS2.
- Good fit for close-range robotic scanning.

Intel RealSense D405 remains a strong alternative. ZED X Nano is technically capable, but it adds GMSL2/ZED Link/Jetson complexity that is not ideal for the first prototype.

## Pose Strategy

Do not depend on camera IMU or gyro data for scan alignment.

Use mechanical pose tracking instead:

- Rotary encoder gives camera angle `theta`.
- Linear encoder or motor position gives camera height `z`, if vertical motion is added.
- Known scanner radius gives the camera position around the hand.

This is better for this scanner than IMU-based tracking because encoder-based motion is repeatable and does not drift.

## First Milestone

Validate the concept with ZED-M:

1. Confirm ZED SDK can stream RGB, depth, and point cloud data.
2. Capture one static frame of a hand or test object.
3. Rotate the camera manually or with a temporary mount.
4. Save depth/point cloud frames at several known angles.
5. Merge the frames using known circular poses.
6. Inspect the merged point cloud in Open3D.

See [docs/zed_m_feasibility.md](docs/zed_m_feasibility.md) for the checklist.
See [docs/transform_clouds.md](docs/transform_clouds.md) for the complete
known-angle transformation, progressive ICP, cleanup, and meshing pipeline.

## Reliable Close-Range KISS-ICP Scan

The real-time scanner is tuned for a ZED Mini moving slowly around a stationary
hand at a working distance of approximately 10–20 cm:

```bash
python scripts/kiss_icp_realtime.py \
  --out outputs/kiss_icp_hand_map.ply
```

The default configuration uses `NEURAL_PLUS`, ZED confidence filtering, an
11–22 cm reliable software depth window, deterministic 3 mm ICP sampling, and a
frame-weighted 2 mm global map. Frames with sparse, low-coverage, degenerate, or
excessively large pose changes are not fused.

Stopping the scan saves three files:

- `kiss_icp_hand_map.ply`: globally fused voxels observed in at least two frames.
- `kiss_icp_hand_map_trajectory.npy`: accepted KISS-ICP poses only.
- `kiss_icp_hand_map_quality.json`: settings, versions, frame rejection reasons,
  depth coverage, pose steps, map statistics, and trajectory closure error.

The active development Python environment currently has ZED SDK 5.3.1 but does
not have the `kiss-icp` or `open3d` Python packages. Run the live scanner in the
camera environment where those packages are installed. ZED depth confidence
values near 100 are least trustworthy; the default threshold rejects values
above 60 while leaving texture confidence at 100 to preserve low-texture skin.
See the Stereolabs documentation for
[confidence filtering](https://www.stereolabs.com/docs/depth-sensing/confidence-filtering)
and [depth modes](https://docs.stereolabs.com/docs/development/zed-sdk/modules/depth-sensing/depth-modes).

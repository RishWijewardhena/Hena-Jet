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

## Active Gemini 305 Scan Workflow

The active capture and reconstruction path is the synchronized circular-scan
workflow in [`scripts/sync_workflow/`](scripts/sync_workflow/README.md). It uses
the calibrated motor pose as the alignment prior, with guarded ICP available for
small, validated corrections.

```bash
conda activate hena_jet
python scripts/sync_workflow/main_scan.py --reconstruct
```

Reconstruction defaults to a cylindrical 85 mm radial crop with a 300 mm axial
half-length, guarded ICP, and `--fusion both`. It writes a cleaned point cloud,
the raw TSDF mesh (`tsdf_mesh.ply`) for inspection, and the cleaned planning
surface (`tsdf_mesh_cleaned.ply`). The cleaned mesh removes small detached
fragments, repairs only tiny holes, and applies non-shrinking Taubin smoothing;
it does not intentionally bridge finger gaps or large openings.

See [`scripts/sync_workflow/WORKFLOW.md`](scripts/sync_workflow/WORKFLOW.md) for
the calibration, capture, registration, and TSDF-fusion flow charts.

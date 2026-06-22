# ZED-M Feasibility Checklist

This phase checks whether the rotating-camera scanner concept is practical before buying or switching to the Orbbec Gemini 305.

## Questions To Answer

- Can the ZED-M produce usable depth at the scanner's expected hand distance?
- Is the point cloud dense enough for hand surface reconstruction?
- Does the depth quality remain stable on skin-like surfaces?
- Can multiple views be merged when the camera pose is supplied by the mechanical rig?
- Is the merged geometry accurate enough to support mehendi path projection?

## Test Setup

Minimum setup:

- ZED-M camera.
- Installed ZED SDK.
- Laptop or Jetson that can run the ZED SDK.
- Simple rotating mount or manual angle marks.
- Rigid test object first, then hand/arm test.

Recommended first scan object:

- Start with a rigid object similar in size to a hand.
- Move to a real hand only after the capture and merge pipeline works.

## Capture Plan

Capture frames around the object at fixed angle steps:

- 0 degrees
- 30 degrees
- 60 degrees
- 90 degrees
- 120 degrees
- 150 degrees
- 180 degrees
- 210 degrees
- 240 degrees
- 270 degrees
- 300 degrees
- 330 degrees

For each angle, save:

- RGB image.
- Depth image.
- Point cloud.
- Camera angle.
- Camera radius from rotation center.
- Camera height.

## Basic Geometry Model

Assume the camera moves around the object on a horizontal circle.

Known values:

- `theta`: camera angle from rotary encoder or manual angle mark.
- `r`: radius from rotation center to camera.
- `z`: camera height.

For the first prototype, use one fixed height and one circular scan path.

## Processing Pipeline

1. Open ZED camera stream.
2. Read RGB and depth.
3. Export per-frame point cloud.
4. Remove invalid depth points.
5. Transform each point cloud from camera coordinates into scanner/world coordinates.
6. Merge transformed point clouds.
7. Downsample the merged cloud.
8. Remove outliers.
9. Visualize in Open3D.
10. Save merged output as `.ply`.

## Success Criteria

The ZED-M concept test is successful if:

- A complete enough object/hand point cloud can be produced from multiple views.
- The merged cloud has recognizable hand or test-object geometry.
- The alignment error is small enough to support surface projection experiments.
- The scanning workflow is repeatable.

The test fails if:

- Depth is missing or too noisy at the required close distance.
- The point cloud is too sparse for pattern projection.
- Multi-view alignment is unreliable even with known poses.

## Next Step After Success

After the ZED-M concept test works:

1. Build the camera pose transform code cleanly.
2. Add encoder input for real camera angle.
3. Add mesh reconstruction.
4. Test simple pattern projection.
5. Re-evaluate camera choice and move to Orbbec Gemini 305 if close-range performance or form factor requires it.

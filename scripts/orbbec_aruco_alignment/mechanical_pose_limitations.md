# Mechanical-Only Camera Pose Limitations

This note explains the limitations of aligning RGB-D point clouds using only the scanner rig's mechanical pose versus the ArUco marker vision-based pose estimation in this folder.

## Minimum Mechanical Pose Inputs

Angle alone is not enough. A circular mechanical pose model needs at least:

- `angle_deg`: scanner rotation angle for the frame.
- `radius_m`: distance from scanner center to the camera reference point.
- `height_m`: camera height above the table/world reference plane.
- `camera_yaw_offset_deg`: correction for the camera not pointing exactly at the scanner center.
- `camera_pitch_deg`: downward camera tilt toward the table.
- `camera_roll_deg`: roll error from the camera mount.
- `center_offset_x_m` and `center_offset_y_m`: offset between the assumed scanner center and the true object/table center.
- Pivot/optical-center offset: whether `radius_m` is measured to the optical sensor center or the mechanical bracket pivot.

## Main Limitations of Mechanical-Only Pose

### Small angle errors become millimeter-level mesh errors
At radius `0.18 m`, a `1 degree` angle error creates about:
```text
0.18 * sin(1 deg) = 0.0031 m = 3.1 mm
```
That is larger than the `1 mm` to `2 mm` scan target resolution, resulting in smeared surfaces after fusion.

### Radius error shifts every frame
If the radius estimate is off by `5 mm`, every frame is shifted by `5 mm`. TSDF fusion averages inconsistent surfaces, producing rough or doubled layers.

### Camera tilt and mounting errors are difficult to calibrate physically
Downwards camera tilt (pitch) and bracket roll produce noticeable misalignments around thin object features.

### Open-Loop vs Closed-Loop
Mechanical-only pose assumes rigid mechanics with zero backlash, slipping, or vibration. ArUco-based PnP pose provides closed-loop optical measurement on every single frame with explicit reprojection error validation.

### Optical Center Offset
Depth and RGB cameras measure geometry from the sensor optical frame. Mechanical rigs rotate around physical pivot joints. ArUco marker pose directly solves the optical frame transform in one step without needing joint calibration.

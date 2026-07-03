# Mechanical-Only Camera Pose Limitations

This note explains the limitations of aligning ZED-M point clouds using only
the scanner's mechanical pose. The current ArUco workflow estimates camera pose
from visible markers with known table positions. A mechanical-only workflow
would instead compute each camera pose from the rig geometry and the commanded
scanner angle.

## Minimum Mechanical Pose Inputs

Angle alone is not enough. A circular mechanical pose model needs at least:

- `angle_deg`: scanner rotation angle for the frame.
- `radius_m`: distance from scanner center to the camera reference point.
- `height_m`: camera height above the table/world reference plane.
- `camera_yaw_offset_deg`: correction for the camera not pointing exactly at
  the scanner center.
- `camera_pitch_deg`: downward camera tilt toward the table.
- `camera_roll_deg`: roll error from the camera mount.
- `center_offset_x_m` and `center_offset_y_m`: offset between the assumed
  scanner center and the true object/table center.
- Pivot/optical-center offset: whether `radius_m` is measured to the left lens,
  right lens, camera body center, or mechanical bracket pivot.

For a simple circular rig, the camera position is:

```text
x = center_x + radius * cos(angle)
y = center_y + radius * sin(angle)
z = height
```

That gives only position. Point-cloud and TSDF alignment also need camera
orientation: yaw, pitch, and roll.

## Main Limitations

### Small angle errors become millimeter-level mesh errors

At radius `0.18 m`, a `1 degree` angle error creates about:

```text
0.18 * sin(1 deg) = 0.0031 m = 3.1 mm
```

That is already larger than a `1 mm` to `2 mm` scan target resolution. The
result is thick, doubled, or smeared surfaces after fusion.

### Radius error shifts every frame

If the radius is wrong by `5 mm`, every frame is placed about `5 mm` from its
true position. TSDF fusion will average inconsistent surfaces and produce a
rough mesh instead of one clean hand/object surface.

### Camera tilt and mounting errors are hard to measure

The ZED-M is tilted downward in the scanner setup. If the pitch estimate is
wrong, every point cloud is rotated incorrectly. Small yaw, pitch, and roll
errors are especially visible around fingers, object edges, and the table
plane.

### Mechanical-only pose is open-loop

A mechanical-only script assumes the rig perfectly matches the calibration for
every frame. It cannot detect:

- vibration
- bracket flex
- backlash
- slipping
- wrong zero angle
- camera mount movement
- object or table movement

With ArUco pose, each frame is measured from the actual RGB image. The script
can also use reprojection error to identify weak poses.

### Optical center and mechanical pivot may be different

Depth is measured from the camera optical frame, but the mechanical system may
rotate around a bracket pivot, camera body center, left lens, right lens, or
another physical point. If this offset is not modeled, all frames are
misregistered even when the angle is correct.

For ZED-M this is important because the left camera optical frame, stereo
baseline midpoint, and physical bracket pivot are not automatically the same
point.

### No per-frame confidence score

ArUco pose estimation provides observable signals such as visible marker count
and reprojection error. Mechanical-only pose does not provide an equivalent
per-frame confidence check. A bad mechanical frame can be fused as if it were
correct.

### Alignment does not solve scene cleanup

Mechanical pose only places frames in a shared coordinate system. It does not
remove the table, paper, ArUco markers, cardboard, tools, or background. A
mechanical-only pipeline still needs depth filtering, world cropping, table
plane removal, and outlier cleanup.

## Comparison With ArUco Pose

| Approach | Strength | Limitation |
|---|---|---|
| ArUco pose | Measures camera pose from each RGB frame and gives reprojection error. | Requires visible markers and a correctly measured marker board. |
| Mechanical pose | Works without visible markers and can run from known motor angles. | Requires accurate calibration of radius, height, tilt, yaw, roll, center offset, and pivot offset. |

## Practical Guidance

Use mechanical pose as an initial estimate or fallback, not as the only
alignment source for high-quality hand meshes.

For clean reconstruction, prefer the ArUco workflow when markers are visible.
If mechanical-only pose is used, calibrate the scanner carefully and verify:

- true radius to the camera optical frame
- true scanner center
- zero-angle direction
- camera pitch/yaw/roll offsets
- bracket pivot offset
- repeatability under motion

The acceptable error should be smaller than the target mesh resolution. For
`1 mm` to `2 mm` hand detail, mechanical calibration must also be near
millimeter-level accurate.

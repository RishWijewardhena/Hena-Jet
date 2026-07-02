# ZED-M ArUco Table-Board Alignment

This folder contains an ArUco-marker based alignment workflow for ZED-M RGB-D
scanning. The goal is to solve the hardest part of multi-view scanning:
knowing the camera pose for every RGB-D frame so that different depth frames
land in the same 3D world coordinate system.

Instead of estimating motion from the object surface, this method uses a printed
table-top ArUco board. The table board is fixed, flat, and known in metric
coordinates. Every time the camera captures an RGB image, the script detects the
visible marker corners and computes the camera pose from those known points.

## Files

| File | Purpose |
|---|---|
| `generate_aruco_table_board.py` | Generates the printable ArUco board PDF/PNG, individual marker images, and the JSON file containing marker positions. |
| `zed_aruco_pose_debug.py` | Opens the ZED-M, detects markers, solves pose, and saves debug overlays without doing TSDF fusion. Use this first. |
| `zed_aruco_tsdf_scan.py` | Captures ZED-M RGB-D frames, solves camera pose from markers, and integrates frames into an Open3D TSDF mesh/cloud. |
| `aruco_common.py` | Shared marker detection, pose solving, depth cleanup, and Open3D helper functions. |

Generated files default to:

```text
outputs/aruco_table_board/aruco_table_board.pdf
outputs/aruco_table_board/aruco_table_board.png
outputs/aruco_table_board/aruco_table_board.json
outputs/aruco_table_board/markers/marker_00.png ... marker_11.png
```

## Board JSON

`aruco_table_board.json` is the source of truth for the printed board geometry.
It tells the scanner where each marker is on the table.

Current generated defaults:

```json
{
  "dictionary": "DICT_5X5_100",
  "marker_size_m": 0.025,
  "marker_count": 12,
  "paper": "a4",
  "dpi": 300
}
```

Important fields:

| Field | Meaning |
|---|---|
| `dictionary` | The OpenCV ArUco dictionary used to generate and detect markers. Current default is `DICT_5X5_100`. |
| `marker_size_m` | Physical marker side length in meters. `0.025` means `25 mm`. |
| `marker_count` | Number of generated markers. Current board uses 12 markers with IDs `0-11`. |
| `world_frame` | Defines the table/world coordinate frame used by pose estimation and reconstruction. |
| `markers` | List of marker IDs, centers, and 3D marker corner coordinates. |

The board world frame is:

```text
origin: center of the printed scan area
+X: right on printed board
+Y: up on printed board
+Z: out of table plane toward the camera
units: meters
```

All markers lie on the table plane, so every marker corner has `Z = 0`.

Example marker:

```json
{
  "id": 0,
  "center_m": [-0.06, 0.11, 0.0],
  "corners_m": [
    [-0.0725, 0.1225, 0.0],
    [-0.0475, 0.1225, 0.0],
    [-0.0475, 0.0975, 0.0],
    [-0.0725, 0.0975, 0.0]
  ]
}
```

This means marker `0` is centered 60 mm left and 110 mm up from the board
origin. Because the marker is 25 mm wide, each corner is 12.5 mm from the
center. The corner order matches OpenCV's detected marker corner order:

```text
top-left, top-right, bottom-right, bottom-left
```

## Why This Solves Alignment

Every RGB-D frame starts in the camera coordinate system. To merge many frames,
the scanner must know how each camera coordinate system sits relative to one
shared world coordinate system.

With this method:

1. The printed table board defines the world coordinate system.
2. The JSON gives exact 3D coordinates for marker corners on that board.
3. The ZED RGB image gives 2D pixel coordinates for the same marker corners.
4. `cv2.solvePnP()` computes the transform from world/table coordinates to the
   camera coordinates.
5. Open3D TSDF integration uses that transform as the frame extrinsic.

So each captured depth frame is placed into the same table/world coordinate
system. This is why point clouds from different viewpoints can overlap.

## Pose Calculation

For every accepted capture, the script builds two matched arrays.

The first array is 3D points from the JSON:

```text
object_points = [
  [marker_0_corner_0_x, marker_0_corner_0_y, 0],
  [marker_0_corner_1_x, marker_0_corner_1_y, 0],
  ...
]
```

The second array is 2D pixels detected in the ZED RGB image:

```text
image_points = [
  [marker_0_corner_0_u, marker_0_corner_0_v],
  [marker_0_corner_1_u, marker_0_corner_1_v],
  ...
]
```

The ZED left-camera intrinsics are read from the ZED SDK:

```text
fx, fy, cx, cy, width, height
```

Those values form the camera matrix:

```text
[ fx   0  cx ]
[  0  fy  cy ]
[  0   0   1 ]
```

Then the script calls:

```python
ok, rvec, tvec = cv2.solvePnP(
    object_points,
    image_points,
    camera_matrix,
    dist_coeffs,
    flags=cv2.SOLVEPNP_ITERATIVE,
)
```

`rvec` and `tvec` describe the transform:

```text
world/table -> camera
```

The script converts `rvec` to a rotation matrix using `cv2.Rodrigues()` and
builds a 4x4 matrix:

```text
world_to_camera =
[ R00 R01 R02 tx ]
[ R10 R11 R12 ty ]
[ R20 R21 R22 tz ]
[  0   0   0  1 ]
```

The reprojection error is also computed. It projects the known 3D marker
corners back into the image and compares them to the detected 2D corners. Lower
error is better. A low error means the marker layout, print scale, and camera
intrinsics are agreeing.

## RGB, Depth, and Coordinate Frames

The ZED-M scripts use:

```python
sl.COORDINATE_SYSTEM.IMAGE
```

That convention is:

```text
+X: image right
+Y: image down
+Z: camera forward
```

This matches the OpenCV/Open3D pinhole image convention used in this workflow.

For every capture, the script gets RGB and depth from the same ZED grab cycle:

```python
zed.grab(runtime)
zed.retrieve_image(color_mat, sl.VIEW.LEFT)
zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
```

That matters. ArUco pose comes from the RGB image, and geometry comes from the
depth image. If RGB and depth are not from the same frame, alignment can be
wrong.

## TSDF Alignment

Open3D TSDF integration needs:

1. RGB-D image.
2. Camera intrinsics.
3. Extrinsic transform from world to camera.

The script already has all three:

```python
rgbd = make_rgbd(color_for_integration, depth, args.max_depth_m)
volume.integrate(rgbd, open3d_intrinsic, world_to_camera)
```

Because `world_to_camera` comes from ArUco markers, every depth frame is fused
into the same table/world coordinate system.

This is different from the earlier mechanical-circle method:

```text
old method: angle + radius + height -> approximate pose
new method: marker corners in RGB image -> measured camera pose
```

Mechanical angle is not required for alignment in this method.

## Marker Masking and ROI

Default ROI is the whole image:

```bash
--roi 0 0 1 1
```

Marker detection always uses the full RGB image. The ROI applies to
reconstruction/depth integration.

Markers are useful for pose, but they should not become part of the final scan.
So `zed_aruco_tsdf_scan.py` masks the depth pixels under detected marker
polygons before TSDF integration:

```text
detect marker -> use it for pose -> set marker depth pixels to 0 -> integrate object/table depth
```

This behavior is enabled by default. Disable it only for debugging:

```bash
--no-mask-markers
```

You can expand the marker mask:

```bash
--marker-mask-padding-px 12
```

## How To Use

Generate the board:

```bash
python3 scripts/zed_aruco_alignment/generate_aruco_table_board.py
```

Print:

```text
outputs/aruco_table_board/aruco_table_board.pdf
```

Printer settings:

```text
scale: 100%
fit to page: off
```

After printing, measure one marker side. It must be 25 mm. If it is not 25 mm,
the pose scale will be wrong.

Run pose debug first:

```bash
python3 scripts/zed_aruco_alignment/zed_aruco_pose_debug.py
```

Check:

```text
outputs/zed_aruco_debug/pose_debug_0000.png
outputs/zed_aruco_debug/pose_debug.json
```

The debug image should show marker outlines, marker IDs, and a drawn pose axis.

Run TSDF scan:

```bash
python3 scripts/zed_aruco_alignment/zed_aruco_tsdf_scan.py
```

The default depth mode is `NEURAL_PLUS` when the installed ZED SDK exposes it.
You can choose another mode explicitly:

```bash
python3 scripts/zed_aruco_alignment/zed_aruco_tsdf_scan.py --depth-mode NEURAL_LIGHT
```

The scanner will prompt:

```text
Press Enter to capture, or q to finish:
```

Move the camera to a view, press Enter, move to another view, press Enter, and
repeat. Type `q` to finish and save outputs.

Default outputs:

```text
outputs/zed_aruco_tsdf_mesh.ply
outputs/zed_aruco_tsdf_cloud.ply
outputs/zed_aruco_poses.npy
outputs/zed_aruco_scan.json
outputs/zed_aruco_debug/scan_0000.png ...
```

## Quality Checks

A good capture should show:

```text
KF 000  markers=[...]  valid_px=...  reproj=...
```

Use these checks:

| Signal | What It Means |
|---|---|
| `markers` has 2 or more IDs | Enough marker observations for pose. More is better. |
| `reproj` is low | Marker pose agrees with the image. Lower is better. |
| Debug overlay axes look attached to the board | Pose direction is plausible. |
| Camera position in JSON is stable | Board scale and intrinsics are likely correct. |

If the script says pose failed, the RGB image did not contain enough valid
markers from the board.

## Troubleshooting

### Markers are detected but reconstruction is shifted or scaled wrong

Most likely the PDF was printed at the wrong scale. Reprint at 100% and measure
that a marker side is exactly 25 mm.

### Pose works from some views but fails from others

The camera cannot see enough markers from those views. Add more visible markers
around the table, reduce occlusion, or move the object away from the marker
border.

### Final point cloud includes table/markers

Use tighter reconstruction ROI or increase marker mask padding:

```bash
--marker-mask-padding-px 16
```

Later, add a world-space crop around the hand/object if table points remain.

### Reprojection error is high

Possible causes:

- Print scale is wrong.
- Board is not flat.
- Marker JSON does not match the printed board.
- The ZED image used for detection is distorted differently than expected.
- Markers are too blurry or viewed at a steep angle.

### Mechanical angle

Mechanical angle is not needed for this workflow. ArUco pose is the alignment
source. Mechanical angle can still be useful as a human sanity check, but the
current scripts do not require it.

## Current Limitations

- Distortion coefficients are currently treated as zeros because the ZED left
  image is used as the working image. If pose bias appears, add explicit ZED
  distortion coefficients and test whether they improve reprojection error.
- The board is one A4 sheet. For a larger scanner bed, a larger board or
  multiple measured marker sheets will be better.
- The script integrates RGB-D frames into TSDF but does not yet do final
  world-space object cropping.
- Marker detection requires visible markers in the RGB image. If the hand or
  robot blocks too many markers, that frame is skipped.

## Implementation References

- OpenCV ArUco generation and detection:
  `cv2.aruco.generateImageMarker`, `cv2.aruco.ArucoDetector`
- OpenCV pose estimation:
  `cv2.solvePnP`, `cv2.Rodrigues`, `cv2.projectPoints`
- Open3D TSDF integration:
  `o3d.pipelines.integration.ScalableTSDFVolume.integrate`

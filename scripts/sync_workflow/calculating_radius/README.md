# Orbit-Radius Calibration

This folder measures the circular path followed by the camera's optical center.
The reported radius is the camera orbit radius used by reconstruction. It is
not the radius of the hand, the profile, or the distance to the nearest visible
surface.

The method observes four ArUco markers attached to a rigid, stationary profile.
The profile does not need to lie on the mechanical rotation axis. It only
provides one fixed 3D coordinate system while the camera moves around it.

## Core idea

At every motor angle, the marker map tells the software where the detected
marker corners exist in the fixed profile coordinate system. OpenCV solves the
camera pose relative to that profile. Inverting that pose gives one 3D camera
center.

After collecting camera centers around the orbit, the software:

1. fits the best 3D plane through the camera centers;
2. projects the centers into that plane;
3. fits a circle to the projected points;
4. removes trajectory outliers and refits;
5. refines the circle with a geometric (true point-to-circle distance) step;
6. reports the fitted circle radius and a bootstrap confidence interval.

```text
fixed marker map + RGB image
             |
             v
       ArUco detection
             |
             v
   profile-to-camera pose
             |
             v
      3D camera center
             |
       repeat at angles
             v
   plane fit + circle fit
             |
             v
      camera orbit radius
```

## Files in this folder

| File | Responsibility |
|---|---|
| `generate_radius_markers.py` | Generates the printable wrap, placement guide, and fixed 3D marker map. |
| `test_radius.py` | Homes the mechanism, captures each angle, estimates poses, evaluates the trajectory, and writes the report. |
| `radius_calibration.py` | Contains marker-map geometry, ArUco detector parameters, PnP pose solving, pose acceptance, RGB/depth transform conversion, robust plane fitting, algebraic + geometric circle fitting, and the radius bootstrap. |
| `capture_orbit_pose_map.py` | Captures a complete per-angle pose map for measured-pose reconstruction; this is different from calculating only one radius. |
| `orbit_pose_map.py` | Helpers for consuming the complete per-angle pose map. |

## Marker coordinate system

`profile_marker_map.json` expresses all four marker corners in one fixed profile
coordinate system. The generated map uses:

- origin: center of the 40 x 20 mm profile cross-section on the shared marker centerline;
- +X: toward the top face;
- +Y: toward the front face;
- +Z: along the printed strip's top-edge direction;
- ID 0: front face, 18 mm marker;
- ID 1: top face, 14 mm marker;
- ID 2: back face, 18 mm marker;
- ID 3: bottom face, 14 mm marker.

All corner coordinates, marker IDs, marker rotations, marker sizes, and face
locations must match the physical wrap. A single-marker pose can appear valid
even when the map is wrong because it does not test the relative geometry
between faces. Multi-marker views reveal those map/wrap inconsistencies.

## Step 1: Capture at each motor angle

`test_radius.py` uses the shared `MotorController` and `CameraController` from
the parent `sync_workflow` folder. By default it:

1. opens the motor and Gemini camera;
2. homes the mechanism;
3. selects absolute positioning with `G90`;
4. moves X to 80 mm;
5. visits `0, 30, 60, 90, 120, 150, 180, -30, -60, -90, -120, -150, -180` degrees
   (a 30-degree-spaced positive sweep followed by the negative sweep);
6. captures ten synchronized RGB-D frames at each angle;
7. returns Y to zero when finished.

Denser angular sampling improves circle-fit conditioning; the sweep still spans
the same `0 -> 180 -> -180` range. Pass `--angles` to override.

The depth burst is median-combined and saved as a diagnostic PLY. Radius
estimation itself comes from the ArUco camera poses, not from measuring the PLY
surface.

## Step 2: Detect markers and solve a camera pose

The RGB image is used for ArUco detection because pose estimation needs the RGB
camera intrinsics and distortion coefficients. Detection runs with sub-pixel
corner refinement enabled, which lowers pose noise.

When two or more mapped markers are visible, all their 2D image corners and 3D
map corners are passed to iterative `solvePnP`. This produces the rigid
transform

```text
P_color = R * P_profile + t
```

or, in homogeneous form:

```text
P_color = T_color_from_profile * P_profile
```

When only one marker is visible, the software uses IPPE square pose estimation.
It converts the marker-local solution into the common profile coordinate system
and rejects the mirrored solution using the mapped outward face normal.

A frame pose is accepted only when:

- a mapped marker pose can be solved;
- all required geometry is in front of the camera;
- the reprojection error is finite;
- at least `--min-markers-per-pose` mapped markers are in view (default 2, so
  single-face views that cannot reveal map/wrap errors are excluded);
- mean reprojection error is at most 1.5 pixels by default.

At least 6 of the default 10 burst frames must contain accepted poses for that
motor angle. By default every accepted per-frame camera center is fed to the
circle fit, giving the fit roughly ten times more data and letting the bootstrap
estimate real scatter. Pass `--per-angle-median` to instead collapse each angle
to a single median camera center (the older behaviour).

If a stricter `--min-markers-per-pose` drops too many angles and the run fails
its angular-coverage check, lower it to `1` and re-run.

## Step 3: Calculate the RGB-camera center

PnP returns the profile-to-camera transform:

```text
T_camera_from_profile = [ R  t ]
                        [ 0  1 ]
```

This matrix says where a fixed profile point appears in the camera coordinate
system. The required trajectory sample is the opposite question: where is the
camera center in the fixed profile system?

The transform is inverted:

```text
T_profile_from_camera = inverse(T_camera_from_profile)
```

For a rigid transform, the camera center is:

```text
C_profile = -R^T * t
```

This gives one RGB-camera center `[X, Y, Z]` in metres for each accepted motor
angle.

## Step 4: Convert RGB and depth camera trajectories

The Gemini exposes a calibrated depth-to-color transform using the convention:

```text
P_color = T_color_from_depth * P_depth
```

Therefore, a profile-to-RGB pose is converted to a profile-to-depth pose using:

```text
T_depth_from_profile
    = inverse(T_color_from_depth) * T_color_from_profile
```

The software then inverts both profile-to-camera transforms to obtain separate
RGB-camera and depth-camera center trajectories.

The report always includes both fitted radii. The recommended value matches the
coordinate frame used to create the PLY files:

- aligned depth back-projected with RGB intrinsics: use the RGB-camera radius;
- point cloud in the depth-camera frame: use the depth-camera radius.

The current camera wrapper normally reports `coordinate_frame: "color"`, so
the recommended `--radius-m` normally corresponds to the RGB optical center.

## Step 5: Fit the orbit plane

Real hardware can be tilted, so the implementation does not assume that the
orbit lies in XY, XZ, or YZ. Given camera centers

```text
C0, C45, C90, C135, ...
```

it calculates their centroid and applies singular value decomposition (SVD) to
the centered points. The direction with the smallest variation is the fitted
plane normal. That normal is also the estimated mechanical orbit-axis
direction. The other two SVD directions form a 2D basis inside the orbit plane.

Every 3D camera center is projected into this fitted plane, producing 2D
coordinates `(xi, yi)`.

## Step 6: Fit a circle

The projected samples should satisfy:

```text
(xi - a)^2 + (yi - b)^2 = radius^2
```

The implementation rearranges this into a linear least-squares problem:

```text
2*a*xi + 2*b*yi + c = xi^2 + yi^2
```

After solving for `a`, `b`, and `c`:

```text
radius = sqrt(c + a^2 + b^2)
```

The 2D center `(a, b)` is converted back into the original 3D profile coordinate
system. The profile origin and fitted orbit center do not need to be the same.

This linear (Kåsa) solution is biased for short or noisy arcs. After the outlier
refit (Step 7) the center and radius are refined with `scipy.optimize.least_squares`,
minimising the true 3D point-to-circle distance (radial error plus out-of-plane
error). If SciPy is unavailable the algebraic result is used unchanged.

## Step 7: Remove trajectory outliers

The first circle fit calculates one residual for every camera center. The
residual combines:

- radial error: distance inside the plane compared with the fitted radius;
- plane error: distance above or below the fitted orbit plane.

The software calculates the residual median and median absolute deviation
(MAD):

```text
robust_sigma = 1.4826 * MAD
limit = median_residual + 3.5 * robust_sigma
```

Samples above this limit are marked as outliers. The plane and circle are then
fitted again using only the inliers.

## Step 8: Decide whether the result is usable

A radius is recommended only when the accepted depth-camera trajectory has:

- at least six unique inlier motor angles;
- no circular angular-coverage gap larger than 90 degrees.

The report also records RMSE and maximum residual. In the current implementation
these residual values are diagnostic only; they are not yet used as additional
valid/invalid thresholds. A result can therefore pass coverage while still
having a physically poor residual. Always inspect the reported residuals and
trajectory before trusting sub-millimetre accuracy.

Each fit additionally reports `radius_std_m`, `radius_ci_low_m`, and
`radius_ci_high_m` from a bootstrap over the trajectory samples. Treat the
recommended `--radius-m` as trustworthy only when this 95% interval is narrow
relative to the accuracy you need.

## Generate and print the marker wrap

From the repository root:

```bash
python scripts/sync_workflow/calculating_radius/generate_radius_markers.py
```

Print the generated PDF using **Actual size / 100%**. Do not use **Fit to page**.
Measure the printed verification ruler and coded marker squares before mounting
the wrap.

## Run calibration

Activate the project environment and run from the repository root:

```bash
conda activate hena_jet

python scripts/sync_workflow/calculating_radius/test_radius.py \
  --output-dir outputs/test_radius \
  --marker-map outputs/radius_markers/profile_marker_map.json
```

Important optional arguments include:

| Argument | Default | Meaning |
|---|---:|---|
| `--x-pos` | `80` mm | Absolute X position used for calibration. |
| `--angles` | thirteen 30-degree views | Motor angles used for trajectory samples. |
| `--frames-per-angle` | `10` | RGB-D frames captured at each motor angle. |
| `--min-markers-per-pose` | `2` | Mapped markers a frame pose must use to enter the fit. |
| `--per-angle-median` | off | Fit one median center per angle instead of every frame. |
| `--min-valid-poses-per-angle` | `6` | Accepted frame poses needed to keep one angle. |
| `--max-reprojection-error-px` | `1.5` | Maximum accepted mean corner reprojection error. |
| `--min-unique-angles` | `6` | Minimum trajectory angles after outlier rejection. |
| `--max-angle-gap-deg` | `90` | Largest allowed gap around the orbit. |

Use the same rigid camera mounting and mechanical configuration for calibration
and reconstruction. A radius measured after moving the camera mount does not
describe the earlier scan.

## Outputs

The output directory contains:

| Output | Meaning |
|---|---|
| `radius_calibration.json` | Complete result, quality decision, marker geometry, camera calibration, frame poses, RGB/depth circle fits (with `radius_std_m` / `radius_ci_low_m` / `radius_ci_high_m`), residuals, and recommended radius. |
| `radius_angle_summary.csv` | Accepted depth-camera centers, inlier flags, and circle residuals by angle. |
| `depth_camera_trajectory.ply` | Camera-center samples; green points are fit inliers and red points are rejected outliers. |
| `pose_diagnostics/` | Annotated RGB frames showing detected IDs, pose axes, acceptance, and reprojection error. |
| `frame_<angle>.ply` | Median-fused diagnostic point cloud captured at each angle. |
| `frame_<angle>_visual.png` | Marker overlay beside the depth visualization. |
| `calibration_capture.json` | Capture settings and link to the calibration report. |

A valid run prints both fitted radii and a directly usable argument:

```text
Recommended reconstruction argument: --radius-m 0.117420
```

## Understanding an invalid run

`only 0/6 valid marker poses` does not necessarily mean marker detection failed.
Check `detected_ids` and `rejection_reason` in `radius_calibration.json`.

If IDs are detected but the reason is `no usable mapped marker pose`, the usual
cause is that the physical marker wrap does not match the JSON map. Check:

- marker IDs are on their assigned profile faces;
- the strip was wrapped in the documented direction;
- no marker is rotated, mirrored, independently moved, or covered by glare;
- marker sizes and profile dimensions match the map;
- the JSON and printed wrap came from the same generator run;
- the profile remained stationary for the entire capture.

Do not reduce the reprojection threshold when `pose_ok` is already false. That
threshold is applied only after a pose has been solved.

If fewer than six angles remain, the circle is underconstrained for this
workflow. If the largest circular gap exceeds 90 degrees, the samples cover too
little of the orbit and can produce a biased radius. Correct the detection or
marker-map problem and repeat the complete calibration.

## Assumptions and limitations

- The marked profile is rigid and stationary during the complete capture.
- The marker map exactly matches the physical wrap.
- Camera intrinsics and RGB/depth extrinsics are valid.
- The camera mount does not move relative to the motor mechanism.
- The camera centers approximately follow one planar circle.
- Human/object depth points do not determine the radius; ArUco camera poses do.
- The method estimates a best-fit radius and cannot make a non-circular orbit circular.


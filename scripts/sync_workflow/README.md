# Synchronized Circular-Scan Workflow

This workflow captures RGB-D point clouds while the camera moves around a stationary object, then reconstructs the captures in one coordinate system. It assumes a circular camera path where the camera keeps looking toward the mechanical rotation center.

- `main_scan.py` controls the motor and Orbbec camera, creates one colored point cloud per motor angle, and records the scan geometry.
- `reconstruct_pipeline.py` converts motor angles into camera poses, optionally refines them with guarded ICP, and uses Open3D and Trimesh to produce a cleaned merged point cloud.

## End-to-end flow

Capture requests **1280×800 at 30 fps, disparity 128** by default. The supplied
Gemini 305g datasheet lists a 95 mm minimum for this resolution/disparity pair;
the software depth gate does not extend the hardware operating range.

The capture filter preset is explicitly set and read back at startup:

| Filter | Explicit settings |
| --- | --- |
| SpatialAdvanced | alpha 0.5, magnitude 1, disp_diff 160, radius 1 |
| Temporal | weight 0.4, diff_scale 0.1 |
| NoiseRemoval | max_size 80, min_diff 256; reference dimensions 848×480 |
| EdgeNoiseRemoval | margins 6/6, limits 70/30, vertical disabled; reference dimensions 1280×800 |

These values match the best-tested chain from the 1280×800, disparity-256
noise-tuning recording. More aggressive trials did not demonstrate an advantage.
They are pinned even where numerically equal to SDK defaults; they are not an
established optimum at disparity 128. NoiseRemoval's `min_diff=256` is a separate
filter threshold, not the camera disparity search range. Reference dimensions
are preserved from the tested filter configuration, not inferred from image size.
The device-recommended processing order and disparity conversion are retained.
Missing requested filters, rejected parameters or read-back mismatches abort
startup; verified settings are written to the capture log.

[WORKFLOW.md](WORKFLOW.md) charts the same pipeline in more detail: the calibration gates, the depth filter chain, the four reconstruction stages, and which measurements are internally consistent versus traceable.

```text
main_scan.py arguments
        |
        +-- synchronized motor movement
        +-- fresh aligned RGB-D frame bursts
        +-- valid-depth median fusion
        v
frame_<station>_<x>_<angle>.ply + intrinsics.json + scan_metadata.json
        |
        v
reconstruct_pipeline.py
        |
        +-- resolve radius, orbit axis, and separate registration/final crops
        +-- calculate a motor pose prior for every X station and angle
        +-- use priors directly or cautiously refine them with ICP
        +-- transform and crop full-resolution scans in Open3D
        +-- merge, subsample, remove noise, and estimate normals
        v
merged_cloud.ply + pose matrices + registration_diagnostics.json
```

The reconstruction is anchored by known motor positions. ICP is optional and cannot make a large, unconstrained correction to the motor geometry.


## Coordinate model and `--radius-m`

`--radius-m` is the physical distance, in metres, from the point cloud's camera origin to the mechanical center of the circular path. In this workflow, successful software depth-to-color alignment means the PLY is back-projected with RGB intrinsics and its origin is the RGB optical center. If alignment is unavailable, the wrapper falls back to the depth intrinsics and depth optical center.

It is not:

- the radius of the hand or scanned object;
- the distance to the nearest visible surface;
- a motor-axis coordinate;
- a measurement from the housing or front glass unless corrected to the optical center.

The rig calibration supplies the measured pivot, axis, and recommended radius.
A radius override changes the recorded radius; it does not replace the measured
pivot with `[0, 0, R]`. Pose priors use the measured pivot and axis.

### Motor axis versus reconstruction orbit axis

The motor moves its configured physical axis. The reconstruction axis is a
unit direction in the zero-degree point-cloud camera frame, supplied by the
calibration. A motor command named `Y` does not imply camera Y. The legacy
`--orbit-axis` option is accepted, but the measured axis takes precedence.

## Environment

Activate the project environment first:

```bash
conda activate hena_jet
```

Capture requires the Orbbec SDK, Open3D, and motor serial-port access. Reconstruction requires Open3D and Trimesh from the `hena_jet` environment; it has no CloudCompare or Flatpak dependency.

## Scan accuracy pipeline

The capture path preserves metric depth as floating-point metres from the SDK through PLY back-projection. The previous Open3D RGB-D path converted depth to unsigned 16-bit millimetres before projection, truncating every sample to the millimetre below and adding an approximately -0.5 mm systematic depth bias. `pointcloud_export.py` now back-projects the float depth directly, so that quantization is no longer part of the geometry.

Five depth filters are enabled by default and run in this order before depth-to-color alignment:

1. `SpatialAdvancedFilter`;
2. `TemporalFilter`;
3. `DisparityTransform`;
4. `NoiseRemovalFilter`;
5. `EdgeNoiseRemovalFilter`.

The spatial and temporal filters work in the disparity domain, so `DisparityTransform` converts back to depth after them, and the noise-removal filters then operate on depth. This is the Gemini 305's own recommended order.

Only the first three come from the device. `get_recommended_filters()` on this device returns ten filters and includes neither `NoiseRemovalFilter` nor `EdgeNoiseRemovalFilter`, so naming them enabled nothing and reported nothing: three of the five requested filters actually ran. Both are now constructed directly from the SDK and appended to the chain, and the startup log lists what really runs.

The remaining seven device filters stay off deliberately. `HoleFillingFilter` invents depth where the sensor measured none, `DecimationFilter` reduces resolution, and `SpatialFastFilter` and `SpatialModerateFilter` would stack redundant smoothing on top of `SpatialAdvancedFilter`. None belong in a chain feeding metric reconstruction.

Each motor angle now captures 6 fresh RGB-D frames by default. A fused pixel must have at least three valid temporal samples (`--min-valid-samples 3`). There is no per-pixel confidence gate: the Gemini 305's datasheet lists only Depth, Color, and IR output streams, and the Orbbec SDK's per-device default profiles ship no confidence stream for this device family, so a confidence-based gate is not achievable on this hardware.

Capture logs print the valid-depth fill rate for every fused angle. After reconstructing each capture, run the quality report and compare its JSON with the preceding run:

```bash
python scripts/sync_workflow/scan_quality_report.py \
  --scan-dir outputs/<scan-directory> \
  --output outputs/<scan-directory>/quality_report.json
```

Pass `--compare-merged` a second merged cloud of the same rigid object to add a repeatability section. It reports the residual twice: `as_reconstructed` compares the clouds where the pipeline put them and therefore carries any drift in the reconstructed frame, while `after_rigid_alignment` re-registers the pair first and isolates how reproducible the measured shape is. A large gap between the two means the shape repeats but the frame does not, which points at the orbit calibration rather than the sensor.

Every metric above is internal consistency: plane-RMS, cross-view residual, repeatability and ICP fitness all measure the pipeline against itself, and all stay happy around a systematically wrong orbit radius. Only a certified artifact detects that. `sphere_bar_report.py` fits both spheres of a two-sphere ball bar in a merged cloud and compares the centre-to-centre distance with the certified length:

```bash
python scripts/sync_workflow/sphere_bar_report.py \
  --merged outputs/<scan-directory>/reconstruction/merged_cloud.ply \
  --certified-distance-mm <certified> --sphere-diameter-mm <certified> \
  --output outputs/<scan-directory>/sphere_bar_report.json
```

Sphere centres are recoverable far more accurately than the point noise, because thousands of points are fitted to one known radius, so the 1.5 mm single-frame noise still resolves a sub-millimetre length error. A proportional error in the measured length is a proportional error in the calibrated radius, reported as `implied_radius_correction_ratio`.

`scan_quality_report.py` measures single-frame and merged-cloud surface plane-RMS and cross-view residual grouped by angular separation; it does not currently copy the capture-time fill rates into its JSON. Together, the logged fill rate and report measurements separate capture noise from multi-view pose error: plane-RMS describes local surface thickness, while cross-view residual shows how well different viewing angles coincide.

LingBot-Depth is deliberately excluded from the geometry path. Measurements in the 0.20-0.35 m working band showed a +74 mm bias and 14.7 mm median error; even an ideal affine alignment left 45.7 mm median error, compared with a 0.9 mm sensor floor. It is therefore not used to generate, correct, or replace the metric depth that enters reconstruction.

## Calibrate the orbit radius with the fixed profile

`calculating_radius/test_radius.py` can measure the orbit radius from four ArUco markers attached to a stationary 20 x 40 mm profile. The profile does not have to be the rotation axis. It only has to remain rigid and stationary while the camera completes the orbit.

### 1. Generate and print the marker kit

```bash
python scripts/sync_workflow/calculating_radius/generate_radius_markers.py
```

This writes the following files under `outputs/radius_markers/`:

- `profile_radius_markers_a4.pdf`: exact print-ready A4 wrap sheet;
- `profile_radius_markers_a4.png`: raster preview;
- `profile_radius_wrap.png`: the exact 130 x 40 mm unwrapped strip;
- `profile_marker_map.json`: exact 3D marker-corner coordinates;
- `profile_marker_placement.png`: fold order, face, seam, and orientation guide.

Print the PDF on ordinary matte A4 paper using **Actual size / 100%**. Do not use **Fit to page**. Before cutting, verify the printed 100 mm ruler, an 18 mm front/back marker, and a 14 mm top/bottom marker.

The generated geometry assumes:

- profile cross-section: 40 mm along map X and 20 mm along map Y;
- wrap: 130 mm around the profile and 40 mm along the bar;
- profile perimeter: 120 mm plus a 10 mm blank glue tab;
- paper/glue marker-plane offset: approximately 0.1 mm;
- ID 0 on the front broad face and ID 2 on the back broad face, both 18 mm;
- ID 1 on the top narrow face and ID 3 on the bottom narrow face, both 14 mm;
- all marker centers on the same longitudinal centerline;
- map +Z along the horizontal profile toward its chosen right end.

Cut only the solid 130 x 40 mm outline. Pre-crease the dashed lines at 10, 30, 70, and 90 mm without bending a marker. Beginning at the lower-rear seam, wrap in this order: glue tab, bottom, front, top, back. The long edge identified by the arrow must point toward the chosen right end of the profile. Apply a thin, even glue layer and keep every coded square flat, without bubbles or glossy tape. Moving, scaling, swapping, or independently rotating a marker invalidates the map.

### 2. Capture and calculate

```bash
python scripts/sync_workflow/calculating_radius/test_radius.py \
  --x-pos 400 \
  --marker-map outputs/radius_markers/profile_marker_map.json \
  --output-dir outputs/test_radius
```

At every commanded angle, the script retains the existing fused PLY and visual image, detects markers in all ten RGB frames, and estimates the fixed-profile-to-RGB-camera pose. It uses joint PnP when two or more mapped markers are visible and the marker's own 18 mm or 14 mm size when only one face is visible. A face-half-space check rejects mirrored planar poses, with an iterative fallback for exactly face-on views. Poses above 1.5 pixels mean reprojection error are rejected; an angle needs at least six accepted burst poses.

The Gemini SDK transform maps depth-camera coordinates to RGB-camera coordinates, so the report contains both trajectories. The recommended `--radius-m` is selected for the actual point-cloud coordinate frame reported by the camera wrapper: normally RGB/color for software depth-to-color alignment, or depth for the unaligned fallback.

Accepted depth-camera centers are fitted to a 3D plane and circle. A robust 3.5-MAD refit removes trajectory outliers. A radius is recommended only with at least six unique inlier angles and no circular coverage gap above 90 degrees.

Important outputs are:

- `radius_calibration.json`: complete calibration, fits, quality decision, and per-frame diagnostics;
- `radius_angle_summary.csv`: accepted angle centers and circle residuals;
- `depth_camera_trajectory.ply`: green inlier and red rejected depth-camera-center samples (the JSON also records which trajectory supplies `--radius-m`);
- `pose_diagnostics/`: annotated ArUco detection images;
- `calibration_capture.json`: capture summary and link to the calibration report.

For a valid run, the console prints a directly usable value:

```text
• Recommended reconstruction argument: --radius-m 0.117420
```

If the report says `quality_status: invalid`, do not copy a radius. Inspect the annotated images, check printed scale and placement, improve illumination, and repeat the complete orbit. The tool reports the value but deliberately does not overwrite scan metadata or reconstruction settings.

## Capture flow: `main_scan.py`

### 1. Validate and plan

The program validates angle spacing, capture dimensions, depth limits, orbit axis, and reconstruction settings. The fixed rig calibration is validated before hardware access and supplies measured geometry and the default radius.

`--dry-run` creates the selected output directory and prints the planned motor/capture sequence without connecting to hardware. It returns before starting reconstruction.

### 2. Initialize the camera

The camera controller configures the Orbbec Gemini 305 for aligned color and depth capture. It:

- maps `--disparity` 128 or 256 to the SDK mode and verifies the read-back value;
- enables software depth-to-color alignment and frame synchronization;
- warms up the stream;
- flushes stale frames before each burst so images correspond to the current motor position.

### 3. Home and move the motor

Unless `--no-home` is used, the motor performs this sequence:

1. Home Z with `G28 Z`.
2. Home A with `G28 A`.
3. Home X with `G28 X`.
4. Move X clear of the limit switches with `G1 X25 F1000`.
5. Home the main rotation axis with `G28 Y`.
6. Move Y to its staging position with `G1 Y40 F500`.
7. Move to the final scan staging position with `G1 X200 Y0 F500`.

After homing, `main_scan.py` sends `G90` to explicitly select absolute positioning. It moves to each requested X station only while Y is at zero. Every `move_x()` and `move_y()` command is followed by `M400`, and the program uses Y feedrate 800 and waits 0.1 seconds after each Y movement before capture.

For a 10-degree step, capture order is:

```text
0, +10, +20, ... +180
reset to 0 without capturing
-10, -20, ... -180
```

The origin is captured once and the reset does not produce a duplicate frame.

By default, this orbit runs once at X200. The opt-in command `--x-positions-mm 200 280` runs the complete orbit at X200, returns Y to zero, moves to X280, and repeats the complete orbit. With a 10-degree step this produces 37 captures per station, or 74 captures total.

X positions are absolute Marlin millimetres. Reconstruction treats the first station as the reference and applies subsequent station differences as negative translations along the measured orbit axis. For an axis `[1, 0, 0]`, X200 to X280 contributes `[-0.080, 0, 0]` metres to the point-cloud transform. This is the implementation's rig convention; the physical stage must remain parallel to the calibrated orbit axis.

### 4. Capture and fuse an RGB-D burst

At each angle, the camera records `--frames-per-angle` fresh aligned frames; the default is 6. For every depth frame, the code:

1. converts SDK depth values to metres;
2. rejects zero, invalid, and non-finite samples;
3. rejects depths outside `--depth-min-m` and `--depth-max-m`;
4. computes the median only from remaining valid samples;
5. keeps a fused pixel only when at least `--min-valid-samples` samples were valid, with a default of three.

Filtering before the median prevents invalid zeros from pulling fused depth toward the camera. Requiring three samples suppresses intermittent depth speckle within the six-frame burst. The latest aligned color frame supplies RGB.

The default interval is 0.02-0.25 m. Keeping the maximum close to the working distance stops much of the room and scanner structure from entering the cloud.

### 5. Save the scan dataset

Open3D projects the fused RGB-D image through the camera intrinsics and writes one colored PLY per motor angle. Console output includes point count and valid-depth coverage to expose weak captures.

The program writes:

- `frame_s<station>_x<position>_y<angle>.ply`: a unique colored cloud at one X/Y motor position;
- `intrinsics.json`: camera projection parameters;
- `scan_metadata.json`: scan geometry, capture settings, and completed angles.

Metadata is updated during scanning, so completed angles remain recorded if a later capture fails. Important fields include:

```json
{
  "schema_version": 2,
  "orbit_radius_m": 0.1175,
  "orbit_axis": [1.0, 0.0, 0.0],
  "step_deg": 10.0,
  "registration_mode": "motor",
  "capture": {
    "width": 848,
    "height": 530,
    "fps": 30,
    "disparity": 256,
    "frames_per_angle": 6,
    "min_valid_samples": 3,
    "depth_range_m": [0.02, 0.25]
  },
  "reconstruction": {
    "crop_radius_m": 0.15,
    "registration_crop_radius_m": 0.10
  },
  "x_stage": {
    "positions_mm": [200.0, 280.0],
    "reference_position_mm": 200.0,
    "positive_direction": "+orbit_axis"
  },
  "captured_angles_deg": [0.0, 10.0],
  "captures": [
    {
      "filename": "frame_s00_x200.0_y+000.0.ply",
      "station_index": 0,
      "x_position_mm": 200.0,
      "x_offset_m": 0.0,
      "angle_deg": 0.0
    }
  ]
}
```

The ordered capture manifest is authoritative for schema-version-2 scans. Older `frame_<angle>.ply` datasets without a manifest remain supported as a single X station with zero translation offset.

### 6. Optionally reconstruct

With `--reconstruct`, the program closes the hardware after capture and launches `reconstruct_pipeline.py` with the scan directory, calibration file, radius, final crop, registration crop, and registration mode.

## `main_scan.py` options

| Argument | Meaning |
|---|---|
| `--port`, `--baud` | Motor-controller serial connection. |
| `--step-deg` | Angular spacing between captures. |
| `--disparity` | Gemini 305 disparity range: 128 (default) or 256. |
| `--width`, `--height`, `--fps` | Requested RGB-D stream configuration. |
| `--orbit-geometry` | Measured calibration file; default repository `outputs/test_radius_x100/radius_calibration.json`. |
| `--radius-m` | Optical-center to orbit-center distance in metres; defaults to the calibration recommended radius. |
| `--x-positions-mm` | One or more absolute X scan stations in millimetres; default `200`. |
| `--orbit-axis X Y Z` | Legacy option; measured calibration axis takes precedence. |
| `--registration-mode` | `motor` or `guarded-icp`; default `guarded-icp`. Full-pose reconstruction is launched separately. |
| `--frames-per-angle` | Fresh frames fused per angle; default 6. |
| `--min-valid-samples` | Valid temporal depth samples required per fused pixel; default 3. |
| `--depth-min-m`, `--depth-max-m` | Accepted depth interval; defaults 0.02 and 0.25 m. |
| `--crop-radius-m` | Final output crop radial limit; default 0.085 m. |
| `--registration-crop-radius-m` | Tighter crop used only for guarded ICP; default `min(final crop, 0.10 m)`, therefore 0.085 m with the standard crop. |
| `--crop-shape` | `cylinder` (default) or `cube`; the cylinder separates radial and axial limits. |
| `--crop-axial-half-length-m` | Cylinder half-length along the orbit axis; default 0.30 m. |
| `--exclude-high-drift-frames` | When `--reconstruct` runs guarded ICP, omit sequential frames whose ICP correction exceeds the pose guards. |
| `--output-dir` | Directory for the scan PLYs, intrinsics, and metadata; default `outputs/scan`. |
| `--dry-run` | Print the plan without opening hardware. |
| `--no-home` | Skip homing; only safe when the motor origin is already valid. |
| `--reconstruct` | Launch reconstruction after capture. |

Run `python scripts/sync_workflow/main_scan.py --help` for the exact current defaults.

## Reconstruction flow: `reconstruct_pipeline.py`

For schema-version-2 datasets, the program reads station, X position, offset, angle, and filename from the capture manifest, then groups captures by station and angle. Legacy datasets still extract angles from `frame_<angle>.ply` filenames.

### Configuration precedence

Both entrypoints default to **guarded ICP** and the fixed rig calibration:
`outputs/test_radius_x100/radius_calibration.json`. The default path is resolved
relative to the repository, independently of the working directory.

Capture accepts `--orbit-geometry` to choose another file and records its
absolute path as `orbit_geometry_source` in `scan_metadata.json`. Reconstruction
selects an explicit `--orbit-geometry`, then the recorded scan path, then the
fixed default. No calibration file is copied for each scan or X station.
Replacing the shared file affects future reconstructions that reference it.

The calibration must have `quality_status: valid`, finite measured pivot and
nonzero axis, a positive recommended radius, and a recorded motor X station.
Missing or invalid calibration stops the run instead of using assumed geometry.
Known point-cloud camera frames must match the calibration camera frame.

The measured axis is authoritative. `--pivot` overrides the measured pivot and
disables the calibration-station crop shift. Radius comes from
`recommended_radius_m`, unless `--radius-m` (capture) or `--orbit-radius-m`
(reconstruction) overrides it. Changing the scalar radius does not alter the
measured pivot or axis.
`--registration-mode motor` disables ICP while retaining measured geometry.

A calibration is accepted on the uncertainty of the fitted radius, not on the scatter of the observations behind it. Single-marker ArUco poses are noisy but unbiased, so hundreds of them average to a stable radius; `--max-radius-std-mm` (0.25) is the gate on that bootstrap standard deviation, and per-angle median residuals are reported in `angle_median_residual_m` as diagnostics.

Excluding high-residual angles was tried and removed: which angles look worst differs run to run, and dropping them made independent runs agree *less*, not more. Treat a large per-angle median as a view to inspect -- a small, steeply oblique, or overexposed marker -- not as a sample to delete.

The bootstrap resamples whole angles, not individual frames, because the frames captured at one angle share that angle's pose error; resampling by frame understates the true spread of repeated runs.

A single run typically determines the radius to only about +/- 0.5 mm, which will not clear the 0.25 mm gate on its own. Average several independent runs for a tighter number -- a lone run reporting well under 0.25 mm is more likely a resampling artefact than a genuinely precise measurement.

Two limitations remain. A bootstrap standard deviation measures precision, not accuracy, so it cannot detect a biased calibration from a single run -- sparse or gappy angular coverage (few angles, large step, some contributing no poses) is a known source of that bias, and the fixed default calibration at `outputs/test_radius_x100/radius_calibration.json` has exactly that limitation. The gate also cannot distinguish a rig that needs better markers from one that just needs more runs. Comparing repeat runs is what exposes both, so treat a single calibration run as provisional until it agrees with a second one.

One `--orbit-geometry` calibration serves every X station. The radius and axis are properties of the mechanism and do not depend on X; the pivot only slides along the axis, which leaves the pose priors untouched because rotation about a line is invariant to where along it the pivot sits. Reconstruction reads the calibration's own `motor.x_position_mm` and shifts the pivot by `scan_X - calibration_X` before using it as a crop centre, so a calibration captured at X=100 reconstructs an X=50 scan correctly. A calibration must record its station; an explicit `--pivot` overrides the shift.


Explicit crop values override recorded crop metadata and defaults. Legacy metadata without `registration_crop_radius_m` automatically uses the smaller of 0.10 m and the positive final crop, so existing datasets gain the safer ICP region without changing their final output extent.


### `reconstruct_pipeline.py` options

| Argument | Meaning |
|---|---|
| `--input-dir` | Required directory containing `frame_*.ply` and optional scan metadata. |
| `--output-dir` | Reconstruction destination; default `<input-dir>/reconstruction`. |
| `--registration-mode` | `motor` or `guarded-icp`; default `guarded-icp`. |
| `--orbit-radius-m` | Explicit orbit radius; otherwise the calibration recommended radius. |
| `--orbit-geometry` | `radius_calibration.json` whose measured `orbit_geometry` supplies the orbit axis and pivot; selection is CLI, recorded scan path, then the fixed rig file. |
| `--orbit-axis X Y Z` | Legacy option; measured calibration axis takes precedence. |
| `--pivot X Y Z` | Explicit orbit center in zero-frame camera coordinates; default measured calibration pivot. |
| `--reference-angle-deg` | Motor angle treated as the reference pose; default 0 degrees. |
| `--angle-sign` | Converts the recorded motor-angle direction to the reconstruction convention; use `1` or `-1`. |
| `--crop-radius-m` | Final output crop radial limit around the pivot; metadata or 0.085 m by default, and `<= 0` disables it. |
| `--registration-crop-radius-m` | Crop limit used only to construct ICP clouds; metadata or `min(final crop, 0.10 m)` by default, and `<= 0` disables it. |
| `--crop-shape` | `cylinder` (default) or `cube`; the cylinder reads both crop radii as radial limits around the orbit axis. |
| `--crop-axial-half-length-m` | Half-length along the orbit axis when `--crop-shape cylinder`; metadata or 0.30 m by default. |
| `--exclude-high-drift-frames` | Omit a sequential frame from the final merge when its guarded-ICP correction exceeds the pose guards (guarded ICP modes only). |
| `--keep-all-components` | Keep detached point clusters in the merged cloud; by default a component smaller than 1% of the largest is discarded, since SOR cannot see a compact blob of noise that floats clear of the object. |
| `--fusion` | `both` (default) writes the cleaned point merge and TSDF artifacts; `points` or `tsdf` select one path. |
| `--tsdf-voxel-m` | TSDF voxel edge length; default 0.001 m (matches the camera's 1 mm depth quantization -- finer adds no information, coarser trades detail for smoothness). |
| `--tsdf-trunc-m` | TSDF truncation distance; default 0.003 m, kept a few voxels wide so the field can interpolate across a surface. |
| `--tsdf-depth` | `sensor` (default) or `output`; which saved depth to fuse when a scan recorded `rgbd/`. |
| `--skip-per-scan-sor` | Skip the per-frame outlier filter; final merged-cloud SOR still runs. |

### Stage 1: build pose priors

The pipeline constructs the circular motor transform and adds the station offset along the orbit axis. These matrices are saved as `*_prior.txt`.

### Stage 2: select or refine poses

| Mode | Behavior | Best use |
|---|---|---|
| `motor` | Uses each motor prior and skips ICP. | Default for calibrated hardware and noisy or symmetric subjects. |
| `guarded-icp` | Attempts a small correction around each prior and rejects unsafe or unhelpful results. | Experiments with distinctive, overlapping geometry. |

Both `main_scan.py` and `reconstruct_pipeline.py` default to guarded ICP. Guarded ICP is the capture-time default for its diagnostics rather than its corrections: motor mode writes no edges at all, so nothing records per-edge fitness, residual, or whether the orbit closes. The guards keep it safe, since any correction that fails them falls back to the motor prior.

Do not expect it to improve the surface. Its corrections are typically under 1 mm, similar to or smaller than the surface noise, so it is adjusting poses by less than the uncertainty of the points it aligns. Smooth hands, repeated geometry, background points, and partial overlap can also give ICP a plausible but physically incorrect match, which is what the guards exist to catch.

The reason to keep it on is the loop edge: matching the two ends of a complete orbit against each other is the one measurement in the pipeline that can catch a geometric error larger than the radius uncertainty or the surface noise, and motor mode never computes it. A rejected loop edge (reported in `registration_diagnostics.json`) is worth inspecting even though the fallback keeps the reconstruction safe.

Guarded ICP processes adjacent angles within each station, a loop edge for each complete orbit, and same-angle links between adjacent X stations:

1. Transform both clouds with their priors and keep only the tighter registration crop (10 cm half-extent by default), excluding most enclosure/background points.
2. Downsample registration copies to 2 mm voxels.
3. Apply statistical outlier removal with 20 neighbors and sigma 1.5.
4. Estimate normals in a 6 mm neighborhood.
5. Run coarse point-to-point ICP with a 6 mm correspondence limit.
6. Run fine point-to-plane ICP with a 2 mm limit for up to 100 iterations.
7. Accept only a result that improves the prior and passes every guard.
8. Fall back to the motor prior on any failure.

Current guards and graph settings are:

| Parameter | Value | Purpose |
|---|---:|---|
| Registration crop half-extent | 0.10 m | Limits ICP to expected object geometry without shrinking the final output. |
| Registration voxel | 0.002 m | Reduces raw depth noise and computation while retaining 2 mm-scale structure. |
| Normal radius | 0.006 m | Defines local surfaces for point-to-plane ICP. |
| Coarse/fine distance | 0.006/0.002 m | Limits correspondence search around the prior. |
| ICP iterations | 100 | Caps fine optimization work. |
| Minimum fitness | 0.35 | Requires useful overlapping inliers. |
| Maximum RMSE | 0.0015 m | Rejects dispersed inlier correspondences below the 2 mm correspondence window. |
| Maximum translation correction | 0.005 m | Keeps ICP within 5 mm of the motor prediction. |
| Maximum rotation correction | 2.5 degrees | Keeps ICP close to the motor prediction. |
| Minimum points | 100 | Avoids registration of sparse clouds. |
| Accepted prior weight | 50 | Keeps accepted ICP anchored to motor geometry. |
| Fallback prior weight | 200 | Trusts the motor much more after rejection. |
| Edge prune threshold | 0.25 | Removes low-confidence uncertain graph edges. |

The pose graph produces `optimized_poses.npy` and per-frame `*_optimized.txt` matrices. In motor mode, optimized poses equal the priors.

### Stage 3: transform full-resolution scans

Registration uses reduced copies, but Open3D applies final matrices to the original PLYs. Four bounded workers transform, crop, optionally filter, and write the scans to `01_transformed/` in stable frame order.

`--crop-shape` selects the crop geometry. With the default `cylinder`, the crop radii are radial limits around the orbit axis and `--crop-axial-half-length-m` bounds the axis separately. `cube` uses both crop arguments as axis-aligned half-extents.

Prefer the cylinder on this rig. The enclosure ring sits 90-115 mm from the orbit axis, the object stays inside 55 mm, and the object also runs +/-120 mm along the axis. Measured on `outputs/new_scan`, an 80 mm cylinder keeps 100% of the object and 0% of the ring, whereas every cube small enough to drop the ring (90 mm half-extent or less) clips 21-39% of the object, because shrinking a cube shortens it along the axis at the same time.

Despite their historical `radius` names, both crop arguments are half-extents of axis-aligned cubes, not spherical radii. Each station receives station-adjusted crop cubes around its expected pivot. `--registration-crop-radius-m` affects only the reduced copies used to estimate ICP poses. `--crop-radius-m` affects the full-resolution clouds written to `01_transformed/` and therefore the final merge. Changing the registration crop does not remove additional points from `merged_cloud.ply`. A value at or below zero disables the corresponding crop when running reconstruction directly.

Per-scan SOR uses 10 neighbors and sigma 2.0 unless `--skip-per-scan-sor` is set.

### Stage 4: merge and clean

The transformed scans are merged, then processed with:

- 1 mm spatial subsampling;
- connected-component filtering, discarding any cluster smaller than 1 percent of the largest;
- Trimesh quantized duplicate grouping at 0.1 mm;
- final SOR with 20 neighbors and sigma 1.5;
- normal estimation in a 4 mm neighborhood;
- graph/MST-based consistent normal orientation, followed by a global outward-direction check against the orbit pivot;
- final PLY validation by reopening it independently with Open3D and Trimesh.

The result is `merged_cloud.ply`. This pipeline currently produces a cleaned point cloud; it does not run Poisson mesh reconstruction.

## Reconstruction outputs

```text
reconstruction/
|-- matrices/
|   |-- frame_0.0_prior.txt
|   |-- frame_0.0_optimized.txt
|   `-- frame_0.0_optimized_matrix.txt
|-- 01_transformed/
|   `-- frame_0.0_transformed.ply
|-- optimized_poses.npy
|-- registration_diagnostics.json
|-- processing.log
|-- merged_cloud.ply
|-- tsdf_cloud.ply
|-- tsdf_mesh.ply
`-- tsdf_mesh_cleaned.ply
```

- `*_prior.txt`: pose from motor angle, measured axis/pivot, and station offset.
- `*_optimized.txt`: selected final pose; identical to the prior in motor mode.
- `*_optimized_matrix.txt`: transform copy used while processing the full-resolution scan.
- `optimized_poses.npy`: all final poses for programmatic inspection.
- `registration_diagnostics.json`: effective settings, radius source, metrics, decisions, rejection reasons, and corrections.
- `processing.log`: per-stage Python processing progress, point counts, timing, and failures.
- `merged_cloud.ply`: final cleaned, normal-estimated point cloud.
- `tsdf_cloud.ply`: point cloud extracted from the TSDF volume.
- `tsdf_mesh.ply`: raw TSDF mesh retained for inspection.
- `tsdf_mesh_cleaned.ply`: TSDF mesh with small-hole repair, fragment removal, and Taubin smoothing.

## Reading diagnostics

First confirm the effective registration mode and radius source. In motor mode, an empty ICP-edge list and zero pose corrections are expected.

In guarded-ICP mode:

- `fitness` is an overlap indicator, not proof of correct alignment;
- `rmse_m` covers accepted correspondences only, so a wrong repetitive match may still have low RMSE;
- correction translation and rotation show how far ICP tried to leave the prior;
- the rejection reason identifies the failed safety or improvement test.

Frequent fallback is not automatically an error. It means the cloud evidence was insufficient to override calibrated motor geometry.

## Common commands

Preview without hardware:

```bash
python scripts/sync_workflow/main_scan.py --dry-run \
  --step-deg 10 --radius-m 0.1175 --reconstruct
```

Capture only:

```bash
python scripts/sync_workflow/main_scan.py --port /dev/ttyUSB0 \
  --step-deg 10 --radius-m 0.1175 --output-dir scans/example
```

Capture and reconstruct from motor poses:

```bash
python scripts/sync_workflow/main_scan.py --port /dev/ttyUSB0 \
  --step-deg 10 --radius-m 0.1175 \
  --registration-mode motor --reconstruct
```

Capture full 360-degree scans at X200 and X280, then fuse all 74 captures:

```bash
python scripts/sync_workflow/main_scan.py --port /dev/ttyUSB0 \
  --x-positions-mm 200 280 --step-deg 10 --radius-m 0.1175 \
  --registration-mode motor --reconstruct
```

Try guarded ICP:

```bash
python scripts/sync_workflow/main_scan.py --port /dev/ttyUSB0 \
  --step-deg 10 --radius-m 0.1175 \
  --registration-mode guarded-icp \
  --registration-crop-radius-m 0.10 --reconstruct
```

Reconstruct a dataset containing metadata:

```bash
python scripts/sync_workflow/reconstruct_pipeline.py \
  --input-dir scans/<scan-directory>
```

Prefer the measured orbit center and axis from a valid radius calibration:

```bash
python scripts/sync_workflow/reconstruct_pipeline.py \
  --input-dir scans/<scan-directory> \
  --orbit-geometry outputs/test_radius/radius_calibration.json \
  --registration-mode motor
```

Override recorded geometry or reconstruct older data:

```bash
python scripts/sync_workflow/reconstruct_pipeline.py \
  --input-dir scans/<scan-directory> \
  --orbit-radius-m 0.1175 --orbit-axis 1 0 0 \
  --registration-mode motor --crop-radius-m 0.15
```

Use a separate output while comparing settings:

```bash
python scripts/sync_workflow/reconstruct_pipeline.py \
  --input-dir scans/<scan-directory> \
  --orbit-radius-m 0.1175 \
  --output-dir reconstructions/radius_117_5mm
```

## Radius calibration

Measure from the mechanical rotation center to the depth camera's optical center. Correct housing measurements using the manufacturer's optical-center offset.

`calculating_radius/test_radius.py` captures a multi-angle dataset using the fixed marked profile, fits the camera trajectory, and reports a radius only when its coverage and residual checks pass.

A valid report also records the measured pivot and axis in `orbit_geometry`. Both entrypoints use that complete geometry by default; `--orbit-geometry` selects a replacement calibration.

Never use the nearest hand-surface depth as orbit radius. That value changes with shape and view and describes the object, not the mechanism.

## Tuning noisy scans

Tune in this order:

1. Verify motor zero, angle sign, orbit axis, and physical radius.
2. Check each raw PLY before registration.
3. Reduce `--depth-max-m` and `--registration-crop-radius-m` to keep background out of ICP; adjust `--crop-radius-m` separately for the final output.
4. Increase `--frames-per-angle` for temporal depth speckle.
5. Keep motor registration until the deterministic reconstruction closes.
6. Try guarded ICP only when priors are close and overlap is distinctive.
7. Adjust SOR and voxel settings last; filtering cannot correct a wrong trajectory.

Voxel size suppresses detail near that scale. Correspondence distances control how far ICP searches. Limits much larger than expected motor error can match unrelated surfaces.

## Troubleshooting

### Doubled or thick surfaces

Check radius first, especially near 180 degrees. Then check orbit axis and sign, motor zero, and whether the camera consistently looks at the same mechanical center.

### Background dominates

Inspect a raw PLY. Reduce `--depth-max-m` or final `--crop-radius-m`, or improve physical isolation. If background is misleading ICP, reduce `--registration-crop-radius-m` without changing the final output crop.

### Guarded ICP keeps falling back

Read rejection reasons in `registration_diagnostics.json`. Sparse overlap, low fitness, high RMSE, excessive correction, or no improvement are intentional guards. A correct motor reconstruction makes fallback safe.

### ICP is smooth but wrong

Return to `--registration-mode motor`. Smooth, symmetric, or repetitive surfaces can satisfy nearest-neighbor metrics in the wrong pose.

### No radius is available

Restore the fixed calibration file or pass `--orbit-geometry` with a valid measured calibration. A radius override does not replace missing geometry.

### Open3D/Trimesh processing fails

Inspect `processing.log`, confirm that Open3D and Trimesh import inside `hena_jet`, and verify read/write access to input and output directories. Empty clouds, missing RGB attributes, non-finite coordinates, incomplete normals, or a failed Trimesh round trip are reported as explicit errors.

## Verification

Run tests after changing capture, pose, or reconstruction logic:

```bash
conda activate hena_jet
python -m unittest discover -s tests -p 'test_*.py'
```

For hardware changes, dry-run first, capture a short range, inspect individual PLYs, and then perform a full scan.

# Synchronized Circular-Scan Workflow

This workflow captures RGB-D point clouds while the camera moves around a stationary object, then reconstructs the captures in one coordinate system. It assumes a circular camera path where the camera keeps looking toward the mechanical rotation center.

- `main_scan.py` controls the motor and Orbbec camera, creates one colored point cloud per motor angle, and records the scan geometry.
- `reconstruct_pipeline.py` converts motor angles into camera poses, optionally refines them with guarded ICP, applies them with CloudCompare, and produces a cleaned merged point cloud.

## End-to-end flow

```text
main_scan.py arguments
        |
        +-- synchronized motor movement
        +-- fresh aligned RGB-D frame bursts
        +-- valid-depth median fusion
        v
frame_<angle>.ply + intrinsics.json + scan_metadata.json
        |
        v
reconstruct_pipeline.py
        |
        +-- resolve radius, orbit axis, and crop settings
        +-- calculate a motor pose prior for every angle
        +-- use priors directly or cautiously refine them with ICP
        +-- transform and crop full-resolution scans in CloudCompare
        +-- merge, subsample, remove noise, and estimate normals
        v
merged_cloud.ply + pose matrices + registration_diagnostics.json
```

The reconstruction is anchored by known motor positions. ICP is optional and cannot make a large, unconstrained correction to the motor geometry.

## Coordinate model and `--radius-m`

`--radius-m` is the physical distance, in metres, from the depth camera's projection/optical center to the mechanical center of the circular path.

It is not:

- the radius of the hand or scanned object;
- the distance to the nearest visible surface;
- a motor-axis coordinate;
- a measurement from the housing or front glass unless corrected to the optical center.

For `--radius-m 0.1175`, the assumed camera-to-center distance is 117.5 mm. With the default camera-coordinate orbit axis `[1, 0, 0]`, the default pivot is `[0, 0, 0.1175]` metres. Rotating about this pivot produces both rotation and translation. At 180 degrees, the implied displacement from the starting camera frame is twice the radius: 235 mm in this example.

A small radius error becomes visible across a wide scan as duplicated surfaces, thick edges, or failure to close. The value flows through the system as follows:

```text
main_scan.py --radius-m
    -> scan_metadata.json: orbit_radius_m
    -> reconstruct_pipeline.py --orbit-radius-m
    -> pivot and motor pose priors
    -> per-frame transformation matrices
    -> merged_cloud.ply
```

When `main_scan.py --reconstruct` is used, it passes the same radius to the reconstruction subprocess. The radius affects reconstruction geometry only; it does not change motor travel.

### Motor axis versus reconstruction orbit axis

The motor moves its configured physical axis. The reconstruction axis is expressed in each point cloud's camera coordinate system. The default is `[1, 0, 0]`, even if the physical motor command uses an axis named `Y`.

If the reconstruction axis or sign is wrong, clouds rotate in the wrong plane or open outward instead of overlapping. Use `--orbit-axis X Y Z` when calibration shows the default is incorrect.

## Environment

Activate the project environment first:

```bash
conda activate hena_jet
```

Capture requires the Orbbec SDK, Open3D, and motor serial-port access. Reconstruction also requires CloudCompare; the current pipeline invokes its Flatpak command-line application.

## Capture flow: `main_scan.py`

### 1. Validate and plan

The program validates angle spacing, capture dimensions, depth limits, orbit axis, and reconstruction settings. A radius is required with `--reconstruct`, because meaningful motor priors cannot be built without it.

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

For every scan-angle movement, `move_y()` sends the absolute `G1 Y...` command followed by `M400`. `main_scan.py` then waits 0.5 seconds before capture.

For a 10-degree step, capture order is:

```text
0, +10, +20, ... +180
reset to 0 without capturing
-10, -20, ... -180
```

The origin is captured once and the reset does not produce a duplicate frame.

### 4. Capture and fuse an RGB-D burst

At each angle, the camera records `--frames-per-angle` fresh aligned frames; the default is five. For every depth frame, the code:

1. converts SDK depth values to metres;
2. rejects zero, invalid, and non-finite samples;
3. rejects depths outside `--depth-min-m` and `--depth-max-m`;
4. computes the median only from remaining valid samples.

Filtering before the median prevents invalid zeros from pulling fused depth toward the camera. A pixel is retained when at least one burst frame has a valid in-range measurement. The latest aligned color frame supplies RGB.

The default interval is 0.02-0.35 m. Keeping the maximum close to the working distance stops much of the room and scanner structure from entering the cloud.

### 5. Save the scan dataset

Open3D projects the fused RGB-D image through the camera intrinsics and writes one colored PLY per motor angle. Console output includes point count and valid-depth coverage to expose weak captures.

The program writes:

- `frame_<angle>.ply`: the colored cloud at one motor position;
- `intrinsics.json`: camera projection parameters;
- `scan_metadata.json`: scan geometry, capture settings, and completed angles.

Metadata is updated during scanning, so completed angles remain recorded if a later capture fails. Important fields include:

```json
{
  "schema_version": 1,
  "orbit_radius_m": 0.1175,
  "orbit_axis": [1.0, 0.0, 0.0],
  "step_deg": 10.0,
  "registration_mode": "motor",
  "capture": {
    "width": 848,
    "height": 530,
    "fps": 30,
    "disparity": 256,
    "frames_per_angle": 5,
    "depth_range_m": [0.02, 0.35]
  },
  "reconstruction": {"crop_radius_m": 0.15},
  "captured_angles_deg": [0.0, 10.0]
}
```

Exact fields may expand; `orbit_radius_m`, `orbit_axis`, and captured angles are the key reconstruction inputs.

### 6. Optionally reconstruct

With `--reconstruct`, the program closes the hardware after capture and launches `reconstruct_pipeline.py` with the scan directory, radius, axis, crop extent, and registration mode.

## `main_scan.py` options

| Argument | Meaning |
|---|---|
| `--port`, `--baud` | Motor-controller serial connection. |
| `--step-deg` | Angular spacing between captures. |
| `--disparity` | Gemini 305 disparity range: 128 or 256. |
| `--width`, `--height`, `--fps` | Requested RGB-D stream configuration. |
| `--radius-m` | Optical-center to orbit-center distance in metres; required with `--reconstruct`. |
| `--orbit-axis X Y Z` | Orbit axis in camera coordinates; default `1 0 0`. |
| `--registration-mode` | `motor` or `guarded-icp`; default `motor`. |
| `--frames-per-angle` | Fresh frames fused per angle; default 5. |
| `--depth-min-m`, `--depth-max-m` | Accepted depth interval; defaults 0.02 and 0.35 m. |
| `--crop-radius-m` | Reconstruction crop half-extent; default 0.15 m. |
| `--output-dir` | Directory for the scan PLYs, intrinsics, and metadata; default `outputs/scan`. |
| `--dry-run` | Print the plan without opening hardware. |
| `--no-home` | Skip homing; only safe when the motor origin is already valid. |
| `--reconstruct` | Launch reconstruction after capture. |

Run `python scripts/sync_workflow/main_scan.py --help` for the exact current defaults.

## Reconstruction flow: `reconstruct_pipeline.py`

The program finds `frame_*.ply` files, extracts numeric angles, and sorts by angle rather than filename text.

### Configuration precedence

Orbit radius is resolved in this order:

1. a rough visible-surface estimate when `--auto-radius` is requested (this deliberately ignores `--orbit-radius-m`);
2. otherwise, explicit `--orbit-radius-m`;
3. otherwise, `orbit_radius_m` from `scan_metadata.json`.

`--auto-radius` is diagnostic, not physical calibration. It cannot reliably distinguish the object's visible surface from the mechanical rotation center. Explicit CLI axis and crop values similarly override recorded metadata and defaults.

### `reconstruct_pipeline.py` options

| Argument | Meaning |
|---|---|
| `--input-dir` | Required directory containing `frame_*.ply` and optional scan metadata. |
| `--output-dir` | Reconstruction destination; default `<input-dir>/reconstruction`. |
| `--registration-mode` | `motor` or `guarded-icp`; default `motor`. |
| `--orbit-radius-m` | Explicit calibrated orbit radius; otherwise read from metadata. |
| `--auto-radius` | Use a rough first-frame surface-depth estimate instead of calibrated radius. |
| `--orbit-axis X Y Z` | Axis in camera coordinates; otherwise metadata or `1 0 0`. |
| `--pivot X Y Z` | Explicit orbit center in zero-frame camera coordinates; default `[0, 0, radius]`. |
| `--reference-angle-deg` | Motor angle treated as the reference pose; default 0 degrees. |
| `--angle-sign` | Converts the recorded motor-angle direction to the reconstruction convention; use `1` or `-1`. |
| `--crop-radius-m` | Crop-cube half-extent around the pivot; metadata or 0.15 m by default, and `<= 0` disables it. |
| `--skip-per-scan-sor` | Skip the per-frame outlier filter; final merged-cloud SOR still runs. |

### Stage 1: build motor priors

For each angle, the pipeline constructs a rigid transform that rotates the scan around the calibrated axis and pivot. The zero-degree frame is the reference system. These deterministic matrices are saved as `*_prior.txt` and prevent registration drift around the orbit.

### Stage 2: select or refine poses

| Mode | Behavior | Best use |
|---|---|---|
| `motor` | Uses each motor prior and skips ICP. | Default for calibrated hardware and noisy or symmetric subjects. |
| `guarded-icp` | Attempts a small correction around each prior and rejects unsafe or unhelpful results. | Experiments with distinctive, overlapping geometry. |

Motor mode is deliberately the default. Smooth hands, repeated geometry, background points, and partial overlap can give ICP a plausible but physically incorrect match.

Guarded ICP processes adjacent pairs and a loop edge where applicable:

1. Transform and crop both clouds near their expected locations using motor priors.
2. Downsample registration copies to 3 mm voxels.
3. Apply statistical outlier removal with 20 neighbors and sigma 1.5.
4. Estimate normals in a 6 mm neighborhood.
5. Run coarse point-to-point ICP with a 4 mm correspondence limit.
6. Run fine point-to-plane ICP with a 2 mm limit for up to 100 iterations.
7. Accept only a result that improves the prior and passes every guard.
8. Fall back to the motor prior on any failure.

Current guards and graph settings are:

| Parameter | Value | Purpose |
|---|---:|---|
| Registration voxel | 0.003 m | Reduces raw depth noise and computation. |
| Normal radius | 0.006 m | Defines local surfaces for point-to-plane ICP. |
| Coarse/fine distance | 0.004/0.002 m | Limits correspondence search around the prior. |
| ICP iterations | 100 | Caps fine optimization work. |
| Minimum fitness | 0.30 | Requires useful overlapping inliers. |
| Maximum RMSE | 0.0015 m | Rejects dispersed inlier correspondences. |
| Maximum translation correction | 0.005 m | Keeps ICP within 5 mm of the motor prediction. |
| Maximum rotation correction | 2 degrees | Keeps ICP within 2 degrees of the motor prediction. |
| Minimum points | 100 | Avoids registration of sparse clouds. |
| Accepted prior weight | 50 | Keeps accepted ICP anchored to motor geometry. |
| Fallback prior weight | 200 | Trusts the motor much more after rejection. |
| Edge prune threshold | 0.25 | Removes low-confidence uncertain graph edges. |

The pose graph produces `optimized_poses.npy` and per-frame `*_optimized.txt` matrices. In motor mode, optimized poses equal the priors.

### Stage 3: transform full-resolution scans

Registration uses reduced copies, but CloudCompare applies final matrices to the original PLYs. Each scan is transformed, cropped, optionally filtered, and written to `01_transformed/`.

Despite its historical name, `--crop-radius-m` is the half-extent of an axis-aligned cube centered at the orbit pivot, not a spherical radius. A value of 0.15 keeps the interval from `pivot - 0.15` to `pivot + 0.15` m on each axis. A value at or below zero disables cropping.

Per-scan SOR uses 10 neighbors and sigma 2.0 unless `--skip-per-scan-sor` is set.

### Stage 4: merge and clean

The transformed scans are merged, then processed with:

- 1 mm spatial subsampling;
- 0.1 mm close-point/duplicate removal where supported;
- final SOR with 20 neighbors and sigma 1.5;
- normal estimation in a 4 mm neighborhood;
- graph/MST-based consistent normal orientation.

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
|-- cloudcompare.log
`-- merged_cloud.ply
```

- `*_prior.txt`: pose from motor angle, axis, pivot, and radius only.
- `*_optimized.txt`: selected final pose; identical to the prior in motor mode.
- `*_optimized_matrix.txt`: CloudCompare transform copy generated while processing the full-resolution scan.
- `optimized_poses.npy`: all final poses for programmatic inspection.
- `registration_diagnostics.json`: effective settings, radius source, metrics, decisions, rejection reasons, and corrections.
- `cloudcompare.log`: CloudCompare merge and final-cleanup log.
- `merged_cloud.ply`: final cleaned, normal-estimated point cloud.

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

Try guarded ICP:

```bash
python scripts/sync_workflow/main_scan.py --port /dev/ttyUSB0 \
  --step-deg 10 --radius-m 0.1175 \
  --registration-mode guarded-icp --reconstruct
```

Reconstruct a dataset containing metadata:

```bash
python scripts/sync_workflow/reconstruct_pipeline.py \
  --input-dir scans/<scan-directory>
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

`test_radius.py` captures a multi-angle dataset using a rigid, asymmetric target. It does not automatically calculate the radius. Use its captures to fit or validate the circular trajectory, preferably across widely separated angles where error is most observable.

Never use the nearest hand-surface depth as orbit radius. That value changes with shape and view and describes the object, not the mechanism.

## Tuning noisy scans

Tune in this order:

1. Verify motor zero, angle sign, orbit axis, and physical radius.
2. Check each raw PLY before registration.
3. Reduce `--depth-max-m` and crop extent to exclude background.
4. Increase `--frames-per-angle` for temporal depth speckle.
5. Keep motor registration until the deterministic reconstruction closes.
6. Try guarded ICP only when priors are close and overlap is distinctive.
7. Adjust SOR and voxel settings last; filtering cannot correct a wrong trajectory.

Voxel size suppresses detail near that scale. Correspondence distances control how far ICP searches. Limits much larger than expected motor error can match unrelated surfaces.

## Troubleshooting

### Doubled or thick surfaces

Check radius first, especially near 180 degrees. Then check orbit axis and sign, motor zero, and whether the camera consistently looks at the same mechanical center.

### Background dominates

Inspect a raw PLY. Reduce `--depth-max-m` or `--crop-radius-m`, or improve physical isolation. The reconstruction crop is a cube centered at the orbit pivot.

### Guarded ICP keeps falling back

Read rejection reasons in `registration_diagnostics.json`. Sparse overlap, low fitness, high RMSE, excessive correction, or no improvement are intentional guards. A correct motor reconstruction makes fallback safe.

### ICP is smooth but wrong

Return to `--registration-mode motor`. Smooth, symmetric, or repetitive surfaces can satisfy nearest-neighbor metrics in the wrong pose.

### No radius is available

Pass `--orbit-radius-m` or record calibrated `orbit_radius_m` metadata. Use `--auto-radius` only for rough investigation.

### CloudCompare fails

Inspect `cloudcompare.log`, confirm the Flatpak application is installed, and verify read/write access to input and output directories.

## Verification

Run tests after changing capture, pose, or reconstruction logic:

```bash
conda activate hena_jet
python -m unittest discover -s tests -p 'test_*.py'
```

For hardware changes, dry-run first, capture a short range, inspect individual PLYs, and then perform a full scan.

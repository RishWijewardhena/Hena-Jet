# Markerless Point-Cloud Registration

[`scripts/transform_clouds.py`](../scripts/transform_clouds.py) registers the
angle-indexed ZED-M captures into one reference frame, preserves the measured
platform, produces a cleaned merged PLY, and optionally reconstructs a Poisson
mesh.

The pipeline does not require ArUco markers or VSLAM poses. For hand scanning,
the default is deliberately simple:

1. Motor angles provide the yaw and circular-orbit prior.
2. A previously calibrated fixed axis and pivot define the camera orbit.
3. Guarded hand-only ICP provides small local refinements.

Platform correction is optional and disabled by default:

```python
USE_PLATFORM_ALIGNMENT = False
```

Set it to `True` only for a calibration object scan where the platform is
clearly visible. A hand scan does not require or attempt a platform fit.

The absolute poses are optimized together in an Open3D pose graph. This avoids
the cumulative drift caused by progressively registering every scan against an
ever-growing merged cloud.

## Requirements and Running

The project environment already contains NumPy and Open3D:

```bash
source /home/rishmika/miniconda3/etc/profile.d/conda.sh
conda activate hena_jet
python -u scripts/transform_clouds.py
```

Open3D is required for plane fitting, registration, pose-graph optimization,
and meshing. CloudCompare remains responsible for applying the optimized
matrices to the original-resolution clouds and for crop, SOR, duplicate
removal, subsampling, normals, and merge operations.

The script selects a native `CloudCompare` executable when available and
otherwise uses the installed Flatpak:

```text
/usr/bin/flatpak run org.cloudcompare.CloudCompare
```

If the Flatpak reports `Could not connect: Operation not permitted` inside a
sandboxed editor or agent, CloudCompare is already installed; the caller lacks
permission to connect to Flatpak/D-Bus. Run the command from a normal terminal
or grant that process permission. This is different from a missing
CloudCompare installation.

There are no required command-line arguments. Settings are source-level
constants near the top of the script.

## Input Contract

`INPUT_DIR` contains raw PLY files named like:

```text
angle_0005p00.ply
angle_0010p00.ply
...
angle_0360p00.ply
```

The script searches only that directory, parses the numeric angle, sorts
captures numerically, and rotates the list so `REFERENCE_ANGLE_DEG` is first.
The configured reference capture must exist. Coordinates and all distance
settings are in metres.

The default scanner model uses:

```python
REFERENCE_ANGLE_DEG = 5.0
ORBIT_RADIUS_M = 0.175
PIVOT_IN_REFERENCE = np.array([0.025, 0.0, 0.175])
ANGLE_SIGN = 1.0
```

`PIVOT_IN_REFERENCE` is a calibrated estimate in the reference camera frame,
not a value that can be inferred exactly from a single capture. Systematic
double surfaces around the full orbit usually mean this pivot, the motor-angle
scale, or the camera mount still needs physical calibration.

## Geometry

### Arbitrary-axis orbit prior

For a capture angle \(\theta_c\), the relative motor angle is:

\[
\theta = \operatorname{normalize}
\left(s(\theta_c-\theta_{reference})\right)
\]

where \(s\) is `ANGLE_SIGN`. The script uses Rodrigues' formula to rotate
around the unit orbit axis \(a\):

\[
R = I\cos\theta + (1-\cos\theta)aa^T + [a]_\times\sin\theta
\]

For pivot \(c\), translation \(t=c-Rc\) keeps every point on the orbit axis
fixed. The resulting matrix maps raw camera points to the reference frame:

\[
T =
\begin{bmatrix}
R&t\\
0&1
\end{bmatrix}
\]

In the default hand mode, the script directly uses:

```python
ORBIT_AXIS_IN_REFERENCE = np.array(
    [0.03097, 0.99918, -0.02609]
)
```

This fixed axis and `PIVOT_IN_REFERENCE` must be calibrated once while the
camera mount and motor geometry are unchanged. When
`USE_PLATFORM_ALIGNMENT=True` and `AUTO_CALIBRATE_ORBIT_AXIS=True`, visible
platform normals can refine the axis.

### Platform correction

This section applies only when `USE_PLATFORM_ALIGNMENT=True`.

Every raw scan receives an Open3D RANSAC plane fit. The fit:

- uses `PLANE_FIT_BOUNDS_M`;
- excludes the central `PLANE_EXCLUSION_RADIUS_M = 0.055` m mascot footprint;
- uses a 1.5 mm inlier distance;
- requires at least 1,000 inliers, a 30% inlier ratio, and acceptable RMSE.

After applying the motor prior, the script minimally rotates the measured plane
normal onto the reference plane and translates it along the plane normal to
match the reference offset. The rotation is performed about the point on the
reference platform beneath the orbit pivot. Rotating about the camera origin
would move the mascot laterally by several millimetres for even a small tilt.

An unreliable per-scan plane falls back to the calibrated orbit prior. After
pose-graph optimization, reliable planes are projected onto the reference
plane again so unconstrained 6-DoF ICP cannot reintroduce platform layering.

## Processing Stages

### 1. Calibrated orbit initialization

In default hand mode, the script uses the fixed calibrated axis, pivot, and
motor angle to create every pose prior. It does not search for a platform.

In optional platform mode, it additionally fits each platform plane, estimates
the orbit axis, and creates plane-corrected poses.

### 2. Object-only registration and global optimization

In hand mode, the cropped hand points are downsampled to 2 mm and assigned
normals from a 6 mm neighborhood. In platform mode, the platform is retained
for final output but points within 4 mm of its fitted plane are excluded from
registration.

Coarse and fine point-to-plane ICP run for:

- adjacent scans;
- second-neighbor scans;
- the final-to-reference loop edge.

An ICP refinement is accepted only when:

| Gate | Default |
|---|---:|
| Fitness | at least `0.30` |
| RMSE | at most `0.004` m |
| Translation correction | at most `0.010` m |
| Rotation correction | at most `4.0` degrees |

Rejected ICP never replaces the motor/plane prior. A sequential constraint is
still *usable* when that prior itself has sufficient fitness and RMSE. This is
important for a smooth or nearly symmetric mascot: unconstrained point-to-plane
ICP can lower RMSE slightly by sliding around the surface and incorrectly
undoing part of a known 5-degree motor step.

The pose graph contains:

- strong sequential motor/plane edges;
- accepted adjacent and second-neighbor ICP edges;
- an accepted final-to-reference loop edge.

The reference node is fixed, optimization distributes consistent corrections
globally, and final platform projection preserves pitch, roll, and height.

### 3. Original-resolution transformation

CloudCompare applies each optimized 4x4 matrix to the corresponding raw PLY,
then performs the configured crop and per-scan SOR. These transformed clouds
are saved in `01_transformed/`.

### 4. Merge and cleanup

CloudCompare merges the already optimized scans without additional progressive
ICP. It then performs final SOR, duplicate removal, spatial subsampling, normal
estimation, and normal orientation before saving
`mascot_merged_cleaned.ply`.

### 5. Optional Poisson mesh

When `CREATE_POISSON_MESH=True`, Open3D re-estimates consistent normals, runs
Poisson reconstruction, trims low-density vertices, crops to measured bounds,
cleans invalid geometry, and saves `mascot_mesh_poisson.ply`.

Set `CREATE_POISSON_MESH=False` to stop after the merged cloud.

## Outputs

Outputs are created below `INPUT_DIR/reconstruction/`:

```text
reconstruction/
├── 01_transformed/
│   └── angle_*_transformed.ply
├── matrices/
│   ├── angle_*_prior_matrix.txt
│   ├── angle_*_plane_corrected_matrix.txt
│   ├── angle_*_optimized_matrix.txt
│   └── angle_*_matrix.txt
├── optimized_poses.npy
├── registration_diagnostics.json
├── cloudcompare_pipeline.log
├── mascot_merged_cleaned.ply
└── mascot_mesh_poisson.ply
```

`angle_*_matrix.txt` is retained as the compatibility filename and contains the
same authoritative optimized pose as `angle_*_optimized_matrix.txt`.

`registration_diagnostics.json` records:

- platform inliers, ratios, RMSE, normals, offsets, and rejection reasons;
- estimated orbit axis and whether fallback was used;
- every edge's ICP and prior fitness/RMSE;
- accepted, rejected, fallback, and usable edge status;
- edge correction translation and rotation;
- optimized correction for every scan;
- platform residuals, loop residuals, and quality-gate results.

## Quality Gates

Matrices and diagnostics are always written first. Final full-resolution
outputs are stopped unless all of these pass:

| Metric | Requirement |
|---|---:|
| Usable sequential constraints | at least 90% |
| Optimized orbit loop translation | at most 3 mm |
| Optimized orbit loop rotation | at most 1 degree |

When `USE_PLATFORM_ALIGNMENT=True`, three additional gates apply:

| Platform metric | Requirement |
|---|---:|
| Reliable platform planes | at least 80% |
| Optimized platform-normal residual p90 | at most 0.5 degrees |
| Optimized platform height span | at most 2 mm |

Loop residual compares the optimized final-to-reference relation with the
motor/plane relation. Raw ICP correction is still recorded, but it is not a
valid closure measurement when ICP is ambiguous on a smooth surface.

`ALLOW_LOW_QUALITY_OUTPUT=False` is the safe default. Set it to `True` only to
produce an explicitly experimental output after inspecting the diagnostics.

## Important Settings

| Area | Settings |
|---|---|
| Geometry | `REFERENCE_ANGLE_DEG`, `ORBIT_RADIUS_M`, `PIVOT_IN_REFERENCE`, `ANGLE_SIGN` |
| Mode and axis | `USE_PLATFORM_ALIGNMENT`, `AUTO_CALIBRATE_ORBIT_AXIS`, `ORBIT_AXIS_IN_REFERENCE` |
| Plane ROI | `PLANE_FIT_BOUNDS_M`, `PLANE_EXCLUSION_RADIUS_M` |
| Plane RANSAC | `PLANE_RANSAC_DISTANCE_M`, `PLANE_MIN_INLIERS`, `PLANE_MIN_INLIER_RATIO` |
| Registration | `REGISTRATION_VOXEL_M`, `REGISTRATION_NORMAL_RADIUS_M`, `ICP_COARSE_DISTANCE_M`, `ICP_FINE_DISTANCE_M` |
| Guards | `ICP_MIN_FITNESS`, `ICP_MAX_RMSE_M`, `ICP_MAX_CORRECTION_M`, `ICP_MAX_CORRECTION_DEG` |
| Graph | `REGISTRATION_NEIGHBOR_SPAN`, `ORBIT_PRIOR_WEIGHT`, `POSE_GRAPH_EDGE_PRUNE_THRESHOLD` |
| Output safety | `MINIMUM_*`, `MAXIMUM_*`, `ALLOW_LOW_QUALITY_OUTPUT` |
| Cleanup | `PRE_ICP_SOR_*`, `FINAL_SOR_*`, `REMOVE_DUPLICATES_DISTANCE_M`, `SPATIAL_SUBSAMPLE_M` |
| Mesh | `CREATE_POISSON_MESH`, `POISSON_DEPTH`, `POISSON_DENSITY_TRIM_QUANTILE`, `POISSON_SCALE` |

Do not tune gates merely to force a run to pass. Correct the geometry or input
quality first.

## Troubleshooting

### Horizontal platform layers or gaps

Common causes are:

- plane correction rotating around the wrong anchor;
- unreliable plane ROI or the mascot entering the plane candidates;
- ICP reintroducing pitch/roll after plane correction;
- an incorrect pivot or orbit axis.

Inspect `plane_normal_residual_p90_deg`, `plane_height_span_m`, individual plane
fits, and the three matrix variants. A small height span with doubled mascot
surfaces points to orbit/pivot error rather than platform-plane error.

### Mascot surfaces are doubled but the platform is flat

Check:

- `ANGLE_SIGN`;
- encoder angle labels and missed motor steps;
- `PIVOT_IN_REFERENCE`;
- orbit radius and camera mounting;
- accepted ICP corrections in the diagnostics.

For a smooth mascot, a high-fitness ICP result is not automatically correct.
If it improves RMSE only slightly while changing a known 5-degree step by
several degrees, the motor/plane fallback is safer.

### Too few planes fit

Visualize `PLANE_FIT_BOUNDS_M` in raw camera coordinates. Ensure the platform
is inside the bounds, the central exclusion covers the mascot base, and at
least 1,000 platform points remain. Increase the RANSAC distance only when
measured ZED noise justifies it.

### Registration is slow

The main costs are reading 72 full-resolution PLY files twice, RANSAC on every
scan, roughly 142 pairwise coarse/fine ICP registrations, 72 CloudCompare
process launches, final merge, and Poisson meshing. Accuracy is intentionally
preferred over runtime. For diagnostics only, disable meshing; do not increase
the 2 mm registration voxel or reduce neighbors without checking accuracy.

### Quality gate stops the final reconstruction

Open `registration_diagnostics.json` and start with `quality.failure_reasons`.
Then inspect the relevant planes or edges. The script stops before overwriting
the final merged cloud precisely to prevent a plausible-looking but invalid
reconstruction.

### CloudCompare Flatpak cannot access the shared drive

Confirm the application has filesystem access to the project mount. If the
message is `Could not connect: Operation not permitted` only inside Codex or a
sandboxed IDE, run the script from a normal terminal; reinstalling CloudCompare
does not fix caller sandbox permissions.

## Tests

Run the focused unit suite in the project environment:

```bash
source /home/rishmika/miniconda3/etc/profile.d/conda.sh
conda activate hena_jet
python -m unittest scripts.test.test_transform_clouds
python -m py_compile scripts/transform_clouds.py
```

The tests cover arbitrary-axis pivot rotation, anchored plane correction,
synthetic platform projection, plane-fit reliability, orbit-axis fallback,
guard decisions, pair topology, loop and quality gates, filename ordering,
diagnostic-compatible behavior, and CloudCompare command selection.

## Known Limits

- Motor angles and filenames remain the only yaw observations.
- A fixed circular orbit and fixed pivot are assumed.
- Platform projection cannot correct an incorrect yaw angle.
- Smooth, repetitive, or symmetric geometry weakly constrains ICP yaw.
- RANSAC has stochastic sampling, though the global quality gates protect the
  output.
- Markerless registration cannot generally match the absolute observability of
  a well-calibrated marker rig.
- Poisson reconstruction can invent surfaces in unobserved regions.

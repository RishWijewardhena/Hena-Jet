# Synchronized Circular-Scan Workflow — Flow Charts

How the three programs fit together, what each stage produces, and which gates
can stop a run. Prose and parameter tables live in [README.md](README.md); this
file is the map.

```mermaid
flowchart TD
    subgraph ONCE ["ONCE per machine — ArUco profile bar in the rig"]
        A["test_radius.py"] --> AJ["radius_calibration.json<br/>radius, pivot, axis"]
    end

    subgraph EVERY ["EVERY scan — hand in the rig, no markers"]
        B["main_scan.py"] --> BJ["frame_*.ply<br/>scan_metadata.json"]
    end

    AJ -.->|numbers only| C
    BJ --> C["reconstruct_pipeline.py"]
    C --> MC["merged_cloud.ply"]
    MC --> D["scan_quality_report.py<br/>sphere_bar_report.py"]

    style ONCE fill:#e8f0fe,stroke:#4285f4
    style EVERY fill:#e6f4ea,stroke:#34a853
    style C fill:#fef7e0,stroke:#fbbc04
    style D fill:#fce8e6,stroke:#ea4335
```

**The marker bar and the hand are never in the rig at the same time.**
`test_radius.py` looks at the ArUco bar to measure where the camera travels, and
writes that geometry to a file. `main_scan.py` contains no marker code at all --
it only moves the motor and saves point clouds. Reconstruction reads the
calibration *file*, so a scan never needs a marker in view.

Re-run the calibration only when the mechanics change. **A different X station
does not need its own calibration** -- the radius and axis do not depend on X,
and reconstruction shifts the pivot by the station difference automatically.

---

## 1. Orbit calibration — `calculating_radius/test_radius.py`

Measures where the camera actually travels, by watching a stationary four-face
ArUco profile from many motor angles and fitting a circle to the recovered
camera centres.

```mermaid
flowchart TD
    START([test_radius.py]) --> HOME["home all axes<br/>move X to --x-pos"]
    HOME --> LOOP{"for each angle<br/>in --angles"}
    LOOP -->|move Y, settle| CAP[capture N frames]
    CAP --> DET[detect ArUco markers]
    DET --> SOLVE{markers<br/>in view?}

    SOLVE -->|"2 or more"| MULTI["solvePnP ITERATIVE<br/>multi-marker-iterative"]
    SOLVE -->|"exactly 1"| IPPE["solvePnPGeneric IPPE_SQUARE<br/>two mirrored solutions"]

    IPPE --> CHEIR[drop solutions failing<br/>cheirality + outward normal]
    CHEIR --> RATIO["ippe_error_ratio =<br/>runner-up error / winner error"]

    MULTI --> GATE
    RATIO --> GATE

    GATE{"classify_pose"} -->|"multi: err &gt; 3.0 px"| REJ[reject pose]
    GATE -->|"single: err &gt; 1.0 px"| REJ
    GATE -->|"ratio &lt; 2.0"| REJ2[reject: ambiguous]
    GATE -->|passes| ACC[accept pose<br/>record camera centre]

    REJ --> LOOP
    REJ2 --> LOOP
    ACC --> LOOP

    LOOP -->|all angles done| BUILD["build_fit_samples<br/>prefer multi-marker per angle<br/>drop angles under 6 poses"]
    BUILD --> FIT["fit_orbit_circle<br/>+ MAD inlier rejection"]
    FIT --> BOOT["bootstrap_radius_ci<br/>resample whole ANGLES"]
    BOOT --> QGATE{"radius std<br/>&lt;= 0.25 mm?"}
    QGATE -->|no| INVALID["quality_status: invalid<br/>no radius recommended"]
    QGATE -->|yes| VALID["quality_status: valid<br/>recommended_radius_m<br/>orbit_geometry: pivot + axis"]

    style REJ fill:#fce8e6
    style REJ2 fill:#fce8e6
    style INVALID fill:#fce8e6
    style VALID fill:#e6f4ea
```

**Why the gates are shaped this way.** Reprojection error means different things
on the two solver paths: a single marker gives four coplanar points that IPPE
solves almost exactly, so the error is near zero however wrong the pose is. On
one recorded run the 145 single-marker poses had a 0.124 px median while the 90
better-conditioned two-marker poses had 1.050 px — a shared 1.5 px limit
admitted every single-marker pose and rejected 22 poses, *all* two-marker.
Hence the split thresholds and the ambiguity ratio.

**Why the bootstrap resamples angles.** The frames captured at one angle share
that angle's pose error and are not independent draws. Per-frame resampling
claimed a 0.165 mm mean standard deviation across three runs whose radii
actually spread by 0.324 mm — optimistic by 1.9x.

---

## 2. Capture — `main_scan.py`

```mermaid
flowchart TD
    START([main_scan.py]) --> VAL[validate args]
    VAL --> CAM[start camera<br/>848x530 @ 30 fps]
    CAM --> FILT["enable 5 depth filters"]
    FILT --> HOME[home all axes]
    HOME --> PLAN["generate_angle_sequence<br/>0 to +180, back down to -180"]

    PLAN --> XLOOP{for each X station}
    XLOOP --> YLOOP{for each Y angle}
    YLOOP -->|"G1 Y.. F500 + M400"| SETTLE[blocking move]
    SETTLE --> BURST["capture_fused_rgbd<br/>15 fresh RGB-D frames"]
    BURST --> FUSE["per-pixel median<br/>needs &gt;= 3 valid samples"]
    FUSE --> PROJ["backproject_to_points<br/>float metres, RGB intrinsics"]
    PROJ --> SAVE["frame_s&lt;st&gt;_x&lt;X&gt;_y&lt;angle&gt;.ply"]
    SAVE --> YLOOP
    YLOOP -->|angles done| XLOOP
    XLOOP -->|stations done| META[scan_metadata.json]
    META --> RECON{"--reconstruct?"}
    RECON -->|yes| SUB["launch reconstruct_pipeline.py<br/>radius + axis + crop only"]
    RECON -->|no| DONE([captures on disk])

    style FILT fill:#e8f0fe
    style SUB fill:#fef7e0
```

**Depth filter chain**, applied to the raw unaligned depth frame *before* D2C
alignment, since the filters assume the native sensor neighbourhood:

```mermaid
flowchart LR
    RAW[raw depth Y16] --> S[SpatialAdvancedFilter] --> T[TemporalFilter] --> D[DisparityTransform] --> N[NoiseRemovalFilter] --> E[EdgeNoiseRemovalFilter] --> AL[AlignFilter to colour] --> OUT[aligned depth + RGB]

    style D fill:#e8f0fe
    style N fill:#fef7e0
    style E fill:#fef7e0
```

Spatial and temporal filters work in the disparity domain, so
`DisparityTransform` converts back to depth after them; the noise-removal
filters then operate on depth. The last two are **not** in the device's
`get_recommended_filters()` list and are constructed directly from the SDK.
`HoleFillingFilter` is deliberately excluded — it invents depth the sensor
never measured.

> `--reconstruct` passes only the scalar radius and axis, never
> `--orbit-geometry`, so the pivot falls back to `[0, 0, R]`. Run
> `reconstruct_pipeline.py` separately to use a measured pivot and axis.

---

## 3. Reconstruction — `reconstruct_pipeline.py`

```mermaid
flowchart TD
    START([reconstruct_pipeline.py]) --> DISC["Stage 1/4<br/>discover frame_*.ply<br/>read scan_metadata.json"]
    DISC --> GEO{"--orbit-geometry?"}
    GEO -->|yes| XCHK["read calibration's<br/>motor.x_position_mm"]
    XCHK --> MEAS["measured pivot + axis<br/>pivot shifted by scan_X - calibration_X"]
    GEO -->|no| ASSUME["assumed pivot [0,0,R]<br/>axis [1,0,0]"]

    MEAS --> PRIOR
    ASSUME --> PRIOR["build_orbit_poses<br/>rotate about axis by motor angle<br/>+ station X offset"]

    PRIOR --> MODE{"--registration-mode"}
    MODE -->|motor| USE[use priors directly<br/>no edges recorded]
    MODE -->|guarded-icp| ICP

    subgraph ICP ["Stage 2/4 — guarded ICP"]
        direction TB
        C1["crop to registration cylinder/cube<br/>then voxel 2 mm, SOR, normals"]
        C1 --> C2["coarse point-to-point, 6 mm"]
        C2 --> C3["fine point-to-plane, 2 mm, 100 iters"]
        C3 --> C4{"guards:<br/>fitness &gt;= 0.35<br/>rmse &lt;= 1.5 mm<br/>shift &lt;= 5 mm<br/>rot &lt;= 2.5 deg<br/>improves prior"}
        C4 -->|pass| C5[accept edge<br/>prior weight 50]
        C4 -->|fail| C6[fall back to prior<br/>prior weight 200]
    end

    ICP --> GRAPH["pose graph<br/>sequential + loop + cross-station<br/>optimise"]
    USE --> XFORM
    GRAPH --> XFORM["Stage 3/4<br/>transform full-resolution clouds<br/>crop, per-scan SOR"]

    XFORM --> MERGE
    subgraph MERGE ["Stage 4/4 — merge and clean"]
        direction TB
        M1[concatenate all clouds] --> M2[voxel 1 mm + min-distance sample]
        M2 --> M3[dedupe at 0.1 mm]
        M3 --> M4[statistical outlier removal]
        M4 --> M5["drop components &lt; 1% of largest"]
        M5 --> M6[estimate + orient normals]
    end

    MERGE --> OUT(["merged_cloud.ply<br/>optimized_poses.npy<br/>registration_diagnostics.json"])

    style OUT fill:#e6f4ea
```

### Crop geometry

Both crops are centred on each station's orbit pivot. `--crop-shape cylinder`
is strongly preferred on this rig.

```mermaid
flowchart LR
    subgraph CUBE ["cube — cannot separate"]
        direction TB
        CB["object inside 55 mm radial<br/>but +/-120 mm axial<br/>ring at 90-115 mm radial"]
        CB --> CB2["any cube small enough<br/>to drop the ring<br/>clips 21-39% of the object"]
    end
    subgraph CYL ["cylinder — separates cleanly"]
        direction TB
        CY["radial limit 80 mm<br/>axial limit +/-150 mm"]
        CY --> CY2["keeps 100% of the object<br/>and 0% of the ring"]
    end

    style CB2 fill:#fce8e6
    style CY2 fill:#e6f4ea
```

Two independent crops: `--registration-crop-radius-m` shapes only the reduced
clouds ICP sees, and `--crop-radius-m` shapes the full-resolution output.
Tightening the registration crop keeps rig geometry out of ICP without
shrinking the result.

---

## 4. Validation

```mermaid
flowchart TD
    MC[merged_cloud.ply] --> Q[scan_quality_report.py]
    MC --> SB[sphere_bar_report.py]

    Q --> Q1["single-frame plane RMS<br/>merged plane RMS<br/>cross-view residual by angle"]
    Q --> Q2{"--compare-merged<br/>second scan?"}
    Q2 -->|yes| Q3["as_reconstructed: shape + frame drift<br/>after_rigid_alignment: shape only"]

    SB --> S1[cluster into two spheres]
    S1 --> S2[least-squares fit each]
    S2 --> S3["centre-to-centre vs certified length<br/>scale_error_ratio<br/>implied_radius_correction_ratio"]

    Q1 --> NOTE
    Q3 --> NOTE
    S3 --> TRACE

    NOTE["internal consistency:<br/>stays happy around a wrong radius"]
    TRACE["traceable accuracy:<br/>the only check that detects scale error"]

    style NOTE fill:#fef7e0
    style TRACE fill:#e6f4ea
```

Everything except the ball bar measures the pipeline against itself. A
reconstruction can be perfectly self-consistent around a systematically wrong
orbit radius, and only a certified length detects that.

---

## Where the numbers come from

| Quantity | Produced by | Consumed by |
|---|---|---|
| `orbit_radius_m` | `test_radius.py` | pose priors, crop centres |
| `orbit_geometry.pivot_m` / `.axis` | `test_radius.py` | pose priors, crop centres, merge pivot |
| `frame_*.ply` | `main_scan.py` | reconstruction stages 1 and 3 |
| `optimized_poses.npy` | reconstruction stage 2 | stage 3 transforms |
| `registration_diagnostics.json` | reconstruction | per-edge fitness, residual, loop closure |
| `merged_cloud.ply` | reconstruction stage 4 | quality and ball-bar reports |

A calibration serves every X station. The pivot is a point in the reference
camera frame at the calibration station; reconstruction shifts it by
`scan_X - calibration_X` along the axis before using it as a crop centre. The
pose priors never needed the shift, since rotation about a line is invariant to
where along it the pivot sits.

# 360-Degree Motor-Controlled Scan Pipeline

This folder contains the automated workflow for performing 360-degree 3D scans using a Marlin-based motor controller (SKR Pro v1.2) and an Orbbec Gemini 305 RGB-D camera.

## Architecture

The pipeline is split into **two independent phases**: Capture and Reconstruction.

### Phase 1: Capture (`main_scan.py`)

Drives the motor and camera to collect per-angle point clouds:

1. **`motor_controller.py`** — Serial G-Code communication with M400 synchronization.
2. **`camera_controller.py`** — Orbbec SDK pipeline (SW D2C alignment, configurable disparity).
3. **`main_scan.py`** — Orchestrates homing → motor stepping → frame capture → PLY export.

At every angle the workflow median-combines several fresh depth frames, rejects
depth outside the configured close-range window, and reports valid coverage.

**Outputs:** `frame_<angle>.ply`, `intrinsics.json`, and `scan_metadata.json`.

### Phase 2: Reconstruction (`reconstruct_pipeline.py`)

Operates offline on the saved PLY files. Adapted from the proven [`transform_clouds_pipeline.py`](../transform_clouds_pipeline.py):

| Stage | Tool | What it does |
|-------|------|-------------|
| 1 | Python | Build orbit pose priors from motor angles using Rodrigues rotation |
| 2 | Python / Open3D | Use deterministic motor poses (default), or tightly guarded residual ICP with motor fallback |
| 3 | CloudCompare | Apply optimized 4×4 transforms, crop (optional), SOR per scan |
| 4 | CloudCompare | Merge all scans, dedup, spatial subsample, compute + orient normals |

**Outputs:** `merged_cloud.ply`, pose matrices, `cloudcompare.log`, and
`registration_diagnostics.json`. Diagnostics record the effective radius,
pivot, axis, mode, thresholds, prior scores, and final pose corrections.

---

## Usage

### Capture only (recommended first run)

```bash
conda run -n hena_jet python scripts/sync_workflow/main_scan.py \
  --step-deg 10.0 \
  --disparity 256 \
  --frames-per-angle 5 \
  --depth-min-m 0.02 --depth-max-m 0.35 \
  --radius-m 0.1175 \
  --output-dir outputs/scan_01
```

### Capture + auto-reconstruct

```bash
conda run -n hena_jet python scripts/sync_workflow/main_scan.py \
  --step-deg 10.0 \
  --disparity 256 \
  --frames-per-angle 5 \
  --depth-min-m 0.02 --depth-max-m 0.35 \
  --radius-m 0.1175 \
  --output-dir outputs/scan_01 \
  --registration-mode motor \
  --reconstruct
```

### Reconstruct from existing captures

```bash
conda run -n hena_jet python scripts/sync_workflow/reconstruct_pipeline.py \
  --input-dir outputs/scan_01 \
  --registration-mode motor
```

The radius and orbit axis are read from `scan_metadata.json`. An explicit
`--orbit-radius-m` overrides the recorded radius.

Use `--registration-mode guarded-icp` only when the motor-only baseline is
correct. ICP is accepted only when it stays within 5 mm / 2 degrees and
improves the motor prior; otherwise that edge falls back to the motor pose.

### Dry run (no hardware)

```bash
conda run -n hena_jet python scripts/sync_workflow/main_scan.py --dry-run
```

---

## Motor Homing Sequence

To prevent limit switch conflicts, the homing follows this specific order:

1. `G28 Z`, `G28 A`, `G28 X` — Home individual axes
2. `G1 X25 F1000` — Move X clear of limit switches
3. `G28 Y` — Home Y (main rotation axis)
4. `G1 Y40 F500` → `G1 X200 Y0 F500` — Final staging position

## Scan Sequence

1. **Positive sweep**: `Y0` → `Y+180` in steps, capturing at each
2. **Reset**: Return to `Y0` (no capture)
3. **Negative sweep**: `Y0` → `Y-180` in steps, capturing at each

Every `G1` move is followed by `M400` to guarantee the motor has physically stopped before the camera captures.

## Key Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--step-deg` | 10.0 | Degrees per capture step |
| `--radius-m` | None | Optical-center to rotation-center distance; required for auto-reconstruction |
| `--disparity` | 256 | Depth disparity search range (128 or 256) |
| `--width` / `--height` | 848×530 | Camera resolution |
| `--frames-per-angle` | 5 | Fresh frames median-combined at each angle |
| `--depth-min-m` / `--depth-max-m` | 0.02 / 0.35 | Capture-time valid depth window |
| `--orbit-axis` | 1 0 0 | Rotation axis in camera coordinates (the motor axis is still named Y) |
| `--crop-radius-m` | 0.15 | Half-extent of the reconstruction crop cube |
| `--registration-mode` | motor | `motor` or `guarded-icp` |

## Orbit-radius calibration capture

`test_radius.py` now captures a rigid asymmetric target from multiple angles:

```bash
conda run -n hena_jet python scripts/sync_workflow/test_radius.py \
  --output-dir outputs/radius_calibration \
  --frames-per-angle 5
```

Keep the target fixed at the mechanical orbit center. The script writes a
`calibration_capture.json` manifest and multi-angle PLY/image pairs suitable
for fitting camera poses and a circle. Do not use the nearest visible hand
surface depth as the orbit radius: it measures the surface, not the rotation
center. Until a pose-fitting target is integrated, use the mechanically
measured optical-center radius and verify it with the motor-only reconstruction.

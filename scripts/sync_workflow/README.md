# 360-Degree Motor-Controlled Scan Pipeline

This folder contains the automated workflow for performing 360-degree 3D scans using a Marlin-based motor controller (SKR Pro v1.2) and an Orbbec Gemini 305 RGB-D camera.

## Architecture

The pipeline is split into **two independent phases**: Capture and Reconstruction.

### Phase 1: Capture (`main_scan.py`)

Drives the motor and camera to collect per-angle point clouds:

1. **`motor_controller.py`** — Serial G-Code communication with M400 synchronization.
2. **`camera_controller.py`** — Orbbec SDK pipeline (SW D2C alignment, configurable disparity).
3. **`main_scan.py`** — Orchestrates homing → motor stepping → frame capture → PLY export.

**Outputs:** A folder of `frame_<angle>.ply` files + `intrinsics.json`.

### Phase 2: Reconstruction (`reconstruct_pipeline.py`)

Operates offline on the saved PLY files. Adapted from the proven [`transform_clouds_pipeline.py`](../transform_clouds_pipeline.py):

| Stage | Tool | What it does |
|-------|------|-------------|
| 1 | Python | Build orbit pose priors from motor angles using Rodrigues rotation |
| 2 | Open3D | Guarded coarse-to-fine ICP (8mm → 3mm) + pose-graph optimization with orbit prior edges (weight=50) |
| 3 | CloudCompare | Apply optimized 4×4 transforms, crop (optional), SOR per scan |
| 4 | CloudCompare | Merge all scans, dedup, spatial subsample, compute + orient normals |
| 5 | Open3D | Poisson surface reconstruction with density trimming |

**Outputs:** `merged_cloud.ply`, `poisson_mesh.ply`, `registration_diagnostics.json`.

---

## Usage

### Capture only (recommended first run)

```bash
python scripts/sync_workflow/main_scan.py \
  --step-deg 10.0 \
  --disparity 128 \
  --width 1280 --height 800 \
  --radius-m 0.12 \
  --output-dir outputs/scan_01
```

### Capture + auto-reconstruct

```bash
python scripts/sync_workflow/main_scan.py \
  --step-deg 10.0 \
  --disparity 128 \
  --width 1280 --height 800 \
  --radius-m 0.12 \
  --output-dir outputs/scan_01 \
  --reconstruct
```

### Reconstruct from existing captures

```bash
python scripts/sync_workflow/reconstruct_pipeline.py \
  --input-dir outputs/scan_01 \
  --orbit-radius-m 0.12
```

### Dry run (no hardware)

```bash
python scripts/sync_workflow/main_scan.py --dry-run
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
| `--radius-m` | 0.12 | Distance from camera lens to rotation center |
| `--disparity` | 256 | Depth disparity search range (128 or 256) |
| `--width` / `--height` | 848×530 | Camera resolution |
| `--orbit-axis` | 0 1 0 | Rotation axis (Y-axis default) |
| `--crop-bounds` | None | Optional 6-value crop box for reconstruction |

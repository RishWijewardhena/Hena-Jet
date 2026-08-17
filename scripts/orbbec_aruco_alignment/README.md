# Orbbec ArUco Table-Board Alignment & TSDF Scanning

This directory provides an ArUco-marker based alignment and 3D reconstruction workflow for **Orbbec RGB-D cameras** (e.g. Gemini 305 / Gemini 335 / Femto Bolt series) using `pyorbbecsdk` and Open3D.

Instead of estimating camera motion from the object surface or relying on mechanical motor angles, this method uses a printed tabletop ArUco board. The table board is fixed, flat, and known in metric coordinates. Every time the camera captures an RGB-D frame, the visible marker corners are detected to calculate the 6-DoF camera pose in the table coordinate frame. Aligned depth frames are then fused into a global Open3D TSDF volume.

---

## Environment Setup

Run all scripts inside the `hena_jet` conda environment:

```bash
source /home/rishmika/miniconda3/etc/profile.d/conda.sh
conda activate hena_jet
```

Verify camera and library dependencies:

```bash
python -c "import pyorbbecsdk, open3d, cv2; print('All dependencies ready!')"
```

---

## Files

| File | Purpose |
|---|---|
| `generate_aruco_table_board.py` | Generates printable ArUco board PDF/PNG, individual marker images, and the board JSON geometry file. |
| `orbbec_aruco_pose_debug.py` | Streams Orbbec RGB, detects markers, solves 6-DoF pose with factory intrinsics/distortion, and outputs debug overlays without TSDF fusion. |
| `orbbec_aruco_tsdf_scan.py` | Streams synchronized Orbbec Color + Depth, aligns depth to color (SW AlignFilter or HW D2C), solves marker pose, and integrates frames into an Open3D TSDF mesh/cloud. |
| `aruco_common.py` | Shared marker detection, pose solving, Orbbec camera parameter parsing, depth cleanup, and Open3D helpers. |

---

## Board Geometry (`aruco_table_board.json`)

The board JSON file defines the metric ground truth for all marker corners on the table plane:

```json
{
  "dictionary": "DICT_5X5_100",
  "marker_size_m": 0.025,
  "marker_count": 12,
  "paper": "a4",
  "dpi": 300
}
```

The board defines the world coordinate frame:
- **Origin**: Center of the printed scan area on the table.
- **+X**: Right along the printed sheet.
- **+Y**: Up along the printed sheet.
- **+Z**: Out of the table plane towards the camera.
- **Units**: Meters.

---

## Workflow Guide

### Step 1: Generate & Print the ArUco Board

```bash
python scripts/orbbec_aruco_alignment/generate_aruco_table_board.py \
  --out-dir outputs/aruco_table_board
```

Print `outputs/aruco_table_board/aruco_table_board.pdf` at **100% scale (no page scaling / fit to page off)**. Measure a marker side with a ruler to confirm it is exactly **25 mm**.

### Step 2: Debug ArUco Pose Estimation

Check that the Orbbec color camera detects markers and computes a stable pose:

```bash
python scripts/orbbec_aruco_alignment/orbbec_aruco_pose_debug.py \
  --board-json outputs/aruco_table_board/aruco_table_board.json \
  --out-dir outputs/orbbec_aruco_debug \
  --width 1280 --height 800 --fps 30 \
  --frames 5
```

Review the outputs in `outputs/orbbec_aruco_debug/pose_debug_0000.png` and `pose_debug.json`. Ensure 3D coordinate axes are firmly anchored to the board origin and reprojection error is low (< 1.5 px).

### Step 3: Run Interactive or Automated TSDF Scan

#### Interactive Capture (Press Enter per view, `q` when done):

```bash
python scripts/orbbec_aruco_alignment/orbbec_aruco_tsdf_scan.py \
  --board-json outputs/aruco_table_board/aruco_table_board.json \
  --width 1280 --height 800 --fps 30 \
  --min-depth-m 0.10 \
  --max-depth-m 0.50 \
  --voxel-length-m 0.0015 \
  --sdf-trunc-m 0.008
```

#### Automated Capture:

```bash
python scripts/orbbec_aruco_alignment/orbbec_aruco_tsdf_scan.py \
  --auto-capture \
  --auto-capture-interval-s 0.5 \
  --max-frames 60 \
  --min-markers 3 \
  --min-valid-depth-px 10000 \
  --voxel-length-m 0.0015 \
  --sdf-trunc-m 0.008 \
  --mesh-out outputs/aruco_scans/orbbec_aruco_tsdf_mesh.ply \
  --cloud-out outputs/aruco_scans/orbbec_aruco_tsdf_cloud.ply \
  --poses-out outputs/aruco_scans/orbbec_aruco_poses.npy \
  --scan-json outputs/aruco_scans/orbbec_aruco_scan.json
```

---

## Key Tuning Parameters

| Parameter | Default | Description |
|---|---:|---|
| `--width` / `--height` | `1280` / `800` | Stream resolution (16:10 full sensor FOV for Gemini series). Can also use `1280x720`, `848x530`, or `1920x1080`. |
| `--hw-d2c` | `False` | Enables hardware Depth-to-Color alignment (uses software `AlignFilter` by default). |
| `--min-depth-m` / `--max-depth-m` | `0.10` / `1.00` | Valid depth range in meters for TSDF integration. |
| `--roi X0 Y0 X1 Y1` | `0 0 1 1` | Image-space crop for depth integration. Marker detection still uses full image. |
| `--voxel-length-m` | `0.002` (2 mm) | TSDF voxel size in meters. |
| `--sdf-trunc-m` | `0.012` (12 mm) | TSDF truncation distance. |
| `--hole-fill-size-m` | `0.003` (3 mm) | Maximum mesh hole size to fill automatically. |
| `--cleanup-outlier-neighbors` | `20` | Outlier neighbor count for statistical point cloud filtering. |
| `--cleanup-min-cluster-fraction` | `0.02` | Removes disconnected mesh fragments smaller than 2% of the main cluster. |
| `--marker-mask-padding-px` | `8` | Expands detected marker depth masks to keep markers out of the final mesh. |

---

## Coordinate Systems & Fusion

- **Camera Convention**: Open3D / OpenCV standard (`+X` right, `+Y` down, `+Z` optical forward).
- **Depth Scale**: Automatically extracted via `depth_frame.get_depth_scale()` from the Orbbec SDK.
- **Pose Integration**: `ScalableTSDFVolume.integrate(rgbd, open3d_intrinsic, world_to_camera)` fuses each view directly into table world space.

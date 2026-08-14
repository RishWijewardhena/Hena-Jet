# Gemini 305 TUM capture

This tool records SDK-filtered, D2C-aligned Gemini 305 RGB-D frames and their
KISS-ICP poses for offline SurfelMeshing experiments. It does not run
SurfelMeshing and does not change the existing live map exporter.

## Laptop capture

Run from the repository root in the `hena_jet` environment:

```bash
python scripts/gemini305_tum_capture/capture_tum_rgbd.py \
  --out outputs/hand_capture_001 \
  --n-scans 300
```

Use `--n-scans -1` to capture until `Ctrl+C`. The output path must be new or
empty so an earlier scan cannot be overwritten.

The capture contains:

```text
hand_capture_001/
├── rgb/*.png
├── depth/*.png
├── rgb.txt
├── depth.txt
├── associated.txt
├── trajectory.txt
├── calibration.txt
└── capture_info.txt
```

`capture_info.txt` records the aligned depth scale required for
SurfelMeshing's `--depth_scaling` option. The depth PNGs retain the Gemini
integer depth values without rescaling.

## Tests

These tests do not require a connected camera:

```bash
python -m unittest scripts/gemini305_tum_capture/test_capture_tum_rgbd.py
```

## Jetson Orin Nano boundary

The capture format and NumPy/OpenCV serialization are architecture-neutral.
The Jetson deployment will still need compatible ARM64 builds of the Orbbec
Python SDK, KISS-ICP dependencies, OpenCV, and NumPy. Validate laptop captures
before building that deployment environment.

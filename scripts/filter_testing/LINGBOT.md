# Gemini 305 LingBot test

This separate test follows the `MDMModel.from_pretrained` / `infer` approach in
`lingbot-depth/example.py`. It saves one capture, then runs inference offline.
It does not integrate learned depth into the moving scan or reconstruction.

Activate `hena_jet` and run from the repository root. Hold the camera and target
still, with the target in the requested depth range. Capture defaults match the
current scanner: 1280×800, disparity 128, 30 fps, explicitly configured filters.

```bash
python scripts/filter_testing/lingbot_gemini.py capture \
  --output outputs/lingbot_gemini/capture

python scripts/filter_testing/lingbot_gemini.py refine \
  --input outputs/lingbot_gemini/capture \
  --output outputs/lingbot_gemini/refined --device cuda
```

Capture saves `rgb.png`, float32 `depth_input.npy` in metres, `intrinsics.txt`,
and `capture.json` with timestamps, filter settings and active disparity. It
uses a single filtered frame, without burst median fusion. RGB is decoded by
its actual format and rectified to the aligned depth's pinhole grid. Nonzero
depth distortion or unsupported colour distortion models fail explicitly.

Refinement saves `comparison.png`, `comparison.npz` and `report.json`.
The figure shows RGB, sensor depth, refined depth, signed depth changes,
model-only depth and removed sensor pixels. Depth previews share one colour
scale. NPZ preserves unmodified model output as well as range-gated arrays and
separate validity masks. Coverage and sensor-to-model changes are not accuracy
measurements; inspect real edge preservation and compare known dimensions.

The default evaluation range is 0.095–0.25 m, reflecting the supplied datasheet's
95 mm minimum at 1280×800/disparity 128. Override with `--min-depth` and
`--max-depth` for a different supported capture setup. For disparity 256, use
`capture --disparity 256` and `refine --min-depth 0.05` after verifying Close-Range
mode. A software range change does not change hardware capability.

`--resolution-level 0` uses the model's lowest token budget to start on the 4 GB
GPU; output retains the input dimensions. `--resolution-level 9` matches the
example's default and requires more memory. Neither setting guarantees a fit.
`--device cpu` is available, but can be slow. Model ID defaults to the same v0.5
checkpoint as the example; `--model /path/to/model.pt` uses an existing local
checkpoint without needing a model download. First use of a remote ID may download
weights. The repository's `lingbot-depth` directory is preferred over editable
packages installed from another checkout.

The refine command also accepts existing example folders with `rgb.png`,
`raw_depth.png` (uint16 millimetres), and `intrinsics.txt`. Those PNG inputs retain
their original whole-millimetre precision; new captures avoid this rounding.
Use a new output directory for each experiment; existing results are not overwritten.

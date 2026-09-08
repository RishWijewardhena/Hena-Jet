# Software filter test bench

Standalone experiment; does not change `main_scan.py`, move motors, or change
production filtering defaults. Uses the SDK recorder/playback APIs and the
post-processing pattern in `examples/orbbec_python_sdk/advanced/09_post_processing.py`.

Hold the camera and a rigid test object still. Include a flat surface, an edge,
and a background behind that edge. Keep exposure and illumination consistent.
This records **native depth only**, without colour or alignment; it tests depth
post-processing, not colour registration or reconstruction accuracy.

From the repository root, with the camera available:

```bash
conda run --no-capture-output -n hena_jet python scripts/filter_testing/filter_bench.py capture \
  --bag outputs/filter_test/raw.bag --width 848 --height 530 --disparity 256 --seconds 3

conda run --no-capture-output -n hena_jet python scripts/filter_testing/filter_bench.py compare \
  --bag outputs/filter_test/raw.bag --output outputs/filter_test/comparison \
  --seconds 5 --min-depth 0.04 --max-depth 0.25
```

Capture sets and verifies the disparity property and exports device settings.
The settings sidecar is capture metadata only: hardware properties are not
restored onto the read-only playback device. Playback uses bag calibration and
profiles, with host filter settings explicitly selected for each trial.
It does not select a Close-Range preset: select/verify the appropriate preset
beforehand. A 40 mm software gate does not guarantee 40 mm hardware operation.
Existing recordings/results are not overwritten. Close other camera applications.

The default suite tries all 13 single-depth post-processing candidates enumerated
in the script, plus a baseline and the production filter selection. A class in
the SDK does not imply Gemini 305 compatibility. Failed trials are recorded and
the suite continues, returning a nonzero exit code if any trial fails. Native
SDK crashes are isolated to worker processes; see each `worker.log`.

Recommended order is retained; requested filters outside that list are appended
in JSON order. The recommended disparity transform is retained even in the
baseline. Non-Y16 output is rejected rather than treated as metric depth. Device
and firmware processing remain active, so baseline is not raw stereo imagery.
Each trial starts with fresh filter state. Temporal state persists within its
recording, as intended. Do separate recordings for different camera positions.

Output per trial:

- `inventory.json`: SDK-reported ranges, steps, defaults and active settings.
- `depths.npz`: per-frame depth in metres, source frame IDs/timestamps and output
  intrinsics (`fx, fy, cx, cy`). No distortion correction is claimed.
- `depth_preview.png`: last frame using the same depth colour scale for all trials.
- `metrics.json`: valid coverage, frame-to-frame depth change and filter runtime.
- `summary.json` in the comparison folder: status and input-frame equality checks.

Playback is collected before filtering, with callbacks and copied frames. Use a
collection window longer than the recording. A comparison with differing frame
IDs is marked invalid. Short recordings limit memory use; SDK buffering or slow
storage can still cause losses. The equality check verifies trials against the
baseline, not completeness against every frame originally recorded.

## Parameter experiments

Read a successful trial's `inventory.json`, then create a JSON list such as:

```json
[
  {"name": "edge_default", "filters": {"EdgeNoiseRemovalFilter": {}}},
  {"name": "noise_then_edge", "filters": {
    "NoiseRemovalFilter": {}, "EdgeNoiseRemovalFilter": {}
  }}
]
```

Replace an empty object with `{"EXACT_PARAMETER_NAME_FROM_INVENTORY": VALUE}`
to tune a parameter. Names, bounds and steps are validated before processing;
the SDK may impose additional constraints. Run with `--trials path/to/trials.json`
and a new output directory. Start with individual settings before combinations.
Do not assume stronger filtering means better accuracy: inspect lost edges and
coverage alongside stability, and validate dimensions against a known target.

## Coverage limits

This bench covers SDK single-depth filtering. HDRMerge and SequenceId need
purpose-built interleaved/HDR recordings; alignment, format conversion and point
cloud generation are geometric/data operations, not denoisers. Confidence gating
needs a recorded confidence stream. These are not silently treated as tested.
Multi-frame median fusion, Open3D outlier/component cleanup and LingBot are not
implemented in this bench yet. The results do not establish an optimal preset,
absolute dimensional accuracy, edge error, or a fix for X-station seams.

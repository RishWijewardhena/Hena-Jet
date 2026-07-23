# Continuous Multilevel ZED Scan

Use `scripts/capture_zed_multilevel.py` when one horizontal orbit does not see
the full object height. The command captures several continuous revolutions
while keeping one ZED GEN_3 positional-tracking session alive. The motor stops
only between revolutions so the camera mount can be raised manually.

## Physical requirements

- Keep the object, pivot, table, and background stationary for the entire scan.
- Measure orbit radius from the pivot to the ZED left optical center.
- Begin with the camera at the lowest intended height and level orientation.
- Raise the complete camera mount vertically; do not rotate or move it sideways.
- The default height offsets are 0, 20, and 40 mm.
- ZED IMU fusion is enabled by default. Use `--no-vslam-use-imu` only for a
  deliberate camera-only experiment.

Direction alternation requires the current motor-controller firmware. It
extends the continuous command with an optional direction field:
`start_continuous,<step>,<ppr>,<rpm>,<forward|reverse>`. Upload that firmware
before running a multilevel scan.

## Capture

Activate the `hena_jet` environment, disconnect the Tkinter GUI from the serial
port, and run:

```bash
python3 scripts/capture_zed_multilevel.py \
  --radius-m 0.192 \
  --serial-port /dev/ttyACM0 \
  --pulses-per-revolution 10000 \
  --step-deg 5 \
  --motor-rpm 0.25 \
  --first-pass-direction forward \
  --height-offsets-m 0 0.02 0.04 \
  --between-pass-wait-s 30 \
  --out-dir captures/zed_m_vslam_multilevel
```

Replace the serial port and pulses per revolution with the controller's actual
values. A 5-degree step produces 72 captures per pass and 216 captures across
the default three heights.

Pass directions automatically alternate `forward`, `reverse`, `forward` for
the default three heights. `forward` preserves the firmware's original DIR-pin
polarity. If the first orbit must use the opposite physical direction, start
with `--first-pass-direction reverse`; the following passes still alternate.
The second pass therefore unwinds the first pass instead of adding another
cable turn.

After a revolution completes:

1. Raise the camera to the absolute height printed by the prompt.
2. Keep the camera facing and orbit radius unchanged.
3. Press Enter after positioning it. Pressing early is safe: the next pass
   cannot start until at least 30 seconds have elapsed.
4. The script checks the VSLAM displacement. Vertical and lateral errors must
   each be at most 2 mm and orientation change must be at most 1 degree.
5. If validation fails, adjust the mount and press Enter again. ZED tracking
   remains active during every retry.

If tracking becomes invalid during a lift, the command aborts and marks the
session failed. Preserve that directory for diagnostics, return the camera to
its initial position, and restart all passes with a new `--out-dir`.

## Dataset layout

```text
captures/zed_m_vslam_multilevel/
├── scan_session.json
├── vslam_trajectory.jsonl
├── pass_00_height_000mm/
│   ├── angle_0005p00.json
│   ├── angle_0005p00.npz
│   └── ...
├── pass_01_height_020mm/
└── pass_02_height_040mm/
```

Each capture stores its pass index, absolute height offset, global capture
index, depth confidence map, and camera-to-VSLAM-world pose. The top-level
manifest records pass-start, saved 360-degree, and stopped/runout poses so loop
closure and manual-lift validation use the correct measurements.

## Fusion

First choose explicit object-relative crop limits for the hand. The center and
up vector are loaded automatically from `scan_session.json`; crop dimensions
are never guessed from the point-cloud centroid.

```bash
python3 scripts/fuse_orbit_posegraph.py \
  --capture-dir captures/zed_m_vslam_multilevel \
  --pose-source vslam \
  --max-depth-confidence 50 \
  --max-object-radius-m 0.10 \
  --min-object-height-m -0.08 \
  --max-object-height-m 0.16 \
  --voxel-length-m 0.0008 \
  --icp-voxel-m 0.0008 \
  --mesh-out outputs/hand_multilevel/mesh.ply \
  --cloud-out outputs/hand_multilevel/cloud.ply \
  --diagnostics-out outputs/hand_multilevel/diagnostics.json \
  --feasibility-out outputs/hand_multilevel/feasibility.json
```

The crop numbers above are starting examples, not calibration values. Inspect
an uncropped result or representative captures and adjust them to the actual
hand position and height. Fusion orders frames by pass metadata, creates
sequential and loop edges inside each revolution, and pairs adjacent heights by
nearest VSLAM camera geometry rather than matching repeated angle filenames.

The feasibility report passes only when every height contains the expected 72
valid captures and each pass independently satisfies the configured 2 mm orbit
radius/closure and 1-degree rotation gates.

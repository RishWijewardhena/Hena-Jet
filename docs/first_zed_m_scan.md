# First ZED-M Object Scan

This is the simplest way to test the scanner concept with the ZED-M before moving to Orbbec.

## Important Mechanical Rule

Keep the object fixed.

Only move the camera around the object. For this first test, use known angle marks such as 0, 30, 60, 90 degrees, etc. Later, the rotary encoder will provide this angle automatically.

## Setup

Place a rigid object at the center of your scanner area. A small box, bottle, or 3D printed object is better than a hand for the first test because it will not move.

Measure:

- `radius_m`: distance from the object center to the ZED camera center, in meters.
- `height_m`: camera height relative to the object center. Use `0` for the first test if the camera is level with the object center.

Example:

```bash
python3 scripts/capture_zed_angle.py --angle-deg 0 --radius-m 0.35 --height-m 0
```

Rotate the camera to the next marked angle and run:

```bash
python3 scripts/capture_zed_angle.py --angle-deg 30 --radius-m 0.35 --height-m 0
python3 scripts/capture_zed_angle.py --angle-deg 60 --radius-m 0.35 --height-m 0
python3 scripts/capture_zed_angle.py --angle-deg 90 --radius-m 0.35 --height-m 0
```

For a full circular scan, capture:

```text
0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330
```

## Merge The Scan

After capturing all angles:

```bash
python3 scripts/merge_circular_scan.py
```

The merged point cloud is saved to:

```text
outputs/zed_m_merged_scan.ply
```

Open that `.ply` file in CloudCompare or MeshLab.

## What To Check

The merged cloud should look like the object from all captured sides.

If the object appears duplicated or spread around a ring, check:

- The radius is wrong.
- The angle direction is reversed.
- The camera was not pointing at the object center.
- The object moved during capture.

If the cloud is too noisy:

- Increase lighting.
- Use a more textured object.
- Increase the camera distance.
- Lower `--max-depth-m`.
- Try `--resolution VGA` first for faster tests.

## Next Improvement

Once the manual angle test works, replace manual angle entry with rotary encoder readings and save the real encoder angle with each frame.

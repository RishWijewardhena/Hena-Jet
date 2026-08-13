# Point-Cloud Transformation Guide

This guide explains how to use the simplified point-cloud transformation
entry point. It aligns angle-indexed ZED point clouds, merges them, and creates
an optional Poisson mesh.

## Files

- `scripts/transform_clouds.py` is the small public command-line entry point.
- `scripts/transform_clouds_pipeline.py` contains the geometry, guarded ICP,
  pose-graph optimization, CloudCompare cleanup, diagnostics, and meshing
  implementation.
- `docs/transform_clouds.md` contains detailed information about the
  registration mathematics and quality checks.

The implementation was separated so normal usage does not require navigating
the approximately 2,000 lines of numerical pipeline code.

## Requirements

Activate the project Conda environment:

```bash
conda activate hena_jet
```

The pipeline requires:

- Python and NumPy
- Open3D
- CloudCompare, installed natively or through Flatpak

## Input Layout

The input directory must contain point clouds directly at its root:

```text
captures/Plastic_Hand_full_lingbot/
├── angle_0010p00.ply
├── angle_0020p00.ply
├── angle_0030p00.ply
└── ...
```

Only files matching `angle_*p*.ply` are processed. The angle encoded in each
filename is used as the camera-orbit angle.

Do not place the input PLY files inside the `metadata/` or `qc/` directories.

## Run the Plastic-Hand Reconstruction

From the project root, run:

```bash
python scripts/transform_clouds.py \
  --input-dir captures/Plastic_Hand_full_lingbot \
  --reference-angle-deg 10 \
  --orbit-radius-m 0.18
```

Use the first available capture as `--reference-angle-deg`. For example, use
`10` when the first file is `angle_0010p00.ply`.

The orbit radius must be the physical distance in metres from the rotation
centre to the camera optical centre.

To choose a different output location:

```bash
python scripts/transform_clouds.py \
  --input-dir captures/Plastic_Hand_full_lingbot \
  --output-dir outputs/plastic_hand_reconstruction \
  --reference-angle-deg 10 \
  --orbit-radius-m 0.18
```

View all available options with:

```bash
python scripts/transform_clouds.py --help
```

## Pipeline Stages

The command performs five stages:

1. Load and numerically sort the `angle_*.ply` captures.
2. Place each cloud using its motor angle and the calibrated camera orbit.
3. Refine the alignment using guarded ICP and a global pose graph.
4. Crop, clean, merge, and calculate normals with CloudCompare.
5. Create a Poisson mesh with Open3D when meshing is enabled.

## Outputs

When `--output-dir` is omitted, results are written to:

```text
<input-dir>/reconstruction_new_without_platform_alignment/
```

Important outputs are:

```text
reconstruction_new_without_platform_alignment/
├── 01_transformed/                 # Individual aligned clouds
├── matrices/                       # Prior and optimized transformations
├── mascot_merged_cleaned.ply       # Final merged point cloud
├── mascot_mesh_poisson.ply         # Reconstructed mesh
├── optimized_poses.npy             # Optimized camera poses
├── registration_diagnostics.json   # Registration quality evidence
└── cloudcompare_pipeline.log       # CloudCompare processing log
```

Inspect `registration_diagnostics.json` before accepting the final mesh. A
pipeline that finishes without an exception can still contain low-overlap
capture pairs.

## Important LingBot Warning

A 30 cm depth limit alone does not guarantee that every retained point belongs
to the hand. LingBot can predict false near-depth surfaces in large unsupported
background holes. Those surfaces can mislead ICP and create doubled geometry.

Before transformation, accept LingBot-filled pixels only near valid ZED hand
measurements. Keep the original ZED measurement wherever it exists, and use
LingBot only for supported missing pixels.

## Troubleshooting

### Reference scan not found

If the first capture is `angle_0010p00.ply`, pass:

```text
--reference-angle-deg 10
```

The configured reference angle must have a matching PLY file.

### CloudCompare cannot be opened

When CloudCompare is installed as Flatpak, run the transformation from a
normal terminal. Sandboxed applications may not have permission to access
Flatpak or D-Bus.

### Doubled or thick surfaces

Check, in this order:

1. Remove unsupported LingBot background fills.
2. Confirm the physical orbit radius.
3. Confirm the reference capture angle.
4. Inspect rejected ICP edges in `registration_diagnostics.json`.
5. Verify that the crop contains the hand without nearby background objects.

Do not reverse the angle direction without measuring overlap. For the existing
`Plastic_Hand_full_lingbot` capture, the configured direction produced much
better adjacent-cloud overlap than the reversed direction.

## Verification

Run the transformation unit tests with:

```bash
python -m unittest scripts/test/test_transform_clouds.py
```

Compile both modules without running the reconstruction:

```bash
python -m py_compile \
  scripts/transform_clouds.py \
  scripts/transform_clouds_pipeline.py
```

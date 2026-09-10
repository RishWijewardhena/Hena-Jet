#!/usr/bin/env python3
"""Batch LingBot depth completion for angle-indexed ZED captures.

The transform-compatible ``angle_*p*.ply`` files are written directly below
the output directory. Larger diagnostics are kept in ``metadata/`` and
``qc/`` so ``scripts/transform_clouds.py`` ignores them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np


ANGLE_STEM_PATTERN = re.compile(r"angle_(\d+)p_?(\d+)$")
DEFAULT_INPUT_DIR = Path("captures/Plastic_Hand_full")
DEFAULT_OUTPUT_DIR = Path("captures/Plastic_Hand_full_lingbot")
DEFAULT_MODEL = "robbyant/lingbot-depth-postrain-dc-vitl14"


@dataclass(frozen=True)
class FramePaths:
    stem: str
    angle_deg: float
    npz_path: Path
    json_path: Path


@dataclass(frozen=True)
class OutputPaths:
    ply: Path
    npz: Path
    json: Path
    comparison: Path
    input_depth: Path
    lingbot_depth: Path
    completed_depth: Path
    raw_valid_mask: Path
    filled_mask: Path

    def required_files(self) -> tuple[Path, ...]:
        return (
            self.ply,
            self.npz,
            self.json,
            self.comparison,
            self.input_depth,
            self.lingbot_depth,
            self.completed_depth,
            self.raw_valid_mask,
            self.filled_mask,
        )


def angle_from_stem(stem: str) -> float:
    match = ANGLE_STEM_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(f"Cannot extract an angle from capture stem: {stem}")
    return float(f"{match.group(1)}.{match.group(2)}")


def discover_frames(input_dir: Path) -> list[FramePaths]:
    input_dir = Path(input_dir)
    frames = []
    for npz_path in input_dir.glob("angle_*p*.npz"):
        json_path = npz_path.with_suffix(".json")
        if not json_path.is_file():
            raise FileNotFoundError(
                f"Missing JSON metadata for {npz_path.name}: {json_path}"
            )
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        filename_angle = angle_from_stem(npz_path.stem)
        metadata_angle = float(metadata["angle_deg"])
        if not np.isclose(filename_angle, metadata_angle, atol=1.0e-6):
            raise ValueError(
                f"Angle mismatch for {npz_path.name}: filename says "
                f"{filename_angle}, metadata says {metadata_angle}"
            )
        frames.append(
            FramePaths(
                stem=npz_path.stem,
                angle_deg=metadata_angle,
                npz_path=npz_path,
                json_path=json_path,
            )
        )
    frames.sort(key=lambda frame: frame.angle_deg)
    if not frames:
        raise FileNotFoundError(
            f"No angle-indexed NPZ captures found in: {input_dir}"
        )
    return frames


def output_paths_for_frame(output_dir: Path, frame: FramePaths) -> OutputPaths:
    output_dir = Path(output_dir)
    metadata_dir = output_dir / "metadata"
    qc_dir = output_dir / "qc"
    return OutputPaths(
        ply=output_dir / f"{frame.stem}.ply",
        npz=metadata_dir / f"{frame.stem}.npz",
        json=metadata_dir / f"{frame.stem}.json",
        comparison=qc_dir / f"{frame.stem}_comparison.png",
        input_depth=qc_dir / f"{frame.stem}_depth_input.png",
        lingbot_depth=qc_dir / f"{frame.stem}_depth_lingbot.png",
        completed_depth=qc_dir / f"{frame.stem}_depth_completed.png",
        raw_valid_mask=qc_dir / f"{frame.stem}_raw_valid_mask.png",
        filled_mask=qc_dir / f"{frame.stem}_lingbot_filled_mask.png",
    )


def frame_is_complete(paths: OutputPaths) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in paths.required_files())


def capture_arrays(
    frame: FramePaths,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    metadata = json.loads(frame.json_path.read_text(encoding="utf-8"))
    with np.load(frame.npz_path) as capture:
        required_keys = {"depth_image_m", "color_image", "confidence_image"}
        missing = required_keys.difference(capture.files)
        if missing:
            raise ValueError(
                f"{frame.npz_path} is missing arrays: {sorted(missing)}"
            )
        depth = np.asarray(capture["depth_image_m"], dtype=np.float32)
        color = np.asarray(capture["color_image"], dtype=np.uint8)
        confidence = np.asarray(capture["confidence_image"], dtype=np.uint8)

    if depth.ndim != 2:
        raise ValueError(f"Depth must have shape (H, W); got {depth.shape}")
    expected_color_shape = (*depth.shape, 3)
    if color.shape != expected_color_shape:
        raise ValueError(
            f"RGB must have shape {expected_color_shape}; got {color.shape}"
        )
    if confidence.shape != depth.shape:
        raise ValueError(
            f"Confidence must have shape {depth.shape}; got {confidence.shape}"
        )
    intrinsics = metadata.get("camera_intrinsics")
    if not isinstance(intrinsics, dict):
        raise ValueError(f"Missing camera_intrinsics in {frame.json_path}")
    expected_dimensions = (int(intrinsics["height"]), int(intrinsics["width"]))
    if depth.shape != expected_dimensions:
        raise ValueError(
            f"Capture shape {depth.shape} does not match intrinsics "
            f"{expected_dimensions} in {frame.json_path}"
        )
    return depth, color, confidence, metadata


def normalized_intrinsics_matrix(intrinsics: dict) -> np.ndarray:
    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    if width <= 0 or height <= 0:
        raise ValueError("Camera width and height must be positive")
    return np.array(
        [
            [float(intrinsics["fx"]) / width, 0.0, float(intrinsics["cx"]) / width],
            [0.0, float(intrinsics["fy"]) / height, float(intrinsics["cy"]) / height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def combine_depth(
    raw_depth: np.ndarray,
    predicted_depth: np.ndarray,
    max_output_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(raw_depth, dtype=np.float32)
    predicted = np.asarray(predicted_depth, dtype=np.float32)
    if raw.shape != predicted.shape:
        raise ValueError(
            "raw_depth and predicted_depth must have the same shape; "
            f"got {raw.shape} and {predicted.shape}"
        )
    if max_output_depth_m <= 0:
        raise ValueError("max_output_depth_m must be positive")

    raw_valid = np.isfinite(raw) & (raw > 0)
    completed = np.where(raw_valid, raw, predicted)
    output_valid = (
        np.isfinite(completed)
        & (completed > 0)
        & (completed <= max_output_depth_m)
    )
    completed = np.where(output_valid, completed, 0.0).astype(np.float32)
    filled = (~raw_valid) & output_valid
    return completed, raw_valid, filled


def backproject_depth(
    depth_m: np.ndarray,
    intrinsics: dict,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Depth must have shape (H, W); got {depth.shape}")
    height, width = depth.shape
    if (int(intrinsics["height"]), int(intrinsics["width"])) != (height, width):
        raise ValueError(
            "Depth dimensions do not match camera intrinsics: "
            f"depth={(height, width)}, intrinsics="
            f"{(intrinsics['height'], intrinsics['width'])}"
        )

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    rows, columns = np.indices((height, width), dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    points = np.stack(
        (
            (columns - cx) * depth / fx,
            (rows - cy) * depth / fy,
            depth,
        ),
        axis=-1,
    )
    return points[valid].astype(np.float32), valid


def write_point_cloud(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3); got {points.shape}")
    if colors.shape != points.shape:
        raise ValueError(
            f"colors must have shape {points.shape}; got {colors.shape}"
        )
    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains non-finite coordinates")

    import trimesh

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.stem}.tmp.ply")
    trimesh.PointCloud(points, colors=colors).export(temporary_path)
    temporary_path.replace(path)


def depth_visualization(
    depth_m: np.ndarray,
    *,
    maximum_m: float | None = None,
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    colored = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not valid.any():
        return colored
    minimum = float(depth[valid].min())
    maximum = float(maximum_m) if maximum_m is not None else float(depth[valid].max())
    scaled = np.clip(
        (depth - minimum) / max(maximum - minimum, 1.0e-8) * 255.0,
        0.0,
        255.0,
    ).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if not cv2.imwrite(str(temporary_path), image):
        raise RuntimeError(f"Could not write image: {temporary_path}")
    temporary_path.replace(path)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.stem}.tmp.json")
    temporary_path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(temporary_path, **arrays)
    temporary_path.replace(path)


def save_frame_outputs(
    paths: OutputPaths,
    *,
    depth_input: np.ndarray,
    depth_predicted: np.ndarray,
    depth_completed: np.ndarray,
    raw_valid: np.ndarray,
    filled: np.ndarray,
    color_image: np.ndarray,
    confidence_image: np.ndarray,
    points: np.ndarray,
    point_colors: np.ndarray,
    metadata: dict,
    maximum_depth_m: float,
) -> None:
    write_point_cloud(paths.ply, points, point_colors)
    write_npz(
        paths.npz,
        depth_input_m=depth_input.astype(np.float32),
        depth_lingbot_m=depth_predicted.astype(np.float32),
        depth_completed_m=depth_completed.astype(np.float32),
        raw_valid_mask=raw_valid.astype(bool),
        lingbot_filled_mask=filled.astype(bool),
        color_image=color_image.astype(np.uint8),
        confidence_image=confidence_image.astype(np.uint8),
        points=points.astype(np.float32),
        colors=point_colors.astype(np.uint8),
    )
    write_json(paths.json, metadata)

    input_color = depth_visualization(depth_input)
    predicted_color = depth_visualization(depth_predicted)
    completed_color = depth_visualization(
        depth_completed,
        maximum_m=maximum_depth_m,
    )
    write_image(paths.input_depth, input_color)
    write_image(paths.lingbot_depth, predicted_color)
    write_image(paths.completed_depth, completed_color)
    write_image(paths.raw_valid_mask, raw_valid.astype(np.uint8) * 255)
    write_image(paths.filled_mask, filled.astype(np.uint8) * 255)
    write_image(
        paths.comparison,
        np.concatenate((input_color, predicted_color, completed_color), axis=1),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply LingBot-Depth to angle-indexed ZED captures while preserving "
            "valid sensor depth and emitting transform_clouds.py-compatible PLYs."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--max-output-depth-m",
        type=float,
        default=0.30,
        help="Keep final geometry up to this distance (default: 0.30 m).",
    )
    parser.add_argument(
        "--resolution-level",
        type=int,
        choices=range(10),
        default=0,
        metavar="0-9",
        help="LingBot token resolution; 0 is safest for a 4 GB GPU.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N frames for a preflight run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess frames whose complete output set already exists.",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Disable LingBot's learned output-validity mask.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all capture files without loading LingBot.",
    )
    return parser.parse_args(argv)


def cuda_synchronize(device) -> None:
    if device.type == "cuda":
        import torch

        torch.cuda.synchronize(device)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_output_depth_m <= 0:
        raise ValueError("--max-output-depth-m must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    frames = discover_frames(args.input_dir)
    if args.limit is not None:
        frames = frames[: args.limit]
    print(f"Found {len(frames)} capture(s) in {args.input_dir}")

    # Validate every selected capture before allocating model/GPU memory.
    for frame in frames:
        capture_arrays(frame)
    print("Capture validation passed.")
    if args.validate_only:
        return 0

    import torch
    from mdm.model.v2 import MDMModel

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    print(f"Device: {device}")
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(
            "GPU memory before model load: "
            f"{free_bytes / 2**30:.2f}/{total_bytes / 2**30:.2f} GiB free"
        )

    print(f"Loading model once: {args.model}")
    cuda_synchronize(device)
    started = time.perf_counter()
    model = MDMModel.from_pretrained(args.model).to(device).eval()
    cuda_synchronize(device)
    load_seconds = time.perf_counter() - started
    print(f"Model loaded in {load_seconds:.3f} s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "qc").mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "max_output_depth_m": args.max_output_depth_m,
        "resolution_level": args.resolution_level,
        "model_load_seconds": load_seconds,
        "frames": {},
    }

    processed = 0
    skipped = 0
    for index, frame in enumerate(frames, start=1):
        paths = output_paths_for_frame(args.output_dir, frame)
        if not args.overwrite and frame_is_complete(paths):
            skipped += 1
            manifest["frames"][frame.stem] = {"status": "skipped_existing"}
            print(f"[{index:02d}/{len(frames):02d}] {frame.stem}: already complete")
            continue

        print(f"[{index:02d}/{len(frames):02d}] {frame.stem}: processing")
        depth, color, confidence, source_metadata = capture_arrays(frame)
        intrinsics = source_metadata["camera_intrinsics"]
        normalized_intrinsics = normalized_intrinsics_matrix(intrinsics)

        image_tensor = torch.tensor(
            color / 255.0,
            dtype=torch.float32,
            device=device,
        ).permute(2, 0, 1).unsqueeze(0)
        depth_tensor = torch.tensor(depth, dtype=torch.float32, device=device)
        intrinsics_tensor = torch.tensor(
            normalized_intrinsics,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        cuda_synchronize(device)
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                output = model.infer(
                    image_tensor,
                    depth_in=depth_tensor,
                    apply_mask=not args.no_mask,
                    intrinsics=intrinsics_tensor,
                    resolution_level=args.resolution_level,
                    use_fp16=True,
                )
            cuda_synchronize(device)
        except Exception as error:
            manifest["frames"][frame.stem] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            write_json(manifest_path, manifest)
            raise
        inference_seconds = time.perf_counter() - started

        depth_predicted = output["depth"].squeeze().float().cpu().numpy()
        depth_completed, raw_valid, filled = combine_depth(
            depth,
            depth_predicted,
            max_output_depth_m=args.max_output_depth_m,
        )
        points, completed_valid = backproject_depth(depth_completed, intrinsics)
        point_colors = color[completed_valid]

        frame_metadata = dict(source_metadata)
        frame_metadata["point_count"] = int(len(points))
        frame_metadata["lingbot_depth"] = {
            "model": args.model,
            "resolution_level": args.resolution_level,
            "max_output_depth_m": args.max_output_depth_m,
            "preserved_raw_valid_depth": True,
            "inference_seconds": inference_seconds,
            "raw_valid_pixels": int(raw_valid.sum()),
            "lingbot_filled_pixels": int(filled.sum()),
            "final_valid_pixels": int(completed_valid.sum()),
        }
        save_frame_outputs(
            paths,
            depth_input=depth,
            depth_predicted=depth_predicted,
            depth_completed=depth_completed,
            raw_valid=raw_valid,
            filled=filled,
            color_image=color,
            confidence_image=confidence,
            points=points,
            point_colors=point_colors,
            metadata=frame_metadata,
            maximum_depth_m=args.max_output_depth_m,
        )

        processed += 1
        manifest["frames"][frame.stem] = {
            "status": "complete",
            "angle_deg": frame.angle_deg,
            "inference_seconds": inference_seconds,
            "raw_valid_pixels": int(raw_valid.sum()),
            "lingbot_filled_pixels": int(filled.sum()),
            "final_valid_pixels": int(completed_valid.sum()),
            "ply": paths.ply.name,
        }
        write_json(manifest_path, manifest)
        print(
            f"  {inference_seconds:.3f} s, {filled.sum():,} filled, "
            f"{len(points):,} points"
        )

        del image_tensor, depth_tensor, intrinsics_tensor, output
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest["summary"] = {
        "selected_frames": len(frames),
        "processed_frames": processed,
        "skipped_frames": skipped,
    }
    write_json(manifest_path, manifest)
    print(
        f"Done: {processed} processed, {skipped} skipped. "
        f"Transform-compatible PLYs: {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

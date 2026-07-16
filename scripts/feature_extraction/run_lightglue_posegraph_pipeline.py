#!/usr/bin/env python3
"""Run the ZED -> LightGlue -> pose graph -> TSDF pipeline."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASET = Path("datasets/zed_m_open3d")
DEFAULT_OUT_PREFIX = Path("outputs/feature_tracking/lightglue_pipeline_10_20cm")
DEFAULT_MODEL_CACHE = Path("outputs/feature_tracking/model_cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture a ZED RGB-D dataset, track it with SuperPoint/LightGlue, "
            "optimize loop closures with a pose graph, and fuse a final TSDF "
            "cloud/mesh."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-capture", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--interval-s", type=float, default=0.15)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--resolution",
        choices=["HD2K", "HD1080", "HD720", "VGA"],
        default="HD720",
    )
    parser.add_argument(
        "--depth-mode",
        choices=["PERFORMANCE", "QUALITY", "ULTRA", "NEURAL", "NEURAL_LIGHT"],
        default="NEURAL",
    )
    parser.add_argument("--depth-min-m", type=float, default=0.10)
    parser.add_argument("--depth-max-m", type=float, default=0.20)

    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--max-keypoints",
        type=int,
        default=4096,
        help="Maximum SuperPoint keypoints per frame. Higher helps small/weak details.",
    )
    parser.add_argument(
        "--filter-threshold",
        type=float,
        default=0.05,
        help="LightGlue match confidence threshold. Lower keeps more tentative matches.",
    )
    parser.add_argument("--ransac-threshold-m", type=float, default=0.012)
    parser.add_argument("--min-depth-matches", type=int, default=20)
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.30)
    parser.add_argument("--max-step-rotation-deg", type=float, default=30.0)
    parser.add_argument("--debug-pairs", type=int, default=80)
    parser.add_argument("--model-cache-dir", type=Path, default=DEFAULT_MODEL_CACHE)

    parser.add_argument("--voxel-length-m", type=float, default=0.0015)
    parser.add_argument("--sdf-trunc-m", type=float, default=0.006)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if args.interval_s < 0:
        raise ValueError("--interval-s must be zero or positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be zero or positive")
    if args.depth_min_m <= 0 or args.depth_max_m <= args.depth_min_m:
        raise ValueError("--depth-max-m must be greater than --depth-min-m")
    if args.max_keypoints <= 0:
        raise ValueError("--max-keypoints must be positive")
    if args.filter_threshold < 0:
        raise ValueError("--filter-threshold must be zero or positive")
    if args.ransac_threshold_m <= 0:
        raise ValueError("--ransac-threshold-m must be positive")
    if args.min_depth_matches < 3:
        raise ValueError("--min-depth-matches must be at least 3")
    if args.min_inliers < 3:
        raise ValueError("--min-inliers must be at least 3")
    if not 0.0 < args.min_inlier_ratio <= 1.0:
        raise ValueError("--min-inlier-ratio must be between 0 and 1")
    if args.max_step_rotation_deg <= 0:
        raise ValueError("--max-step-rotation-deg must be positive")
    if args.debug_pairs < 0:
        raise ValueError("--debug-pairs must be zero or positive")
    if args.voxel_length_m <= 0:
        raise ValueError("--voxel-length-m must be positive")
    if args.sdf_trunc_m <= args.voxel_length_m:
        raise ValueError("--sdf-trunc-m must be greater than --voxel-length-m")


def output_path(prefix: Path, suffix: str) -> Path:
    return prefix.with_name(f"{prefix.name}{suffix}")


def dataset_has_frames(dataset: Path) -> bool:
    image_dir = dataset / "image"
    depth_dir = dataset / "depth"
    return any(image_dir.glob("*.png")) or any(depth_dir.glob("*.png"))


def ensure_dataset_state(args: argparse.Namespace) -> None:
    if args.skip_capture:
        if not dataset_has_frames(args.dataset):
            raise FileNotFoundError(
                f"No image/depth PNG frames found in existing dataset: {args.dataset}"
            )
        return
    if dataset_has_frames(args.dataset) and not args.overwrite:
        raise SystemExit(
            f"Dataset already contains frames: {args.dataset}\n"
            "Pass --overwrite to replace it, or --skip-capture to process it."
        )


def check_cuda_available() -> None:
    script = (
        "import torch; "
        "raise SystemExit(0 if torch.cuda.is_available() else 1)"
    )
    result = subprocess.run([sys.executable, "-c", script], check=False)
    if result.returncode != 0:
        raise SystemExit(
            "--device cuda was requested, but PyTorch cannot see a CUDA device."
        )


def run_command(command: list[str], dry_run: bool) -> None:
    print(flush=True)
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    root = Path(__file__).resolve().parent
    capture_script = root / "capture_zed_open3d_dataset.py"
    tracker_script = root / "superpoint_lightglue_tracker.py"
    posegraph_script = root / "optimize_feature_pose_graph.py"
    fusion_script = root / "fuse_feature_tracked_rgbd.py"

    tracking_poses = output_path(args.out_prefix, "_tracking_poses.npy")
    tracking_report = output_path(args.out_prefix, "_tracking_report.json")
    tracking_debug_dir = output_path(args.out_prefix, "_tracking_debug_matches")
    posegraph_poses = output_path(args.out_prefix, "_posegraph_poses.npy")
    posegraph_report = output_path(args.out_prefix, "_posegraph_report.json")
    posegraph_json = output_path(args.out_prefix, "_posegraph.json")
    cloud_out = output_path(args.out_prefix, "_cloud.ply")
    mesh_out = output_path(args.out_prefix, "_mesh.ply")

    commands: list[list[str]] = []
    if not args.skip_capture:
        capture_command = [
            sys.executable,
            str(capture_script),
            "--out-dir",
            str(args.dataset),
            "--frames",
            str(args.frames),
            "--interval-s",
            str(args.interval_s),
            "--warmup",
            str(args.warmup),
            "--min-depth-m",
            str(args.depth_min_m),
            "--max-depth-m",
            str(args.depth_max_m),
            "--resolution",
            args.resolution,
            "--depth-mode",
            args.depth_mode,
        ]
        if args.overwrite:
            capture_command.append("--overwrite")
        commands.append(capture_command)

    commands.extend(
        [
            [
                sys.executable,
                str(tracker_script),
                "--dataset",
                str(args.dataset),
                "--max-frames",
                str(args.frames),
                "--device",
                args.device,
                "--debug-pairs",
                str(args.debug_pairs),
                "--max-keypoints",
                str(args.max_keypoints),
                "--filter-threshold",
                str(args.filter_threshold),
                "--ransac-threshold-m",
                str(args.ransac_threshold_m),
                "--min-depth-matches",
                str(args.min_depth_matches),
                "--min-inliers",
                str(args.min_inliers),
                "--min-inlier-ratio",
                str(args.min_inlier_ratio),
                "--max-step-rotation-deg",
                str(args.max_step_rotation_deg),
                "--depth-min-m",
                str(args.depth_min_m),
                "--depth-max-m",
                str(args.depth_max_m),
                "--poses-out",
                str(tracking_poses),
                "--report-out",
                str(tracking_report),
                "--debug-dir",
                str(tracking_debug_dir),
                "--model-cache-dir",
                str(args.model_cache_dir),
            ],
            [
                sys.executable,
                str(posegraph_script),
                "--dataset",
                str(args.dataset),
                "--report",
                str(tracking_report),
                "--poses",
                str(tracking_poses),
                "--depth-min-m",
                str(args.depth_min_m),
                "--depth-max-m",
                str(args.depth_max_m),
                "--poses-out",
                str(posegraph_poses),
                "--report-out",
                str(posegraph_report),
                "--posegraph-out",
                str(posegraph_json),
            ],
            [
                sys.executable,
                str(fusion_script),
                "--dataset",
                str(args.dataset),
                "--report",
                str(posegraph_report),
                "--poses",
                str(posegraph_poses),
                "--cloud-out",
                str(cloud_out),
                "--mesh-out",
                str(mesh_out),
                "--depth-min-m",
                str(args.depth_min_m),
                "--depth-max-m",
                str(args.depth_max_m),
                "--voxel-length-m",
                str(args.voxel_length_m),
                "--sdf-trunc-m",
                str(args.sdf_trunc_m),
            ],
        ]
    )
    return commands


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not args.dry_run:
        ensure_dataset_state(args)
        if args.device == "cuda":
            check_cuda_available()

    commands = build_commands(args)
    for command in commands:
        run_command(command, args.dry_run)

    print(flush=True)
    print(f"Pipeline complete. Output prefix: {args.out_prefix}", flush=True)


if __name__ == "__main__":
    main()

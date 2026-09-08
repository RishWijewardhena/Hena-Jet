"""Isolated Orbbec software-filter experiments; never moves the scanner motors."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np

SDK_FILTERS = (
    "DecimationFilter", "SpatialAdvancedFilter", "SpatialFastFilter",
    "SpatialModerateFilter", "TemporalFilter", "NoiseRemovalFilter",
    "EdgeNoiseRemovalFilter", "HoleFillingFilter", "FalsePositiveFilter",
    "ThresholdFilter", "LutNoiseRemovalFilter", "MgcNoiseRemovalFilter",
    "EnhancedDepthFilter",
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_value(schema, value):
    value = float(value)
    if not math.isfinite(value) or not schema.min <= value <= schema.max:
        raise ValueError(f"{schema.name}: expected {schema.min}..{schema.max}")
    if schema.step > 0:
        steps = (value - schema.min) / schema.step
        if not math.isclose(steps, round(steps), abs_tol=1e-5):
            raise ValueError(f"{schema.name}: expected step {schema.step}")
    return value


def depth_array(frame, sdk):
    depth = frame.as_depth_frame()
    if depth is None or depth.get_format() != sdk.OBFormat.Y16:
        raise RuntimeError("Filter output is not Y16 depth; refusing to interpret disparity as metres")
    raw = np.frombuffer(depth.get_data(), dtype=np.uint16)
    return raw.reshape(depth.get_height(), depth.get_width()).astype(np.float32) * (
        depth.get_depth_scale() / 1000.0
    )


def metrics(depths, minimum, maximum):
    """Repeatability and coverage only: neither establishes geometric accuracy."""
    valid = np.isfinite(depths) & (depths >= minimum) & (depths <= maximum)
    pairs = valid[1:] & valid[:-1]
    changes = np.abs(np.diff(depths, axis=0))[pairs]
    return {
        "frames": len(depths), "shape": list(depths.shape[1:]),
        "valid_fraction": float(valid.mean()),
        "median_frame_change_mm": float(np.median(changes) * 1000) if changes.size else None,
        "p95_frame_change_mm": float(np.percentile(changes, 95) * 1000) if changes.size else None,
    }


def collect_frames(pipeline, config, sdk, seconds):
    frames, errors = [], []
    lock = threading.Lock()

    def callback(frameset):
        try:
            frame = frameset.get_depth_frame()
            if frame is not None:
                clone = sdk.Frame.create_frame_from_other_frame(frame, True)
                with lock:
                    frames.append(clone)
        except Exception as exc:
            errors.append(str(exc))

    pipeline.start(config, callback)
    try:
        time.sleep(seconds)
    finally:
        pipeline.stop()
    if errors:
        raise RuntimeError(f"Frame collection failed: {errors[0]}")
    if not frames:
        raise RuntimeError("No depth frames received")
    return frames


def capture(args):
    import pyorbbecsdk as sdk
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sync_workflow"))
    from camera_controller import configure_disparity_search_range

    destination = Path(args.bag).resolve()
    if destination.exists() or destination.with_suffix(".settings.json").exists():
        raise ValueError("Recording destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pipeline = sdk.Pipeline()
    device = pipeline.get_device()
    configure_disparity_search_range(device, args.disparity)
    config = sdk.Config()
    profile = pipeline.get_stream_profile_list(sdk.OBSensorType.DEPTH_SENSOR).get_video_stream_profile(
        args.width, args.height, sdk.OBFormat.Y16, args.fps
    )
    config.enable_stream(profile)
    device.export_settings_as_preset_json_file(str(destination.with_suffix(".settings.json")))
    recorder = sdk.RecordDevice(device, str(destination))
    try:
        frames = collect_frames(pipeline, config, sdk, args.seconds)
    finally:
        del recorder
    write_json(destination.with_suffix(".capture.json"), {
        "width": args.width, "height": args.height, "fps": args.fps,
        "disparity": args.disparity, "frames": len(frames),
        "note": "Native depth only; device processing remains active. No host filters or alignment.",
    })
    print(f"Recorded {len(frames)} depth frames to {destination}")


def worker(args):
    # All native SDK work is isolated here so a native crash leaves other trials intact.
    import pyorbbecsdk as sdk
    spec = json.loads(Path(args.spec).read_text())
    device = sdk.PlaybackDevice(str(Path(args.bag).resolve()))
    # Capture settings document hardware state; replay cannot write those
    # properties. The bag supplies recorded profiles/calibration, while this
    # trial explicitly configures its host filters below.
    pipeline = sdk.Pipeline(device)
    config = sdk.Config()
    config.enable_stream(sdk.OBStreamType.DEPTH_STREAM)
    recommended = list(device.get_sensor(sdk.OBSensorType.DEPTH_SENSOR).get_recommended_filters())
    requested = spec["filters"]
    available = {f.get_name(): f for f in recommended}
    chain = []
    for f in recommended:
        if f.get_name() in requested or f.is_disparity_transform_filter():
            chain.append(f)
    for name in requested:
        if name not in available:
            if name not in SDK_FILTERS or not hasattr(sdk, name):
                raise ValueError(f"Unavailable SDK filter: {name}")
            factory = getattr(sdk, name)
            f = factory(device) if name == "EnhancedDepthFilter" else factory()
            available[name] = f
            chain.append(f)
    inventory = []
    for f in chain:
        f.enable(True)
        schemas = {s.name: s for s in f.get_config_schema_vec()}
        for name, value in requested.get(f.get_name(), {}).items():
            if name not in schemas:
                raise ValueError(f"Unknown parameter {f.get_name()}.{name}")
            f.set_config_value(name, validate_value(schemas[name], value))
        inventory.append({
            "name": f.get_name(), "recommended": f.get_name() in {r.get_name() for r in recommended},
            "parameters": [{"name": s.name, "min": s.min, "max": s.max,
                            "step": s.step, "default": s.default, "description": s.desc,
                            "active": f.get_config_value(s.name)} for s in schemas.values()],
        })
        f.reset()
    out = Path(args.output)
    write_json(out / "inventory.json", inventory)
    # Capture the complete playback before processing so slow filters do not drop frames.
    frames = collect_frames(pipeline, config, sdk, args.seconds)
    arrays, ids, intrinsics = [], [], []
    elapsed = 0.0
    for raw in frames:
        frame = sdk.Frame.create_frame_from_other_frame(raw, True)
        start = time.perf_counter()
        for f in chain:
            frame = f.process(frame)
            if frame is None:
                raise RuntimeError(f"{f.get_name()} returned no frame")
        elapsed += time.perf_counter() - start
        arrays.append(depth_array(frame, sdk))
        ids.append([int(raw.get_index()), int(raw.get_timestamp_us())])
        k = frame.get_stream_profile().as_video_stream_profile().get_intrinsic()
        intrinsics.append([k.fx, k.fy, k.cx, k.cy])
    depths = np.stack(arrays)
    np.savez_compressed(out / "depths.npz", depth_m=depths, frame_ids=ids, intrinsics=intrinsics)
    # Identical colour scale across trials; black means invalid/outside evaluation range.
    import cv2
    last = depths[-1]
    valid = np.isfinite(last) & (last >= args.min_depth) & (last <= args.max_depth)
    scaled = np.clip((np.nan_to_num(last) - args.min_depth) /
                     (args.max_depth - args.min_depth), 0, 1)
    preview = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    if not cv2.imwrite(str(out / "depth_preview.png"), preview):
        raise RuntimeError("Could not save preview")
    result = metrics(depths, args.min_depth, args.max_depth)
    result["filter_ms_per_frame"] = elapsed * 1000 / len(frames)
    result["chain"] = [f.get_name() for f in chain]
    write_json(out / "metrics.json", result)


def compare(args):
    if not Path(args.bag).is_file():
        raise ValueError(f"Recording does not exist: {args.bag}")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    specs = [{"name": "baseline", "filters": {}}]
    if args.trials:
        specs.extend(json.loads(Path(args.trials).read_text()))
    else:
        specs.extend({"name": name, "filters": {name: {}}} for name in SDK_FILTERS)
        specs.append({"name": "current", "filters": {
            name: {} for name in ("SpatialAdvancedFilter", "TemporalFilter",
                                  "NoiseRemovalFilter", "EdgeNoiseRemovalFilter")}})
    names = set()
    for spec in specs:
        name = spec["name"]
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("Trial names must contain only letters, digits, underscores or hyphens")
        if name in names or not isinstance(spec["filters"], dict):
            raise ValueError("Duplicate trial name or invalid filters object")
        names.add(name)
    summary = []
    baseline_ids = None
    for spec in specs:
        target = out / spec["name"]
        target.mkdir()
        write_json(target / "spec.json", spec)
        command = [sys.executable, str(Path(__file__).resolve()), "_worker",
                   "--bag", str(Path(args.bag).resolve()), "--output", str(target.resolve()),
                   "--spec", str((target / "spec.json").resolve()), "--seconds", str(args.seconds),
                   "--min-depth", str(args.min_depth), "--max-depth", str(args.max_depth)]
        print(f"Testing {spec['name']}…", flush=True)
        with (target / "worker.log").open("w") as log:
            try:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=args.timeout, check=False)
                status = "ok" if process.returncode == 0 else f"failed (exit {process.returncode})"
            except subprocess.TimeoutExpired:
                status = "timeout"
        row = {"name": spec["name"], "status": status}
        if status != "ok":
            lines = (target / "worker.log").read_text(errors="replace").splitlines()
            row["error"] = lines[-1] if lines else status
            row["log"] = str(target / "worker.log")
        if status == "ok":
            with np.load(target / "depths.npz") as data:
                ids = data["frame_ids"]
            if spec["name"] == "baseline":
                baseline_ids = ids
            row["same_input_frames"] = baseline_ids is not None and np.array_equal(ids, baseline_ids)
            if not row["same_input_frames"]:
                row["status"] = "invalid comparison: input frames differ or baseline failed"
            row.update(json.loads((target / "metrics.json").read_text()))
        summary.append(row)
        print(f"  {row['status']}" + (f": {row['error']}" if "error" in row else ""), flush=True)
        write_json(out / "summary.json", summary)
    print(f"Results: {out / 'summary.json'}")
    return 0 if all(row["status"] == "ok" for row in summary) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("capture", help="Record native depth without host post-processing")
    record.add_argument("--bag", required=True)
    record.add_argument("--width", type=int, default=848)
    record.add_argument("--height", type=int, default=530)
    record.add_argument("--fps", type=int, default=30)
    record.add_argument("--disparity", choices=("128", "256"), default="256")
    record.add_argument("--seconds", type=float, default=3)
    for name in ("compare", "_worker"):
        command = sub.add_parser(name)
        command.add_argument("--bag", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--seconds", type=float, default=5,
                             help="Playback collection window; use longer than recording duration")
        command.add_argument("--min-depth", type=float, default=0.04)
        command.add_argument("--max-depth", type=float, default=0.25)
        if name == "compare":
            command.add_argument("--trials", help="JSON list of named filter chains and parameter overrides")
            command.add_argument("--timeout", type=float, default=180)
        else:
            command.add_argument("--spec", required=True)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 60:
        parser.error("--seconds must be between 0 and 60; recordings are buffered in memory")
    if hasattr(args, "min_depth") and not 0 < args.min_depth < args.max_depth < math.inf:
        parser.error("Expected 0 < min-depth < max-depth, in metres")
    try:
        if args.command == "capture":
            capture(args)
        elif args.command == "compare":
            return compare(args)
        else:
            worker(args)
    except Exception as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

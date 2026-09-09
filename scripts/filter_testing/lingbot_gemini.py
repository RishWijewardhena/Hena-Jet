"""Capture one Gemini RGB-D pair and compare LingBot refinement offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def normalized_intrinsics(k, shape):
    h, w = shape
    k = np.array(k, dtype=np.float32, copy=True)
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("Expected finite 3x3 intrinsics with positive focal lengths")
    k[0] /= w
    k[1] /= h
    return k


def gate(depth, minimum, maximum):
    valid = np.isfinite(depth) & (depth >= minimum) & (depth <= maximum)
    return np.where(valid, depth, 0).astype(np.float32), valid


def rectify_rgb(rgb, color_profile, depth_profile):
    """Map colour onto the undistorted aligned-depth grid used by the model."""
    import cv2
    def matrix(profile):
        k = profile.get_intrinsic()
        return np.array([[k.fx, 0, k.cx], [0, k.fy, k.cy], [0, 0, 1]], np.float32)
    def coefficients(profile):
        d = profile.get_distortion()
        values = np.array([d.k1, d.k2, d.p1, d.p2, d.k3, d.k4, d.k5, d.k6])
        return d, values
    dd, dc = coefficients(depth_profile)
    if np.any(dc != 0):
        raise ValueError("Aligned depth must have a zero-distortion profile for LingBot pinhole intrinsics")
    cd, cc = coefficients(color_profile)
    from pyorbbecsdk import OBCameraDistortionModel as Model
    if np.any(cc != 0) and cd.model not in (Model.BROWN_CONRADY, Model.BROWN_CONRADY_K6):
        raise ValueError(f"Unsupported colour distortion model for rectification: {cd.model}")
    target_k = matrix(depth_profile)
    rectified = cv2.undistort(rgb, matrix(color_profile), cc, None, target_k)
    return rectified, target_k


def capture(args):
    import cv2
    sys.path.insert(0, str(ROOT / "scripts/sync_workflow"))
    from camera_controller import CameraController

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    with CameraController(width=args.width, height=args.height, disparity=args.disparity) as camera:
        color, depth = camera.capture_aligned_rgbd(timeout_ms=3000)
        if color is None or depth is None:
            raise RuntimeError("No synchronized RGB-D pair received")
        from pyorbbecsdk import OBFormat
        data = np.frombuffer(color.get_data(), np.uint8)
        h, w = color.get_height(), color.get_width()
        if color.get_format() == OBFormat.MJPG:
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("Cannot decode MJPG colour frame")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif color.get_format() in (OBFormat.RGB, OBFormat.BGR):
            rgb = data.reshape(h, w, 3).copy()
            if color.get_format() == OBFormat.BGR:
                rgb = rgb[..., ::-1].copy()
        else:
            raise RuntimeError(f"Unsupported colour format: {color.get_format()}")
        if depth.get_format() != OBFormat.Y16:
            raise RuntimeError("Expected aligned Y16 depth")
        depth_m = np.frombuffer(depth.get_data(), np.uint16).reshape(
            depth.get_height(), depth.get_width()).astype(np.float32) * (depth.get_depth_scale() / 1000)
        if depth_m.shape != rgb.shape[:2]:
            raise RuntimeError("RGB and aligned depth dimensions differ")
        rgb, matrix = rectify_rgb(rgb, color.get_stream_profile().as_video_stream_profile(),
                                  depth.get_stream_profile().as_video_stream_profile())
        # Store actual aligned-frame intrinsics, never substitute native depth K.
        metadata = {"width": w, "height": h, "disparity": camera.active_disparity,
                    "color_format": str(color.get_format()), "depth_scale_mm": depth.get_depth_scale(),
                    "color_timestamp_us": color.get_timestamp_us(),
                    "depth_timestamp_us": depth.get_timestamp_us(),
                    "filter_parameters": camera.active_filter_parameters,
                    "note": "SDK D2C depth, RGB rectified to depth intrinsics; current capture filters; single frame, no median fusion."}
        np.save(out / "depth_input.npy", depth_m)
        np.savetxt(out / "intrinsics.txt", matrix)
        if not cv2.imwrite(str(out / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("Could not save RGB image")
        (out / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Capture saved: {out}. Camera closed; use refine to run the model.")


def refine(args):
    import cv2
    import torch
    sys.path.insert(0, str(ROOT / "lingbot-depth"))
    from mdm.model.v2 import MDMModel

    source, out = Path(args.input), Path(args.output)
    if out.exists():
        raise ValueError("Output already exists; choose a new result directory")
    bgr = cv2.imread(str(source / "rgb.png"))
    if bgr is None:
        raise ValueError("Missing or unreadable rgb.png")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # Preserve sub-millimetre depth from our capture. PNG fallback accepts example.py datasets.
    if (source / "depth_input.npy").exists():
        raw = np.load(source / "depth_input.npy", allow_pickle=False)
    else:
        png = cv2.imread(str(source / "raw_depth.png"), cv2.IMREAD_UNCHANGED)
        if png is None or png.dtype != np.uint16:
            raise ValueError("Expected depth_input.npy in metres or uint16 raw_depth.png in millimetres")
        raw = png.astype(np.float32) / 1000
    if raw.shape != rgb.shape[:2]:
        raise ValueError("Depth and RGB must have identical dimensions")
    depth, sensor_valid = gate(raw, args.min_depth, args.max_depth)
    if not sensor_valid.any():
        raise ValueError("No sensor depth within the requested range")
    k = normalized_intrinsics(np.loadtxt(source / "intrinsics.txt"), depth.shape)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}; resolution level {args.resolution_level}", flush=True)
    model = MDMModel.from_pretrained(args.model).to(device).eval()
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].to(device).float() / 255
    start = time.perf_counter()
    with torch.inference_mode():
        result = model.infer(image, depth_in=torch.from_numpy(depth)[None].to(device),
                             intrinsics=torch.from_numpy(k)[None].to(device), apply_mask=True,
                             resolution_level=args.resolution_level, use_fp16=device == "cuda")
    prediction = result["depth"].squeeze().float().cpu().numpy()
    elapsed = time.perf_counter() - start
    if prediction.shape != depth.shape:
        raise RuntimeError("Refined depth dimensions differ from input")
    refined, refined_valid = gate(prediction, args.min_depth, args.max_depth)
    common = sensor_valid & refined_valid
    added = ~sensor_valid & refined_valid
    delta = np.where(common, (refined - depth) * 1000, np.nan)
    out.mkdir(parents=True)
    np.savez_compressed(out / "comparison.npz", depth_input_m=depth, depth_refined_m=refined,
                        model_depth_m=prediction, sensor_valid=sensor_valid,
                        refined_valid=refined_valid, model_only=added, difference_mm=delta)
    report = {"input": str(source.resolve()), "model": args.model, "device": device,
              "resolution_level": args.resolution_level, "inference_seconds": elapsed,
              "min_depth_m": args.min_depth, "max_depth_m": args.max_depth,
              "sensor_coverage": float(sensor_valid.mean()), "refined_coverage": float(refined_valid.mean()),
              "model_only_pixels": int(added.sum()),
              "median_absolute_change_mm": float(np.median(np.abs(delta[common]))) if common.any() else None,
              "note": "Change from sensor depth is not accuracy against ground truth."}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("black")
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB")
    for ax, values, valid, title in ((axes[0, 1], depth, sensor_valid, "Sensor depth"),
                                    (axes[0, 2], refined, refined_valid, "LingBot refined depth")):
        im = ax.imshow(np.ma.array(values * 1000, mask=~valid), cmap=cmap,
                       vmin=args.min_depth * 1000, vmax=args.max_depth * 1000)
        ax.set_title(title)
    fig.colorbar(im, ax=list(axes[0, 1:]), label="Depth (mm)", shrink=0.7)
    change = axes[1, 0].imshow(delta, cmap="coolwarm", vmin=-10, vmax=10)
    axes[1, 0].set_title("Refined − sensor (common valid pixels)")
    fig.colorbar(change, ax=axes[1, 0], label="Change (mm), clipped at ±10")
    axes[1, 1].imshow(added, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("White: model-only depth")
    axes[1, 2].imshow(sensor_valid & ~refined_valid, cmap="gray", vmin=0, vmax=1)
    axes[1, 2].set_title("White: sensor depth removed")
    for ax in axes.flat:
        ax.set_axis_off()
    fig.suptitle("Gemini 305 / LingBot test — visual smoothness does not establish accuracy")
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)
    print(f"Results saved: {out / 'comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--output", required=True)
    cap.add_argument("--width", type=int, default=1280)
    cap.add_argument("--height", type=int, default=800)
    cap.add_argument("--disparity", choices=("128", "256"), default="128")
    ref = sub.add_parser("refine")
    ref.add_argument("--input", required=True)
    ref.add_argument("--output", required=True)
    ref.add_argument("--model", default="robbyant/lingbot-depth-pretrain-vitl-14-v0.5")
    ref.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ref.add_argument("--resolution-level", type=int, choices=range(10), default=0,
                     help="0 uses the smallest token budget; 9 matches example.py")
    ref.add_argument("--min-depth", type=float, default=0.095)
    ref.add_argument("--max-depth", type=float, default=0.25)
    args = parser.parse_args()
    if args.command == "refine" and not 0 < args.min_depth < args.max_depth < float("inf"):
        parser.error("Expected 0 < min-depth < max-depth in metres")
    if args.command == "capture":
        capture(args)
    else:
        refine(args)


if __name__ == "__main__":
    main()

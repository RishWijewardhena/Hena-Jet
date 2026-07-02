#!/usr/bin/env python3
"""Generate a printable 12-marker ArUco table board for ZED-M alignment."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aruco_common import get_aruco_dictionary, write_json


MM_PER_M = 1000.0
DEFAULT_MARKER_SIZE_M = 0.025
DEFAULT_DPI = 300
DEFAULT_OUT_DIR = Path("outputs/aruco_table_board")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a printable ArUco table board.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dictionary", default="DICT_5X5_100")
    parser.add_argument("--marker-size-m", type=float, default=DEFAULT_MARKER_SIZE_M)
    parser.add_argument("--marker-pixels", type=int, default=600)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--paper", choices=["a4", "letter"], default="a4")
    parser.add_argument(
        "--quiet-zone-m",
        type=float,
        default=0.006,
        help="White space around each printed marker.",
    )
    parser.add_argument(
        "--central-empty-width-m",
        type=float,
        default=0.130,
        help="Printed guide rectangle width for the object/hand area.",
    )
    parser.add_argument(
        "--central-empty-height-m",
        type=float,
        default=0.160,
        help="Printed guide rectangle height for the object/hand area.",
    )
    return parser.parse_args()


def paper_size_m(name: str) -> tuple[float, float]:
    if name == "a4":
        return 0.210, 0.297
    if name == "letter":
        return 0.2159, 0.2794
    raise ValueError(f"Unsupported paper size: {name}")


def marker_centers_m() -> list[tuple[int, float, float]]:
    xs = [-0.060, -0.020, 0.020, 0.060]
    top_y = 0.110
    bottom_y = -0.110
    side_x = 0.085
    side_ys = [0.035, -0.035]

    centers: list[tuple[int, float, float]] = []
    marker_id = 0
    for x in xs:
        centers.append((marker_id, x, top_y))
        marker_id += 1
    for y in side_ys:
        centers.append((marker_id, side_x, y))
        marker_id += 1
    for x in reversed(xs):
        centers.append((marker_id, x, bottom_y))
        marker_id += 1
    for y in reversed(side_ys):
        centers.append((marker_id, -side_x, y))
        marker_id += 1
    return centers


def marker_corners_m(cx: float, cy: float, marker_size_m: float) -> list[list[float]]:
    half = marker_size_m / 2.0
    return [
        [cx - half, cy + half, 0.0],
        [cx + half, cy + half, 0.0],
        [cx + half, cy - half, 0.0],
        [cx - half, cy - half, 0.0],
    ]


def meters_to_pixels(value_m: float, dpi: int) -> int:
    return int(round(value_m / 0.0254 * dpi))


def board_to_pixel(
    x_m: float,
    y_m: float,
    paper_width_px: int,
    paper_height_px: int,
    dpi: int,
) -> tuple[int, int]:
    px_per_m = dpi / 0.0254
    return (
        int(round(paper_width_px / 2.0 + x_m * px_per_m)),
        int(round(paper_height_px / 2.0 - y_m * px_per_m)),
    )


def create_marker_image(dictionary, marker_id: int, marker_pixels: int) -> Image.Image:
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_pixels)
    return Image.fromarray(marker).convert("L")


def draw_center_guides(
    draw: ImageDraw.ImageDraw,
    paper_width_px: int,
    paper_height_px: int,
    dpi: int,
    central_empty_width_m: float,
    central_empty_height_m: float,
) -> None:
    half_w = central_empty_width_m / 2.0
    half_h = central_empty_height_m / 2.0
    x0, y0 = board_to_pixel(-half_w, half_h, paper_width_px, paper_height_px, dpi)
    x1, y1 = board_to_pixel(half_w, -half_h, paper_width_px, paper_height_px, dpi)
    draw.rectangle([x0, y0, x1, y1], outline=(180,), width=2)
    cx, cy = board_to_pixel(0.0, 0.0, paper_width_px, paper_height_px, dpi)
    draw.line([cx - 20, cy, cx + 20, cy], fill=(140,), width=2)
    draw.line([cx, cy - 20, cx, cy + 20], fill=(140,), width=2)
    draw.text((x0, y1 + 8), "scan center / keep hand-object inside", fill=(120,))


def main() -> None:
    args = parse_args()
    if args.marker_size_m <= 0.0:
        raise ValueError("--marker-size-m must be positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    markers_dir = args.out_dir / "markers"
    markers_dir.mkdir(parents=True, exist_ok=True)

    dictionary = get_aruco_dictionary(args.dictionary)
    paper_width_m, paper_height_m = paper_size_m(args.paper)
    paper_width_px = meters_to_pixels(paper_width_m, args.dpi)
    paper_height_px = meters_to_pixels(paper_height_m, args.dpi)
    marker_size_px = meters_to_pixels(args.marker_size_m, args.dpi)
    quiet_zone_px = meters_to_pixels(args.quiet_zone_m, args.dpi)

    board_image = Image.new("L", (paper_width_px, paper_height_px), color=255)
    draw = ImageDraw.Draw(board_image)
    draw_center_guides(
        draw,
        paper_width_px,
        paper_height_px,
        args.dpi,
        args.central_empty_width_m,
        args.central_empty_height_m,
    )

    markers = []
    for marker_id, cx_m, cy_m in marker_centers_m():
        marker_image = create_marker_image(dictionary, marker_id, args.marker_pixels)
        marker_image.save(markers_dir / f"marker_{marker_id:02d}.png")
        marker_image = marker_image.resize((marker_size_px, marker_size_px), Image.Resampling.NEAREST)

        marker_center_px = board_to_pixel(cx_m, cy_m, paper_width_px, paper_height_px, args.dpi)
        x0 = marker_center_px[0] - marker_size_px // 2
        y0 = marker_center_px[1] - marker_size_px // 2
        quiet_rect = [
            x0 - quiet_zone_px,
            y0 - quiet_zone_px,
            x0 + marker_size_px + quiet_zone_px,
            y0 + marker_size_px + quiet_zone_px,
        ]
        draw.rectangle(quiet_rect, fill=(255,))
        board_image.paste(marker_image, (x0, y0))
        draw.text((x0, y0 + marker_size_px + 4), f"id {marker_id}", fill=(0,))

        markers.append(
            {
                "id": marker_id,
                "center_m": [cx_m, cy_m, 0.0],
                "corners_m": marker_corners_m(cx_m, cy_m, args.marker_size_m),
            }
        )

    title = "ZED ArUco table board - print at 100% scale"
    subtitle = f"{args.dictionary}, marker size {args.marker_size_m * MM_PER_M:.1f} mm"
    draw.text((40, 36), title, fill=(0,))
    draw.text((40, 58), subtitle, fill=(0,))
    draw.text((40, paper_height_px - 70), "Measure a marker after printing: side must be 25 mm.", fill=(0,))

    png_path = args.out_dir / "aruco_table_board.png"
    pdf_path = args.out_dir / "aruco_table_board.pdf"
    json_path = args.out_dir / "aruco_table_board.json"
    board_image.save(png_path)
    board_image.convert("RGB").save(pdf_path, "PDF", resolution=float(args.dpi))

    write_json(
        json_path,
        {
            "dictionary": args.dictionary,
            "marker_size_m": args.marker_size_m,
            "marker_count": len(markers),
            "paper": args.paper,
            "dpi": args.dpi,
            "world_frame": {
                "units": "meters",
                "origin": "center of the printed scan area",
                "x_axis": "right on printed board",
                "y_axis": "up on printed board",
                "z_axis": "out of table plane toward camera",
            },
            "markers": markers,
        },
    )

    print(f"PDF: {pdf_path}")
    print(f"PNG: {png_path}")
    print(f"JSON: {json_path}")
    print(f"Individual markers: {markers_dir}")


if __name__ == "__main__":
    main()

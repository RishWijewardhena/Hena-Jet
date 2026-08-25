#!/usr/bin/env python3
"""Generate the printable four-face ArUco orbit-radius calibration kit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from radius_calibration import build_profile_marker_map


MM_PER_INCH = 25.4
A4_SIZE_MM = (210.0, 297.0)
DEFAULT_OUTPUT_DIR = Path("outputs/radius_markers")


def mm_to_px(value_mm: float, dpi: int) -> int:
    return int(round(value_mm / MM_PER_INCH * dpi))


def _font(size_mm: float, dpi: int):
    return ImageFont.truetype("DejaVuSans.ttf", mm_to_px(size_mm, dpi))


def _text_size(draw: ImageDraw.ImageDraw, text: str, font=None) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font or ImageFont.load_default())
    return box[2] - box[0], box[3] - box[1]


def _make_carrier(
    dictionary,
    marker_id: int,
    *,
    marker_size_mm: float,
    carrier_size_mm: float,
    dpi: int,
) -> Image.Image:
    carrier_px = mm_to_px(carrier_size_mm, dpi)
    marker_px = mm_to_px(marker_size_mm, dpi)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)
    image = Image.new("L", (carrier_px, carrier_px), 255)
    offset = (carrier_px - marker_px) // 2
    image.paste(Image.fromarray(marker), (offset, offset))
    return image


def _draw_placement_guide(marker_map: dict[str, Any], output_path: Path) -> None:
    image = Image.new("RGB", (1500, 1050), "white")
    draw = ImageDraw.Draw(image)
    draw.text((45, 35), "20 x 40 mm profile marker placement", fill="black")
    draw.text(
        (45, 65),
        "Z points up the bar. Match every ID, face name, and UP arrow.",
        fill="black",
    )

    face_x = {"front": 220, "right": 520, "back": 820, "left": 1120}
    top = 150
    bottom = 900
    for face, x in face_x.items():
        width = 220 if face in {"front", "back"} else 130
        draw.rectangle((x - width // 2, top, x + width // 2, bottom), outline="gray", width=4)
        draw.text((x - 30, bottom + 15), face.upper(), fill="black")
        draw.line((x, top + 55, x, top + 5), fill="green", width=6)
        draw.polygon([(x, top - 15), (x - 12, top + 10), (x + 12, top + 10)], fill="green")
        draw.text((x + 16, top + 5), "+Z / UP", fill="green")

    z_min = -0.060
    z_max = 0.060
    for marker in marker_map["markers"]:
        face = marker["face"]
        z_m = marker["center_m"][2]
        fraction = (z_max - z_m) / (z_max - z_min)
        y = int(round(top + 95 + fraction * (bottom - top - 180)))
        x = face_x[face]
        draw.rectangle((x - 48, y - 48, x + 48, y + 48), outline="black", width=4)
        label = f"ID {marker['id']}  Z={z_m * 1000:+.0f} mm"
        text_width, _ = _text_size(draw, label)
        draw.text((x - text_width // 2, y + 58), label, fill="black")

    draw.text(
        (45, 995),
        "Reference: Z=0 is halfway between the four marker levels; successive centers are 30 mm apart.",
        fill="black",
    )
    image.save(output_path)


def generate_marker_kit(
    output_dir: Path,
    *,
    dpi: int = 600,
    marker_size_mm: float = 18.0,
    carrier_size_mm: float = 26.0,
    profile_width_mm: float = 40.0,
    profile_depth_mm: float = 20.0,
    carrier_offset_mm: float = 1.0,
    axial_offsets_mm: tuple[float, float, float, float] = (-45.0, -15.0, 15.0, 45.0),
) -> dict[str, Any]:
    """Write a print-ready marker kit and return all generated paths."""
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if carrier_size_mm <= marker_size_mm:
        raise ValueError("carrier_size_mm must be larger than marker_size_mm")

    output_dir = Path(output_dir)
    markers_dir = output_dir / "markers"
    markers_dir.mkdir(parents=True, exist_ok=True)

    marker_map = build_profile_marker_map(
        marker_size_m=marker_size_mm * 0.001,
        profile_width_m=profile_width_mm * 0.001,
        profile_depth_m=profile_depth_mm * 0.001,
        carrier_offset_m=carrier_offset_mm * 0.001,
        axial_offsets_m=[value * 0.001 for value in axial_offsets_mm],
    )
    marker_map.update(
        {
            "carrier_size_m": carrier_size_mm * 0.001,
            "print_dpi": dpi,
            "print_instruction": "Print the PDF at Actual size / 100%; never use Fit to page.",
        }
    )

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    carrier_images: list[Image.Image] = []
    marker_paths: list[Path] = []
    for marker in marker_map["markers"]:
        carrier = _make_carrier(
            dictionary,
            marker["id"],
            marker_size_mm=marker_size_mm,
            carrier_size_mm=carrier_size_mm,
            dpi=dpi,
        )
        marker_path = markers_dir / f"marker_{marker['id']}_{marker['face']}.png"
        carrier.save(marker_path, dpi=(dpi, dpi))
        carrier_images.append(carrier)
        marker_paths.append(marker_path)

    page_width = mm_to_px(A4_SIZE_MM[0], dpi)
    page_height = mm_to_px(A4_SIZE_MM[1], dpi)
    page = Image.new("L", (page_width, page_height), 255)
    draw = ImageDraw.Draw(page)
    margin = mm_to_px(15.0, dpi)
    title_font = _font(4.0, dpi)
    warning_font = _font(3.2, dpi)
    body_font = _font(2.6, dpi)
    label_font = _font(2.4, dpi)
    draw.text(
        (margin, margin),
        "Four-face ArUco orbit-radius calibration kit",
        fill=0,
        font=title_font,
    )
    draw.text(
        (margin, margin + mm_to_px(6.0, dpi)),
        "PRINT AT ACTUAL SIZE / 100% - DO NOT FIT TO PAGE",
        fill=0,
        font=warning_font,
    )
    draw.text(
        (margin, margin + mm_to_px(11.0, dpi)),
        "18.0 mm coded square | 26.0 mm carrier | DICT_5X5_100",
        fill=0,
        font=body_font,
    )

    start_y = margin + mm_to_px(18.0, dpi)
    column_x = [margin, margin + mm_to_px(55.0, dpi)]
    row_y = [start_y, start_y + mm_to_px(55.0, dpi)]
    carrier_px = mm_to_px(carrier_size_mm, dpi)
    for index, (marker, carrier) in enumerate(zip(marker_map["markers"], carrier_images)):
        x = column_x[index % 2]
        y = row_y[index // 2]
        page.paste(carrier, (x, y))
        draw.rectangle((x, y, x + carrier_px - 1, y + carrier_px - 1), outline=128, width=2)
        draw.text(
            (x, y + carrier_px + mm_to_px(1.5, dpi)),
            f"ID {marker['id']} - {marker['face'].upper()} - UP toward +Z",
            fill=0,
            font=label_font,
        )

    ruler_x = margin
    ruler_y = row_y[-1] + carrier_px + mm_to_px(25.0, dpi)
    ruler_length = mm_to_px(100.0, dpi)
    draw.line((ruler_x, ruler_y, ruler_x + ruler_length, ruler_y), fill=0, width=4)
    for value_mm in range(0, 101, 10):
        x = ruler_x + mm_to_px(float(value_mm), dpi)
        tick = mm_to_px(4.0 if value_mm % 50 else 7.0, dpi)
        draw.line((x, ruler_y - tick, x, ruler_y + tick), fill=0, width=3)
        draw.text(
            (x - mm_to_px(1.0, dpi), ruler_y + tick + mm_to_px(1.0, dpi)),
            str(value_mm),
            fill=0,
            font=label_font,
        )
    draw.text(
        (ruler_x, ruler_y + mm_to_px(13.0, dpi)),
        "100 mm verification ruler",
        fill=0,
        font=body_font,
    )

    png_path = output_dir / "profile_radius_markers_a4.png"
    pdf_path = output_dir / "profile_radius_markers_a4.pdf"
    map_path = output_dir / "profile_marker_map.json"
    placement_path = output_dir / "profile_marker_placement.png"
    page.save(png_path, dpi=(dpi, dpi))
    page.convert("RGB").save(pdf_path, "PDF", resolution=float(dpi))
    map_path.write_text(json.dumps(marker_map, indent=2) + "\n", encoding="utf-8")
    _draw_placement_guide(marker_map, placement_path)

    return {
        "pdf": pdf_path,
        "png": png_path,
        "map": map_path,
        "placement_guide": placement_path,
        "markers": marker_paths,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    paths = generate_marker_kit(args.output_dir, dpi=args.dpi)
    print(f"PDF: {paths['pdf']}")
    print(f"Marker map: {paths['map']}")
    print(f"Placement guide: {paths['placement_guide']}")
    print("Print the PDF at Actual size / 100%; do not use Fit to page.")


if __name__ == "__main__":
    main()

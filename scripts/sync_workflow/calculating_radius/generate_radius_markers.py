#!/usr/bin/env python3
"""Generate an exact A4 ArUco wrap for a horizontal 20 x 40 mm profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
from PIL import Image, ImageDraw, ImageFont

# Make the calculating_radius package importable when this file is executed
# directly from the repository root.
SYNC_WORKFLOW_DIR = Path(__file__).resolve().parents[1]
if str(SYNC_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(SYNC_WORKFLOW_DIR))

from calculating_radius.radius_calibration import build_profile_marker_map


MM_PER_INCH = 25.4
A4_SIZE_MM = (210.0, 297.0)
DEFAULT_OUTPUT_DIR = Path("outputs/radius_markers")
WRAP_LENGTH_MM = 130.0
WRAP_HEIGHT_MM = 40.0
FOLD_POSITIONS_MM = (10.0, 30.0, 70.0, 90.0)
MARKER_CENTERS_MM = {
    3: (20.0, 20.0),   # bottom, 14 mm
    0: (50.0, 20.0),   # front, 18 mm
    1: (80.0, 20.0),   # top, 14 mm
    2: (110.0, 20.0),  # back, 18 mm
}
FACE_RANGES_MM = (
    ("glue tab", 0.0, 10.0),
    ("bottom / ID 3", 10.0, 30.0),
    ("front / ID 0", 30.0, 70.0),
    ("top / ID 1", 70.0, 90.0),
    ("back / ID 2", 90.0, 130.0),
)


def mm_to_px(value_mm: float, dpi: int) -> int:
    return int(round(value_mm / MM_PER_INCH * dpi))


def _font(size_mm: float, dpi: int):
    return ImageFont.truetype("DejaVuSans.ttf", mm_to_px(size_mm, dpi))


def _dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill,
    width: int,
    dash_px: int,
) -> None:
    x0, y0 = start
    x1, y1 = end
    if x0 == x1:
        for y in range(y0, y1, dash_px * 2):
            draw.line((x0, y, x1, min(y + dash_px, y1)), fill=fill, width=width)
    elif y0 == y1:
        for x in range(x0, x1, dash_px * 2):
            draw.line((x, y0, min(x + dash_px, x1), y1), fill=fill, width=width)
    else:
        raise ValueError("Only horizontal and vertical dashed lines are supported")


def _marker_image(dictionary, marker_id: int, size_mm: float, dpi: int) -> Image.Image:
    size_px = mm_to_px(size_mm, dpi)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    return Image.fromarray(marker).convert("L")


def _draw_centerline_without_markers(
    draw: ImageDraw.ImageDraw,
    *,
    dpi: int,
    marker_sizes_mm: dict[int, float],
) -> None:
    y = mm_to_px(WRAP_HEIGHT_MM / 2.0, dpi)
    excluded = []
    for marker_id, (center_x_mm, _) in MARKER_CENTERS_MM.items():
        half = marker_sizes_mm[marker_id] / 2.0 + 4.0
        excluded.append((center_x_mm - half, center_x_mm + half))
    cursor_mm = 0.0
    for start_mm, end_mm in sorted(excluded):
        if start_mm > cursor_mm:
            _dashed_line(
                draw,
                (mm_to_px(cursor_mm, dpi), y),
                (mm_to_px(start_mm, dpi), y),
                fill=175,
                width=max(1, mm_to_px(0.15, dpi)),
                dash_px=max(2, mm_to_px(1.2, dpi)),
            )
        cursor_mm = max(cursor_mm, end_mm)
    if cursor_mm < WRAP_LENGTH_MM:
        _dashed_line(
            draw,
            (mm_to_px(cursor_mm, dpi), y),
            (mm_to_px(WRAP_LENGTH_MM, dpi), y),
            fill=175,
            width=max(1, mm_to_px(0.15, dpi)),
            dash_px=max(2, mm_to_px(1.2, dpi)),
        )


def _create_wrap_image(marker_map: dict[str, Any], dpi: int) -> Image.Image:
    width_px = mm_to_px(WRAP_LENGTH_MM, dpi)
    height_px = mm_to_px(WRAP_HEIGHT_MM, dpi)
    wrap = Image.new("L", (width_px, height_px), 255)
    draw = ImageDraw.Draw(wrap)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    marker_sizes_mm = {
        int(marker["id"]): float(marker["size_m"]) * 1000.0
        for marker in marker_map["markers"]
    }

    glue_end_px = mm_to_px(10.0, dpi)
    glue_zone = Image.new("L", (glue_end_px, height_px), 255)
    glue_draw = ImageDraw.Draw(glue_zone)
    hatch_step = max(4, mm_to_px(2.0, dpi))
    for offset in range(-height_px, glue_end_px + height_px, hatch_step):
        glue_draw.line(
            (offset, height_px, offset + height_px, 0),
            fill=235,
            width=1,
        )
    wrap.paste(glue_zone, (0, 0))

    _draw_centerline_without_markers(
        draw,
        dpi=dpi,
        marker_sizes_mm=marker_sizes_mm,
    )

    fold_width = max(1, mm_to_px(0.20, dpi))
    dash_px = max(3, mm_to_px(1.5, dpi))
    for fold_mm in FOLD_POSITIONS_MM:
        x = mm_to_px(fold_mm, dpi)
        _dashed_line(
            draw,
            (x, 0),
            (x, height_px - 1),
            fill=105,
            width=fold_width,
            dash_px=dash_px,
        )

    for marker in marker_map["markers"]:
        marker_id = int(marker["id"])
        center_x_mm, center_y_mm = MARKER_CENTERS_MM[marker_id]
        marker_size_mm = marker_sizes_mm[marker_id]
        image = _marker_image(dictionary, marker_id, marker_size_mm, dpi)
        x = mm_to_px(center_x_mm, dpi) - image.width // 2
        y = mm_to_px(center_y_mm, dpi) - image.height // 2
        wrap.paste(image, (x, y))

    label_font = _font(1.8, dpi)
    labels = {
        3: "BOTTOM  ID 3  14 mm",
        0: "FRONT  ID 0  18 mm",
        1: "TOP  ID 1  14 mm",
        2: "BACK  ID 2  18 mm",
    }
    for marker_id, label in labels.items():
        center_x_mm, _ = MARKER_CENTERS_MM[marker_id]
        box = draw.textbbox((0, 0), label, font=label_font)
        text_width = box[2] - box[0]
        draw.text(
            (mm_to_px(center_x_mm, dpi) - text_width // 2, mm_to_px(1.5, dpi)),
            label,
            fill=60,
            font=label_font,
        )

    glue_font = _font(1.6, dpi)
    draw.text(
        (mm_to_px(1.0, dpi), mm_to_px(34.0, dpi)),
        "GLUE",
        fill=90,
        font=glue_font,
    )
    return wrap


def _draw_ruler(draw: ImageDraw.ImageDraw, x: int, y: int, dpi: int) -> None:
    length = mm_to_px(100.0, dpi)
    label_font = _font(2.2, dpi)
    draw.line((x, y, x + length, y), fill=0, width=max(2, mm_to_px(0.2, dpi)))
    for value_mm in range(0, 101, 10):
        tick_x = x + mm_to_px(float(value_mm), dpi)
        tick = mm_to_px(5.0 if value_mm % 50 else 7.0, dpi)
        draw.line((tick_x, y - tick, tick_x, y + tick), fill=0, width=2)
        draw.text(
            (tick_x - mm_to_px(1.0, dpi), y + tick + mm_to_px(1.0, dpi)),
            str(value_mm),
            fill=0,
            font=label_font,
        )
    draw.text(
        (x, y + mm_to_px(14.0, dpi)),
        "100 mm verification ruler",
        fill=0,
        font=_font(2.6, dpi),
    )


def _draw_placement_guide(output_path: Path) -> None:
    image = Image.new("RGB", (1800, 950), "white")
    draw = ImageDraw.Draw(image)
    title = ImageFont.truetype("DejaVuSans.ttf", 38)
    body = ImageFont.truetype("DejaVuSans.ttf", 24)
    small = ImageFont.truetype("DejaVuSans.ttf", 20)
    draw.text((45, 30), "Continuous wrap for horizontal 20 x 40 mm profile", fill="black", font=title)
    draw.text(
        (45, 85),
        "All marker centers share one line along the bar. The top strip edge points to the chosen RIGHT end.",
        fill="black",
        font=body,
    )

    scale = 10
    x0, y0 = 90, 240
    width, height = int(WRAP_LENGTH_MM * scale), int(WRAP_HEIGHT_MM * scale)
    draw.rectangle((x0, y0, x0 + width, y0 + height), outline="red", width=4)
    draw.rectangle((x0, y0, x0 + 100, y0 + height), fill=(240, 240, 240), outline="gray")
    for fold_mm in FOLD_POSITIONS_MM:
        x = x0 + int(fold_mm * scale)
        for y in range(y0, y0 + height, 24):
            draw.line((x, y, x, min(y + 12, y0 + height)), fill="blue", width=3)

    marker_sizes = {0: 18, 1: 14, 2: 18, 3: 14}
    for marker_id, (cx_mm, cy_mm) in MARKER_CENTERS_MM.items():
        half = marker_sizes[marker_id] * scale / 2
        cx, cy = x0 + cx_mm * scale, y0 + cy_mm * scale
        draw.rectangle((cx - half, cy - half, cx + half, cy + half), fill="black")
        draw.text((cx - 28, cy - 12), f"ID {marker_id}", fill="white", font=small)

    for face, start_mm, end_mm in FACE_RANGES_MM:
        cx = x0 + int((start_mm + end_mm) / 2.0 * scale)
        draw.text((cx - 55, y0 + height + 22), face.upper(), fill="black", font=small)
    draw.text((x0, y0 - 70), "CUT: solid red outline", fill="red", font=body)
    draw.text((x0 + 470, y0 - 70), "FOLD: dashed blue lines", fill="blue", font=body)
    draw.line((x0 + width + 80, y0 + height, x0 + width + 80, y0), fill="green", width=7)
    draw.polygon(
        [
            (x0 + width + 80, y0 - 28),
            (x0 + width + 64, y0 + 8),
            (x0 + width + 96, y0 + 8),
        ],
        fill="green",
    )
    draw.text((x0 + width + 105, y0 + 15), "+Z / RIGHT END", fill="green", font=body)
    draw.text(
        (90, 820),
        "Wrap order from the lower-rear seam: glue tab -> bottom -> front -> top -> back.",
        fill="black",
        font=body,
    )
    image.save(output_path)


def generate_marker_kit(
    output_dir: Path,
    *,
    dpi: int = 600,
    profile_width_mm: float = 40.0,
    profile_depth_mm: float = 20.0,
    carrier_offset_mm: float = 0.1,
) -> dict[str, Any]:
    """Write the replacement continuous wrap, marker map, and guide."""
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if profile_width_mm != 40.0 or profile_depth_mm != 20.0:
        raise ValueError("This exact wrap is designed for a 20 x 40 mm profile")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    marker_map = build_profile_marker_map(
        marker_sizes_m=[0.018, 0.014, 0.018, 0.014],
        profile_width_m=profile_width_mm * 0.001,
        profile_depth_m=profile_depth_mm * 0.001,
        carrier_offset_m=carrier_offset_mm * 0.001,
        axial_offsets_m=[0.0, 0.0, 0.0, 0.0],
    )
    marker_map.update(
        {
            "print_dpi": dpi,
            "print_instruction": "Print at Actual size / 100%; never use Fit to page.",
            "wrap": {
                "material": "ordinary A4 paper attached with a thin, even glue layer",
                "size_m": [0.130, 0.040],
                "profile_perimeter_m": 0.120,
                "overlap_tab_m": 0.010,
                "fold_positions_m": [0.010, 0.030, 0.070, 0.090],
                "seam": "lower-rear profile corner",
                "marker_centers_unwrapped_m": {
                    str(marker_id): [x_mm * 0.001, y_mm * 0.001]
                    for marker_id, (x_mm, y_mm) in MARKER_CENTERS_MM.items()
                },
                "positive_profile_axis": "top edge of the strip toward the chosen right end",
            },
        }
    )

    wrap = _create_wrap_image(marker_map, dpi)
    wrap_preview_path = output_dir / "profile_radius_wrap.png"
    wrap.save(wrap_preview_path, dpi=(dpi, dpi))

    page_width = mm_to_px(A4_SIZE_MM[0], dpi)
    page_height = mm_to_px(A4_SIZE_MM[1], dpi)
    page = Image.new("L", (page_width, page_height), 255)
    draw = ImageDraw.Draw(page)
    margin = mm_to_px(15.0, dpi)
    draw.text(
        (margin, margin),
        "Exact ArUco wrap - horizontal 20 x 40 mm profile",
        fill=0,
        font=_font(4.0, dpi),
    )
    draw.text(
        (margin, margin + mm_to_px(6.0, dpi)),
        "PRINT AT ACTUAL SIZE / 100% - DO NOT FIT TO PAGE",
        fill=0,
        font=_font(3.2, dpi),
    )
    draw.text(
        (margin, margin + mm_to_px(11.0, dpi)),
        "Solid outline = CUT | dashed lines = FOLD | centers lie on one longitudinal position",
        fill=0,
        font=_font(2.4, dpi),
    )

    wrap_x = margin
    wrap_y = margin + mm_to_px(24.0, dpi)
    page.paste(wrap, (wrap_x, wrap_y))
    cut_width = max(2, mm_to_px(0.25, dpi))
    draw.rectangle(
        (wrap_x, wrap_y, wrap_x + wrap.width - 1, wrap_y + wrap.height - 1),
        outline=0,
        width=cut_width,
    )
    draw.text(
        (wrap_x, wrap_y - mm_to_px(5.0, dpi)),
        "The TOP edge of this strip must face the RIGHT end of the horizontal bar",
        fill=0,
        font=_font(2.4, dpi),
    )
    arrow_x = wrap_x + wrap.width + mm_to_px(7.0, dpi)
    draw.line(
        (arrow_x, wrap_y + mm_to_px(15.0, dpi), arrow_x, wrap_y),
        fill=0,
        width=max(3, mm_to_px(0.4, dpi)),
    )
    draw.polygon(
        [
            (arrow_x, wrap_y - mm_to_px(3.0, dpi)),
            (arrow_x - mm_to_px(1.5, dpi), wrap_y + mm_to_px(1.0, dpi)),
            (arrow_x + mm_to_px(1.5, dpi), wrap_y + mm_to_px(1.0, dpi)),
        ],
        fill=0,
    )

    instruction_y = wrap_y + wrap.height + mm_to_px(12.0, dpi)
    instructions = [
        "1. Verify the ruler and marker sizes before cutting.",
        "2. Cut only the solid outer rectangle; do not cut around markers.",
        "3. Pre-crease every dashed fold line, keeping each marker surface flat.",
        "4. Start at the lower-rear corner and wrap: bottom, front, top, back.",
        "5. Apply a thin even glue layer; avoid bubbles, wrinkles, and glossy tape over markers.",
    ]
    body_font = _font(2.5, dpi)
    for index, instruction in enumerate(instructions):
        draw.text(
            (margin, instruction_y + index * mm_to_px(5.0, dpi)),
            instruction,
            fill=0,
            font=body_font,
        )
    _draw_ruler(draw, margin, instruction_y + mm_to_px(42.0, dpi), dpi)

    png_path = output_dir / "profile_radius_markers_a4.png"
    pdf_path = output_dir / "profile_radius_markers_a4.pdf"
    map_path = output_dir / "profile_marker_map.json"
    placement_path = output_dir / "profile_marker_placement.png"
    page.save(png_path, dpi=(dpi, dpi))
    page.convert("RGB").save(pdf_path, "PDF", resolution=float(dpi))
    map_path.write_text(json.dumps(marker_map, indent=2) + "\n", encoding="utf-8")
    _draw_placement_guide(placement_path)
    return {
        "pdf": pdf_path,
        "png": png_path,
        "map": map_path,
        "placement_guide": placement_path,
        "wrap_preview": wrap_preview_path,
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
    print(f"Exact wrap preview: {paths['wrap_preview']}")
    print(f"Marker map: {paths['map']}")
    print(f"Placement guide: {paths['placement_guide']}")
    print("Print at Actual size / 100%; do not use Fit to page.")


if __name__ == "__main__":
    main()

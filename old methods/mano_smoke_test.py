#!/usr/bin/env python3
"""Load the local MANO models and export neutral hand meshes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "Mano" / "mano_v1_2" / "models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mano"


def generate_neutral_hand(
    model_dir: str | Path,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return neutral MANO vertices and triangle faces for one hand."""
    normalized_side = side.lower()
    if normalized_side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")

    model_path = Path(model_dir) / f"MANO_{normalized_side.upper()}.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(f"MANO model not found: {model_path}")

    try:
        import chumpy  # noqa: F401 - required to unpickle MANO 1.2 models
        import smplx
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "MANO 1.2 requires torch, smplx, and chumpy. "
            "Activate the hena_jet environment."
        ) from exc

    model = smplx.MANO(
        model_path=str(model_path),
        is_rhand=normalized_side == "right",
        use_pca=False,
        flat_hand_mean=True,
    )
    with torch.no_grad():
        output = model(return_verts=True)

    vertices = output.vertices[0].detach().cpu().numpy().astype(np.float64)
    faces = np.asarray(model.faces, dtype=np.int64)
    return vertices, faces


def export_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: str | Path,
    *,
    show: bool = False,
) -> Path:
    """Export a triangle mesh and optionally display it with trimesh."""
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "Mesh export requires trimesh. Activate the hena_jet environment."
        ) from exc

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(path)
    if show:
        mesh.show()
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Directory containing MANO_LEFT.pkl and MANO_RIGHT.pkl (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Mesh output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--side",
        choices=("left", "right", "both"),
        default="both",
        help="Hand model to generate (default: both)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open each generated mesh in the trimesh viewer",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sides = ("left", "right") if args.side == "both" else (args.side,)

    for side in sides:
        vertices, faces = generate_neutral_hand(args.model_dir, side)
        output_path = export_mesh(
            vertices,
            faces,
            args.output_dir / f"mano_{side}_neutral.obj",
            show=args.show,
        )
        print(
            f"{side}: {len(vertices)} vertices, {len(faces)} faces -> {output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""TUM-style RGB-D input without consuming an external trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class RgbdFrame:
    timestamp_s: float
    color_rgb: np.ndarray
    depth: np.ndarray


class TumRgbdDataset:
    """Read synchronized RGB-D frames while intentionally ignoring poses."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.intrinsics = self._read_intrinsics()
        self.metadata = self._read_metadata()
        try:
            self.depth_scale_m = float(self.metadata["depth_scale_mm"]) / 1000.0
        except (KeyError, ValueError) as exc:
            raise ValueError("capture_info.txt must define positive depth_scale_mm") from exc
        if self.depth_scale_m <= 0:
            raise ValueError("depth_scale_mm must be positive")
        self._records = self._read_associations()
        if not self._records:
            raise ValueError("associated.txt contains no RGB-D frames")

    def _read_intrinsics(self) -> tuple[float, float, float, float]:
        values = (self.root / "calibration.txt").read_text(encoding="utf-8").split()
        if len(values) != 4:
            raise ValueError("calibration.txt must contain fx fy cx cy")
        intrinsics = tuple(float(value) for value in values)
        if intrinsics[0] <= 0 or intrinsics[1] <= 0 or not np.isfinite(intrinsics).all():
            raise ValueError("camera intrinsics must be finite with positive focal lengths")
        return intrinsics  # type: ignore[return-value]

    def _read_metadata(self) -> dict[str, str]:
        metadata: dict[str, str] = {}
        for line in (self.root / "capture_info.txt").read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                metadata[key.strip()] = value.strip()
        return metadata

    def _read_associations(self) -> list[tuple[float, Path, Path]]:
        records: list[tuple[float, Path, Path]] = []
        for line_number, line in enumerate(
            (self.root / "associated.txt").read_text(encoding="utf-8").splitlines(), 1
        ):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"invalid association at line {line_number}")
            rgb_time, rgb_path, depth_time, depth_path = fields
            if abs(float(rgb_time) - float(depth_time)) > 1e-6:
                raise ValueError(f"RGB and depth timestamps differ at line {line_number}")
            records.append((float(rgb_time), self.root / rgb_path, self.root / depth_path))
        return records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> RgbdFrame:
        timestamp, rgb_path, depth_path = self._records[index]
        color_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if color_bgr is None or depth is None:
            raise RuntimeError(f"failed to read RGB-D frame at timestamp {timestamp:.6f}")
        if depth.ndim != 2 or depth.dtype != np.uint16:
            raise RuntimeError("depth images must be 16-bit single-channel PNGs")
        return RgbdFrame(timestamp, color_bgr[..., ::-1].copy(), depth)

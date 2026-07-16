#!/usr/bin/env python3
"""Compatibility wrapper for the moved ZED Open3D dataset capture script."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    target = (
        Path(__file__).resolve().parent
        / "feature_extraction"
        / "capture_zed_open3d_dataset.py"
    )
    print(
        "Note: scripts/capture_zed_open3d_dataset.py moved to "
        "scripts/feature_extraction/capture_zed_open3d_dataset.py",
        file=sys.stderr,
    )
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()

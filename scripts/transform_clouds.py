#!/usr/bin/env python3
"""Transform, register, merge, and mesh angle-indexed point clouds.

This is deliberately a small public entry point.  The tested numerical
implementation lives in ``transform_clouds_pipeline.py``.

Workflow:
    1. Read root-level ``angle_*.ply`` files.
    2. Place them using the known camera orbit.
    3. Refine alignment with guarded ICP and a pose graph.
    4. Crop, clean, and merge the aligned clouds.
    5. Optionally create a Poisson mesh.
"""

from transform_clouds_pipeline import main


if __name__ == "__main__":
    main()

"""Selection and application of the Orbbec recommended depth filter chain."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# DisparityTransform must stay first: the spatial/temporal filters are designed
# to operate in the disparity domain before conversion back to depth.
DEFAULT_ENABLED_FILTERS: tuple[str, ...] = (
    "DisparityTransform",
    "SpatialAdvancedFilter",
    "TemporalFilter",
    "NoiseRemovalFilter",
    "EdgeNoiseRemovalFilter",
)


def select_depth_filters(recommended, enabled_names) -> list:
    """Enable exactly *enabled_names* within the device's recommended chain."""
    wanted = set(enabled_names)
    filters = list(recommended)
    for depth_filter in filters:
        try:
            depth_filter.enable(depth_filter.get_name() in wanted)
        except Exception as exc:  # pragma: no cover - device-specific
            logger.warning("Could not toggle filter %s: %s", depth_filter, exc)
    return filters


def apply_depth_filters(depth_frame, filters):
    """Run the enabled filters in order, carrying the frame through the chain."""
    processed = depth_frame
    for depth_filter in filters:
        try:
            if not depth_filter.is_enabled():
                continue
            result = depth_filter.process(processed)
        except Exception as exc:  # pragma: no cover - device-specific
            logger.warning("Depth filter %s failed: %s", depth_filter, exc)
            continue
        if result is not None:
            processed = result
    return processed


def gate_by_confidence(depth_m, confidence, *, min_confidence: int):
    """Zero out depth pixels the sensor reports below *min_confidence*."""
    import numpy as np

    depth = np.array(depth_m, dtype=np.float32, copy=True)
    if confidence is None:
        return depth
    conf = np.asarray(confidence)
    if conf.shape != depth.shape:
        raise ValueError("confidence and depth_m must have the same shape")
    depth[conf < min_confidence] = 0.0
    return depth

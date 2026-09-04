"""Selection and application of the Orbbec recommended depth filter chain."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# The spatial and temporal filters operate in the disparity domain, so
# DisparityTransform converts back to depth *after* them, which is where the
# Gemini 305 places it in its own recommended order. The noise-removal filters
# work on depth and therefore follow it.
DEFAULT_ENABLED_FILTERS: tuple[str, ...] = (
    "DisparityTransform",
    "SpatialAdvancedFilter",
    "TemporalFilter",
    "NoiseRemovalFilter",
    "EdgeNoiseRemovalFilter",
)

# Requested filters the Gemini 305 leaves out of get_recommended_filters().
# Naming them there enabled nothing at all, so they are constructed directly.
CONSTRUCTIBLE_FILTERS: tuple[str, ...] = (
    "NoiseRemovalFilter",
    "EdgeNoiseRemovalFilter",
)


def _construct_filter(name: str):
    """Build a filter the device did not recommend, or return None."""
    try:
        import pyorbbecsdk
    except ImportError:  # pragma: no cover - device-specific
        return None
    factory = getattr(pyorbbecsdk, name, None)
    if factory is None:
        return None
    try:
        depth_filter = factory()
        depth_filter.enable(True)
        return depth_filter
    except Exception as exc:  # pragma: no cover - device-specific
        logger.warning("Could not construct filter %s: %s", name, exc)
        return None


def select_depth_filters(recommended, enabled_names) -> list:
    """Enable *enabled_names*, constructing any the device does not recommend.

    ``get_recommended_filters()`` is only a device-supplied list, so naming a
    filter it omits used to enable nothing and report nothing. On the Gemini
    305 that silently dropped NoiseRemovalFilter and EdgeNoiseRemovalFilter,
    leaving three of the five requested filters actually running.
    """
    wanted = set(enabled_names)
    filters = list(recommended)
    available = set()
    for depth_filter in filters:
        try:
            name = depth_filter.get_name()
            depth_filter.enable(name in wanted)
            available.add(name)
        except Exception as exc:  # pragma: no cover - device-specific
            logger.warning("Could not toggle filter %s: %s", depth_filter, exc)

    for name in enabled_names:
        if name in available:
            continue
        if name not in CONSTRUCTIBLE_FILTERS:
            logger.warning(
                "Filter %s is neither recommended by the device nor "
                "constructible; it will not run.", name,
            )
            continue
        constructed = _construct_filter(name)
        if constructed is not None:
            filters.append(constructed)
            logger.info("Constructed depth filter %s outside the device chain.", name)
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

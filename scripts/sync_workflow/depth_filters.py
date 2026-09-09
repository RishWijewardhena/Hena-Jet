"""Selection and application of the Orbbec recommended depth filter chain."""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

# Explicit best-tested preset from filter_test_1280/noise_tuning. The test
# used disparity 256; this is not an established optimum at disparity 128.
# Width/height are SDK parameter reference dimensions, preserved from the
# tested configuration, not requested stream dimensions.
CAPTURE_FILTER_PARAMETERS = {
    "SpatialAdvancedFilter": {"alpha": 0.5, "magnitude": 1, "disp_diff": 160, "radius": 1},
    "TemporalFilter": {"weight": 0.4, "diff_scale": 0.1},
    "NoiseRemovalFilter": {"max_size": 80, "min_diff": 256, "width": 848, "height": 480},
    "EdgeNoiseRemovalFilter": {
        "margin_x_th": 6, "margin_y_th": 6, "limit_x_th": 70, "limit_y_th": 30,
        "enable_vertical_direction": 0, "width": 1280, "height": 800,
    },
}


def configure_capture_filters(filters, requested_names):
    """Set and read back the tested preset; do not silently accept SDK defaults."""
    available = {f.get_name(): f for f in filters if f.is_enabled()}
    verified = {}
    for name in requested_names:
        if name not in available:
            raise RuntimeError(f"Requested capture filter is unavailable: {name}")
        if name not in CAPTURE_FILTER_PARAMETERS:
            continue
        depth_filter = available[name]
        verified[name] = {}
        for key, value in CAPTURE_FILTER_PARAMETERS[name].items():
            depth_filter.set_config_value(key, value)
            active = float(depth_filter.get_config_value(key))
            if not math.isclose(active, value, rel_tol=1e-6, abs_tol=1e-6):
                raise RuntimeError(f"Filter setting mismatch: {name}.{key}: {active} != {value}")
            verified[name][key] = active
        logger.info("Verified capture filter %s: %s", name, verified[name])
    return verified

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

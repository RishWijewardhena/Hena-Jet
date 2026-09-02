import logging
from typing import Tuple, Optional

from depth_filters import DEFAULT_ENABLED_FILTERS, apply_depth_filters, select_depth_filters

logger = logging.getLogger(__name__)

DISPARITY_MODE_BY_PIXELS = {
    128: 1,
    256: 2,
}
DISPARITY_PIXELS_BY_MODE = {
    mode: pixels for pixels, mode in DISPARITY_MODE_BY_PIXELS.items()
}


def aligned_pointcloud_intrinsics(camera_param) -> dict[str, float | int | str]:
    """Return intrinsics for software depth-to-color aligned depth frames.

    D2C alignment reprojects depth pixels into the RGB camera image and
    coordinate system. Back-projecting that aligned depth with the original
    depth-camera intrinsics changes metric X/Y scale and is not geometrically
    valid.
    """
    intrinsic = camera_param.rgb_intrinsic
    return {
        "width": int(intrinsic.width),
        "height": int(intrinsic.height),
        "fx": float(intrinsic.fx),
        "fy": float(intrinsic.fy),
        "cx": float(intrinsic.cx),
        "cy": float(intrinsic.cy),
        "coordinate_frame": "color",
    }


def depth_pointcloud_intrinsics(camera_param) -> dict[str, float | int | str]:
    """Return intrinsics for an unaligned depth image fallback."""
    intrinsic = camera_param.depth_intrinsic
    return {
        "width": int(intrinsic.width),
        "height": int(intrinsic.height),
        "fx": float(intrinsic.fx),
        "fy": float(intrinsic.fy),
        "cx": float(intrinsic.cx),
        "cy": float(intrinsic.cy),
        "coordinate_frame": "depth",
    }

try:
    from pyorbbecsdk import (
        AlignFilter,
        Config,
        Context,
        OBAlignMode,
        OBFormat,
        OBFrameAggregateOutputMode,
        OBSensorType,
        OBStreamType,
        OBPropertyID,
        Pipeline,
    )
except ImportError:
    logger.error("pyorbbecsdk is required but not installed.")
    # Provide dummy classes for type hinting / offline development
    Pipeline = type('Pipeline', (), {})
    AlignFilter = type('AlignFilter', (), {})
    OBPropertyID = type(
        'OBPropertyID',
        (),
        {'OB_PROP_DISP_SEARCH_RANGE_MODE_INT': object()},
    )


def configure_disparity_search_range(device, requested_disparity: str | int) -> int:
    """Configure and verify the SDK disparity search-range mode."""
    try:
        requested_pixels = int(requested_disparity)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Unsupported disparity search range: {requested_disparity!r}"
        ) from exc

    if requested_pixels not in DISPARITY_MODE_BY_PIXELS:
        supported = ", ".join(str(value) for value in DISPARITY_MODE_BY_PIXELS)
        raise ValueError(
            f"Unsupported disparity search range: {requested_pixels}. "
            f"Expected one of: {supported}."
        )

    property_id = getattr(
        OBPropertyID,
        "OB_PROP_DISP_SEARCH_RANGE_MODE_INT",
        None,
    )
    if property_id is None:
        raise RuntimeError(
            "The installed Orbbec SDK does not expose "
            "OB_PROP_DISP_SEARCH_RANGE_MODE_INT."
        )

    requested_mode = DISPARITY_MODE_BY_PIXELS[requested_pixels]
    try:
        current_mode = int(device.get_int_property(property_id))
        if current_mode != requested_mode:
            device.set_int_property(property_id, requested_mode)
        active_mode = int(device.get_int_property(property_id))
    except Exception as exc:
        raise RuntimeError(
            f"Could not configure the Orbbec disparity search range to "
            f"{requested_pixels} pixels."
        ) from exc

    if active_mode != requested_mode:
        active_description = DISPARITY_PIXELS_BY_MODE.get(
            active_mode,
            f"unknown SDK mode {active_mode}",
        )
        raise RuntimeError(
            f"Disparity search-range verification failed: requested "
            f"{requested_pixels} pixels but the camera reported "
            f"{active_description}."
        )

    logger.info(
        "Verified disparity search range: %d pixels (SDK mode %d)",
        requested_pixels,
        active_mode,
    )
    return requested_pixels


class CameraController:
    """Wrapper for Orbbec Gemini 305 camera operations."""

    def __init__(self, width: int = 848, height: int = 530, fps: int = 30,
                 disparity: str = "256", depth_filters=DEFAULT_ENABLED_FILTERS):
        self.width = width
        self.height = height
        self.fps = fps
        self.disparity = disparity

        self.pipeline = None
        self.align_filter = None
        self.depth_filter_names = tuple(depth_filters or ())
        self.depth_filters = []
        self.camera_param = None
        self.intrinsics = None
        self.dist_coeffs = None
        self.active_disparity = None
        self.pointcloud_coordinate_frame = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        logger.info(f"Starting Orbbec camera: {self.width}x{self.height} @ {self.fps}fps")
        ctx = Context()
        devices = ctx.query_devices()
        if devices.get_count() == 0:
            raise RuntimeError("No Orbbec camera detected.")

        self.pipeline = Pipeline()
        config = Config()

        device = self.pipeline.get_device()
        self.active_disparity = configure_disparity_search_range(
            device,
            self.disparity,
        )

        # Setup Color Stream
        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = None
        for fmt in (OBFormat.RGB, OBFormat.MJPG, OBFormat.BGR):
            try:
                color_profile = color_profiles.get_video_stream_profile(
                    self.width, self.height, fmt, self.fps
                )
                if color_profile is not None:
                    break
            except Exception:
                continue

        if color_profile is None:
            color_profile = color_profiles.get_default_video_stream_profile()
            logger.warning("Could not find requested color profile, using default.")

        # Setup Depth Stream
        depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = None
        try:
            depth_profile = depth_profiles.get_video_stream_profile(
                self.width, self.height, OBFormat.Y16, self.fps
            )
        except Exception:
            depth_profile = depth_profiles.get_default_video_stream_profile()
            logger.warning("Could not find requested depth profile, using default.")

        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)
        
        # Match the SDK point-cloud example: enable raw streams here, then run
        # an explicit software depth-to-color AlignFilter on every FrameSet.
        align_to_color = True
        try:
            self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        except Exception as e:
            logger.warning(f"Could not create D2C AlignFilter: {e}. Will not align.")
            align_to_color = False

        try:
            self.pipeline.enable_frame_sync()
        except Exception as e:
            logger.warning(f"Hardware frame sync warning: {e}")

        self.pipeline.start(config)

        try:
            depth_sensor = self.pipeline.get_device().get_sensor(OBSensorType.DEPTH_SENSOR)
            self.depth_filters = select_depth_filters(
                depth_sensor.get_recommended_filters(), self.depth_filter_names
            )
            logger.info(
                "Depth filters enabled: %s",
                [f.get_name() for f in self.depth_filters if f.is_enabled()],
            )
        except Exception as e:
            logger.warning("Depth post-processing unavailable: %s", e)
            self.depth_filters = []

        # Wait a few frames for auto-exposure to settle
        for _ in range(10):
            self.pipeline.wait_for_frames(1000)

        self.camera_param = self.pipeline.get_camera_param()
        # SW_MODE is depth-to-color alignment. The aligned depth image must be
        # back-projected with RGB intrinsics and lives in the RGB camera frame.
        self.intrinsics = (
            aligned_pointcloud_intrinsics(self.camera_param)
            if align_to_color
            else depth_pointcloud_intrinsics(self.camera_param)
        )
        self.pointcloud_coordinate_frame = self.intrinsics["coordinate_frame"]
        logger.info(f"Camera intrinsics initialized: {self.intrinsics}")

    def capture_aligned_rgbd(self, timeout_ms: int = 1000) -> Tuple[Optional[object], Optional[object]]:
        """
        Wait for a frame and return aligned (color_frame, depth_frame).
        Returns (None, None) if a frame pair couldn't be captured.
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline is not started.")

        # Flush stale frames from the buffer
        flushed_count = 0
        while True:
            # Use a tiny timeout to quickly pull frames until the queue is empty
            old_frames = self.pipeline.wait_for_frames(10)
            if old_frames is None:
                break
            flushed_count += 1
            
        logger.debug(f"Flushed {flushed_count} stale frames from camera queue.")

        frames = self.pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            logger.warning("Timeout waiting for fresh frames.")
            return None, None

        if self.align_filter is not None:
            aligned = self.align_filter.process(frames)
            if aligned is None:
                logger.warning("Depth-to-color alignment failed.")
                return None, None
            frames = aligned.as_frame_set() if hasattr(aligned, "as_frame_set") else aligned

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is None or depth_frame is None:
            logger.warning("Incomplete frame pair.")
            return None, None

        if self.depth_filters:
            filtered = apply_depth_filters(depth_frame, self.depth_filters)
            if filtered is not None:
                depth_frame = filtered

        return color_frame, depth_frame

    def stop(self):
        if self.pipeline:
            self.pipeline.stop()
            logger.info("Camera pipeline stopped.")

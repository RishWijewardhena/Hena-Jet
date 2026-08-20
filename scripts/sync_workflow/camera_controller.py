import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

DISPARITY_MODE_BY_PIXELS = {
    128: 1,
    256: 2,
}
DISPARITY_PIXELS_BY_MODE = {
    mode: pixels for pixels, mode in DISPARITY_MODE_BY_PIXELS.items()
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

    def __init__(self, width: int = 848, height: int = 530, fps: int = 30, disparity: str = "256"):
        self.width = width
        self.height = height
        self.fps = fps
        self.disparity = disparity
        
        self.pipeline = None
        self.align_filter = None
        self.camera_param = None
        self.intrinsics = None
        self.dist_coeffs = None
        self.active_disparity = None

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
        
        # We prefer software D2C alignment as hardware alignment is not supported for all resolutions.
        try:
            config.set_align_mode(OBAlignMode.SW_MODE)
        except Exception as e:
            logger.warning(f"Could not set SW align mode: {e}. Will not align.")

        try:
            self.pipeline.enable_frame_sync()
        except Exception as e:
            logger.warning(f"Hardware frame sync warning: {e}")

        self.pipeline.start(config)

        # Wait a few frames for auto-exposure to settle
        for _ in range(10):
            self.pipeline.wait_for_frames(1000)

        self.camera_param = self.pipeline.get_camera_param()
        # Parse intrinsics for typical open3d / cv2 usage
        self.intrinsics = {
            "width": self.camera_param.depth_intrinsic.width,
            "height": self.camera_param.depth_intrinsic.height,
            "fx": self.camera_param.depth_intrinsic.fx,
            "fy": self.camera_param.depth_intrinsic.fy,
            "cx": self.camera_param.depth_intrinsic.cx,
            "cy": self.camera_param.depth_intrinsic.cy,
        }
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

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is None or depth_frame is None:
            logger.warning("Incomplete frame pair.")
            return None, None

        return color_frame, depth_frame

    def stop(self):
        if self.pipeline:
            self.pipeline.stop()
            logger.info("Camera pipeline stopped.")

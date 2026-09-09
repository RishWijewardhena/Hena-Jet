from __future__ import annotations

import sys
import unittest
from pathlib import Path

SYNC_WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "scripts" / "sync_workflow"
sys.path.insert(0, str(SYNC_WORKFLOW_DIR))

from depth_filters import (  # noqa: E402
    DEFAULT_ENABLED_FILTERS,
    apply_depth_filters,
    gate_by_confidence,
    select_depth_filters,
    configure_capture_filters,
    CAPTURE_FILTER_PARAMETERS,
)


class FakeFilter:
    def __init__(self, name, result="processed", enabled=False):
        self._name = name
        self._enabled = enabled
        self._result = result
        self.seen = []

    def get_name(self):
        return self._name

    def enable(self, value):
        self._enabled = bool(value)

    def is_enabled(self):
        return self._enabled

    def process(self, frame):
        self.seen.append(frame)
        return self._result


class SelectDepthFiltersTests(unittest.TestCase):
    def test_preset_overrides_different_sdk_defaults_and_verifies(self):
        from unittest.mock import Mock
        filters = []
        for name, parameters in CAPTURE_FILTER_PARAMETERS.items():
            values = {key: -1 for key in parameters}
            f = Mock()
            f.get_name.return_value = name
            f.is_enabled.return_value = True
            f.set_config_value.side_effect = values.__setitem__
            f.get_config_value.side_effect = values.__getitem__
            filters.append(f)
        actual = configure_capture_filters(filters, CAPTURE_FILTER_PARAMETERS)
        self.assertEqual(actual, CAPTURE_FILTER_PARAMETERS)

    def test_preset_rejects_missing_filter(self):
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            configure_capture_filters([], ["TemporalFilter"])

    def test_preset_rejects_ignored_setting(self):
        from unittest.mock import Mock
        f = Mock()
        f.get_name.return_value = "TemporalFilter"
        f.is_enabled.return_value = True
        f.get_config_value.return_value = 0.9
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            configure_capture_filters([f], ["TemporalFilter"])

    def test_enables_only_the_requested_filters(self):
        recommended = [FakeFilter("TemporalFilter"), FakeFilter("DecimationFilter")]

        select_depth_filters(recommended, ("TemporalFilter",))

        self.assertTrue(recommended[0].is_enabled())
        self.assertFalse(recommended[1].is_enabled())

    def test_disables_a_previously_enabled_filter(self):
        recommended = [FakeFilter("DecimationFilter", enabled=True)]

        select_depth_filters(recommended, ("TemporalFilter",))

        self.assertFalse(recommended[0].is_enabled())

    def test_returns_the_filters_in_device_order(self):
        recommended = [FakeFilter("A"), FakeFilter("B")]

        self.assertEqual(select_depth_filters(recommended, ()), recommended)

    def test_default_chain_keeps_disparity_transform_first(self):
        self.assertEqual(DEFAULT_ENABLED_FILTERS[0], "DisparityTransform")


class ApplyDepthFiltersTests(unittest.TestCase):
    def test_chains_enabled_filters_in_order(self):
        first = FakeFilter("A", result="one", enabled=True)
        second = FakeFilter("B", result="two", enabled=True)

        result = apply_depth_filters("raw", [first, second])

        self.assertEqual(result, "two")
        self.assertEqual(first.seen, ["raw"])
        self.assertEqual(second.seen, ["one"])

    def test_skips_disabled_filters(self):
        enabled = FakeFilter("A", result="one", enabled=True)
        disabled = FakeFilter("B", result="two", enabled=False)

        result = apply_depth_filters("raw", [enabled, disabled])

        self.assertEqual(result, "one")
        self.assertEqual(disabled.seen, [])

    def test_carries_the_previous_frame_when_a_filter_returns_none(self):
        good = FakeFilter("A", result="one", enabled=True)
        broken = FakeFilter("B", result=None, enabled=True)

        self.assertEqual(apply_depth_filters("raw", [good, broken]), "one")

    def test_returns_the_input_when_no_filters_are_enabled(self):
        self.assertEqual(apply_depth_filters("raw", []), "raw")


class GateByConfidenceTests(unittest.TestCase):
    def test_zeroes_low_confidence_pixels(self):
        import numpy as np

        depth = np.array([[0.25, 0.26], [0.27, 0.28]], dtype=np.float32)
        confidence = np.array([[10, 200], [255, 30]], dtype=np.uint8)

        gated = gate_by_confidence(depth, confidence, min_confidence=100)

        np.testing.assert_allclose(gated, [[0.0, 0.26], [0.27, 0.0]])

    def test_passes_through_when_confidence_is_missing(self):
        import numpy as np

        depth = np.array([[0.25]], dtype=np.float32)

        np.testing.assert_allclose(
            gate_by_confidence(depth, None, min_confidence=100), depth
        )

    def test_rejects_mismatched_shapes(self):
        import numpy as np

        with self.assertRaises(ValueError):
            gate_by_confidence(
                np.zeros((2, 2)), np.zeros((3, 3), dtype=np.uint8), min_confidence=1
            )


if __name__ == "__main__":
    unittest.main()

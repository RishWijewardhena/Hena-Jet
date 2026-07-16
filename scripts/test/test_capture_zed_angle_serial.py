import unittest

from scripts.capture_zed_angle import (
    advance_capture_angle,
    parse_segment_acknowledgement,
    serial_start_command,
)


class CaptureZedAngleSerialTests(unittest.TestCase):
    def test_parses_segment_acknowledgement_from_device(self):
        self.assertEqual(parse_segment_acknowledgement("5 degree ok"), 5.0)
        self.assertEqual(parse_segment_acknowledgement("  2.5 DEGREE OK  "), 2.5)

    def test_ignores_non_segment_messages(self):
        for line in ("ready", "started,5,10000", "completed", "stopped", "alert"):
            with self.subTest(line=line):
                self.assertIsNone(parse_segment_acknowledgement(line))

    def test_accumulates_relative_segment_angles(self):
        self.assertEqual(advance_capture_angle(0.0, 5.0), 5.0)
        self.assertEqual(advance_capture_angle(355.0, 5.0), 360.0)

    def test_rejects_invalid_or_overrunning_segment_angles(self):
        with self.assertRaises(ValueError):
            advance_capture_angle(10.0, 0.0)
        with self.assertRaises(ValueError):
            advance_capture_angle(358.0, 5.0)

    def test_formats_firmware_start_command(self):
        self.assertEqual(serial_start_command(5.0, 10000), b"start,5,10000\n")
        self.assertEqual(serial_start_command(2.5, 52100), b"start,2.5,52100\n")


if __name__ == "__main__":
    unittest.main()

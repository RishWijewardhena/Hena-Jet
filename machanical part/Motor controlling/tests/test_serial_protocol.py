import unittest

from serial_protocol import format_start_command, parse_device_message


class SerialProtocolTests(unittest.TestCase):
    def test_formats_integer_angle_without_decimal_suffix(self):
        self.assertEqual(format_start_command(5.0, 52100), b"start,5,52100\n")

    def test_formats_fractional_angle(self):
        self.assertEqual(format_start_command(5.25, 52100), b"start,5.25,52100\n")

    def test_rejects_angle_outside_firmware_range(self):
        with self.assertRaises(ValueError):
            format_start_command(0.001, 52100)
        with self.assertRaises(ValueError):
            format_start_command(361, 52100)

    def test_rejects_invalid_pulses_per_revolution(self):
        with self.assertRaises(ValueError):
            format_start_command(5, 0)
        with self.assertRaises(ValueError):
            format_start_command(5, 100001)

    def test_parses_started_message_with_runtime_ppr(self):
        message = parse_device_message("started,5,52100")
        self.assertEqual(message.kind, "started")
        self.assertEqual(message.value, 5.0)
        self.assertEqual(message.pulses_per_revolution, 52100)

    def test_parses_segment_acknowledgement(self):
        message = parse_device_message("5 degree ok")
        self.assertEqual(message.kind, "segment_ok")
        self.assertEqual(message.value, 5.0)

    def test_parses_alarm_and_error_messages(self):
        self.assertEqual(parse_device_message("alert").kind, "alert")
        error = parse_device_message("error,alarm_active")
        self.assertEqual(error.kind, "error")
        self.assertEqual(error.value, "alarm_active")


if __name__ == "__main__":
    unittest.main()

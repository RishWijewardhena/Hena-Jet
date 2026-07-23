import unittest

from serial_protocol import format_start_command, parse_device_message


class SerialProtocolTests(unittest.TestCase):
    def test_formats_integer_angle_without_decimal_suffix(self):
        self.assertEqual(format_start_command(5.0, 52100), b"start,5,52100\n")

    def test_formats_continuous_start_with_runtime_rpm(self):
        self.assertEqual(
            format_start_command(5.0, 52100, continuous=True, rpm=0.5),
            b"start_continuous,5,52100,0.5\n",
        )

    def test_formats_optional_continuous_direction_without_changing_legacy_command(self):
        self.assertEqual(
            format_start_command(
                5.0,
                52100,
                continuous=True,
                rpm=0.5,
                direction="reverse",
            ),
            b"start_continuous,5,52100,0.5,reverse\n",
        )
        self.assertEqual(
            format_start_command(5.0, 52100, continuous=True, rpm=0.5),
            b"start_continuous,5,52100,0.5\n",
        )

    def test_rejects_invalid_or_noncontinuous_direction(self):
        with self.assertRaises(ValueError):
            format_start_command(
                5.0, 52100, continuous=True, rpm=0.5, direction="sideways"
            )
        with self.assertRaises(ValueError):
            format_start_command(5.0, 52100, direction="reverse")

    def test_continuous_start_requires_valid_rpm(self):
        with self.assertRaises(ValueError):
            format_start_command(5.0, 52100, continuous=True)
        with self.assertRaises(ValueError):
            format_start_command(5.0, 52100, continuous=True, rpm=0.0)

    def test_formats_fractional_angle(self):
        self.assertEqual(format_start_command(5.25, 52100), b"start,5.25,52100\n")

    def test_formats_synchronized_start_without_changing_legacy_default(self):
        self.assertEqual(
            format_start_command(5.0, 52100, synchronized=True),
            b"start_sync,5,52100\n",
        )
        self.assertEqual(format_start_command(5.0, 52100), b"start,5,52100\n")

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

    def test_parses_continuous_start_and_cumulative_angle_event(self):
        started = parse_device_message("started_continuous,5,52100,0.5")
        self.assertEqual(started.kind, "started_continuous")
        self.assertEqual(started.value, 5.0)
        self.assertEqual(started.pulses_per_revolution, 52100)
        self.assertEqual(started.rpm, 0.5)

        reversed_start = parse_device_message(
            "started_continuous,5,52100,0.5,reverse"
        )
        self.assertEqual(reversed_start.kind, "started_continuous")
        self.assertEqual(reversed_start.direction, "reverse")

        event = parse_device_message("angle_ok,125,18090")
        self.assertEqual(event.kind, "angle_ok")
        self.assertEqual(event.value, 125.0)
        self.assertEqual(event.pulse_count, 18090)

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

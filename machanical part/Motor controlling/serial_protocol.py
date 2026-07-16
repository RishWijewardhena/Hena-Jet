"""Shared formatting and parsing for the stepper controller serial protocol."""

from dataclasses import dataclass
import math
import re


MAXIMUM_ANGLE_DEGREES = 360.0
MINIMUM_PULSES_PER_REVOLUTION = 1
MAXIMUM_PULSES_PER_REVOLUTION = 100_000


@dataclass(frozen=True)
class DeviceMessage:
    kind: str
    value: float | str | None = None
    pulses_per_revolution: int | None = None


def format_degrees(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def format_start_command(angle_degrees: float, pulses_per_revolution: int) -> bytes:
    angle = float(angle_degrees)
    ppr_value = float(pulses_per_revolution)
    if (
        not math.isfinite(ppr_value)
        or not ppr_value.is_integer()
        or ppr_value < MINIMUM_PULSES_PER_REVOLUTION
        or ppr_value > MAXIMUM_PULSES_PER_REVOLUTION
    ):
        raise ValueError(
            f"Pulses/revolution must be an integer between "
            f"{MINIMUM_PULSES_PER_REVOLUTION} and "
            f"{MAXIMUM_PULSES_PER_REVOLUTION}"
        )
    ppr = int(ppr_value)
    minimum_angle = 360.0 / ppr
    if (
        not math.isfinite(angle)
        or angle + 1e-12 < minimum_angle
        or angle > MAXIMUM_ANGLE_DEGREES
    ):
        raise ValueError(
            f"Angle must be between {minimum_angle:.6g} and "
            f"{MAXIMUM_ANGLE_DEGREES} degrees"
        )
    return f"start,{format_degrees(angle)},{ppr}\n".encode("ascii")


def parse_device_message(line: str) -> DeviceMessage:
    normalized = line.strip().lower()

    if normalized in {"ready", "completed", "stopped", "alert"}:
        return DeviceMessage(normalized)

    if normalized.startswith("started,"):
        parts = normalized.split(",")
        try:
            if len(parts) != 3:
                raise ValueError
            return DeviceMessage(
                "started",
                float(parts[1]),
                int(parts[2]),
            )
        except ValueError:
            return DeviceMessage("unknown")

    if normalized.startswith("error,"):
        return DeviceMessage("error", normalized.split(",", 1)[1])

    match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+degree\s+ok", normalized
    )
    if match:
        return DeviceMessage("segment_ok", float(match.group(1)))

    return DeviceMessage("unknown")

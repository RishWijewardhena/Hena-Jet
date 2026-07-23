"""Shared formatting and parsing for the stepper controller serial protocol."""

from dataclasses import dataclass
import math
import re


MAXIMUM_ANGLE_DEGREES = 360.0
MINIMUM_PULSES_PER_REVOLUTION = 1
MAXIMUM_PULSES_PER_REVOLUTION = 100_000
MINIMUM_CONTINUOUS_RPM = 0.05
MAXIMUM_CONTINUOUS_RPM = 2.0


@dataclass(frozen=True)
class DeviceMessage:
    kind: str
    value: float | str | None = None
    pulses_per_revolution: int | None = None
    rpm: float | None = None
    pulse_count: int | None = None
    direction: str | None = None


def format_degrees(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def format_start_command(
    angle_degrees: float,
    pulses_per_revolution: int,
    *,
    synchronized: bool = False,
    continuous: bool = False,
    rpm: float | None = None,
    direction: str | None = None,
) -> bytes:
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
    if synchronized and continuous:
        raise ValueError("Synchronized and continuous modes are mutually exclusive")
    if continuous:
        if rpm is None or not math.isfinite(float(rpm)):
            raise ValueError("Continuous mode requires a finite RPM")
        rpm = float(rpm)
        if rpm < MINIMUM_CONTINUOUS_RPM or rpm > MAXIMUM_CONTINUOUS_RPM:
            raise ValueError(
                f"RPM must be between {MINIMUM_CONTINUOUS_RPM:g} and "
                f"{MAXIMUM_CONTINUOUS_RPM:g}"
            )
        if direction not in {None, "forward", "reverse"}:
            raise ValueError("Direction must be 'forward' or 'reverse'")
    elif rpm is not None:
        raise ValueError("RPM is only valid in continuous mode")
    elif direction is not None:
        raise ValueError("Direction is only valid in continuous mode")
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
    if continuous:
        command = (
            f"start_continuous,{format_degrees(angle)},{ppr},"
            f"{format_degrees(rpm)}"
        )
        if direction is not None:
            command += f",{direction}"
        return f"{command}\n".encode("ascii")
    prefix = "start_sync" if synchronized else "start"
    return f"{prefix},{format_degrees(angle)},{ppr}\n".encode("ascii")


def parse_device_message(line: str) -> DeviceMessage:
    normalized = line.strip().lower()

    if normalized in {"ready", "completed", "stopped", "alert"}:
        return DeviceMessage(normalized)

    if normalized.startswith("started_continuous,"):
        parts = normalized.split(",")
        try:
            if len(parts) not in {4, 5}:
                raise ValueError
            direction = None if len(parts) == 4 else parts[4]
            if direction not in {None, "forward", "reverse"}:
                raise ValueError
            return DeviceMessage(
                "started_continuous",
                float(parts[1]),
                int(parts[2]),
                float(parts[3]),
                direction=direction,
            )
        except ValueError:
            return DeviceMessage("unknown")

    if normalized.startswith("angle_ok,"):
        parts = normalized.split(",")
        try:
            if len(parts) != 3:
                raise ValueError
            return DeviceMessage(
                "angle_ok",
                float(parts[1]),
                pulse_count=int(parts[2]),
            )
        except ValueError:
            return DeviceMessage("unknown")

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

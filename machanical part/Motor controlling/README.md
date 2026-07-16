# HBT4248C serial motor controller

This project drives an HBT4248C integrated stepper motor from a Seeed XIAO
ESP32-S3 and provides a Tkinter serial controller for a PC.

## Serial protocol

The firmware uses 115200 baud and newline-terminated ASCII commands.

| Direction | Message | Meaning |
| --- | --- | --- |
| PC → ESP32 | `start,5,52100` | Run one 360° sequence in nominal 5° segments at 52,100 PPR |
| PC → ESP32 | `stop` | Stop the active sequence immediately |
| ESP32 → PC | `started,5,52100` | Start command and runtime PPR accepted |
| ESP32 → PC | `5 degree ok` | One requested segment completed |
| ESP32 → PC | `completed` | The 360° sequence completed |
| ESP32 → PC | `alert` | A driver alarm edge was detected |
| ESP32 → PC | `stopped` | The sequence was cancelled |
| ESP32 → PC | `error,<reason>` | The command could not be performed |

There is a one-second delay between segments. Pulses per revolution is supplied
with every Start command instead of being compiled into the firmware. At 52,100
PPR, one pulse is approximately 0.00691°. Cumulative rounding makes the final
360° position exactly the requested PPR even when an individual segment is not
an exact number of pulses.

Each segment uses a symmetric quintic S-curve: it starts at 0.5 RPM, smoothly
accelerates toward the configured 2 RPM maximum, and smoothly decelerates
before the final pulse. The 8 RPM/s setting determines the nominal ramp length.
A Serial `stop` command uses an S-curve controlled stop. A driver alarm remains
an emergency condition and stops STEP pulses immediately.

## Alarm wiring warning

The HBT4248C `AL+` and `AL-` terminals are an isolated output pair. They are
not the same as the motor supply `V+` and `V-`, and `AL-` must not be assumed to
be motor-power ground.

The firmware expects **D4 to be LOW normally and safely driven HIGH at 3.3 V
during a fault**. Never apply 5 V directly to D4. Use the alarm-output circuit
specified by the exact HBT4248C manual, an appropriate optocoupler interface,
or a level-conditioning circuit verified with a meter before connecting D4.
The exact `AL+`/`AL-` circuit could not be verified from a manufacturer-hosted
manual, so have the interface reviewed before powering it.

## Build and run

Build and upload the firmware with PlatformIO using the
`seeed_xiao_esp32s3` environment.

Install the GUI dependency and start the controller:

```bash
python3 -m pip install -r requirements.txt
python3 motor_controller_gui.py
```

Select the ESP32 serial port, connect, enter the angle increment and the
driver's configured pulses per revolution, then choose **Start 360° sequence**.
Use **Stop** to cancel it.

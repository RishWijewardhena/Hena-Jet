# HBT4248C serial motor controller

This project drives an HBT4248C integrated stepper motor from a Seeed XIAO
ESP32-S3 and provides a Tkinter serial controller for a PC.

## Serial protocol

The firmware uses 115200 baud and newline-terminated ASCII commands.

| Direction | Message | Meaning |
| --- | --- | --- |
| PC → ESP32 | `start,5,52100` | Run the legacy segmented 360° sequence |
| PC → ESP32 | `start_sync,5,52100` | Move 5° and wait for `next` after each segment |
| PC → ESP32 | `start_continuous,5,52100,0.5` | Run continuously at 0.5 RPM and report 5° crossings |
| PC → ESP32 | `stop` | Stop the active sequence immediately |
| ESP32 → PC | `started,5,52100` | Legacy start command accepted |
| ESP32 → PC | `started_continuous,5,52100,0.5` | Continuous command and runtime RPM accepted |
| ESP32 → PC | `angle_ok,5,724` | Cumulative angle crossed at the reported pulse count |
| ESP32 → PC | `5 degree ok` | One requested segment completed |
| ESP32 → PC | `completed` | The 360° sequence completed |
| ESP32 → PC | `alert` | A driver alarm edge was detected |
| ESP32 → PC | `stopped` | The sequence was cancelled |
| ESP32 → PC | `error,<reason>` | The command could not be performed |

Legacy automatic mode inserts a 200 ms delay between segments. Synchronized mode waits for `next`. Continuous mode uses one uninterrupted pulse train, reports cumulative angle crossings, passes through 360°, and stops after one increment of runout so the 360° frame is captured before stop vibration. Pulses per revolution is supplied at runtime; cumulative rounding keeps angle events aligned to integer pulse counts.

Legacy segments use a symmetric quintic S-curve. Continuous mode ramps once to the requested 0.05–2 RPM speed and decelerates only during the runout. A serial `stop` performs a controlled stop; a driver alarm remains an emergency condition and stops STEP pulses immediately.

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

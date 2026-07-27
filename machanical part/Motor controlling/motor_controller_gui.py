#!/usr/bin/env python3
"""Tkinter controller for the serial stepper firmware."""

from __future__ import annotations

from datetime import datetime
import queue
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

import serial
from serial.tools import list_ports

from serial_protocol import format_start_command, parse_device_message


BAUD_RATE = 115200


class MotorControllerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("HBT4248C Motor Controller")
        self.root.geometry("720x540")
        self.root.minsize(620, 460)

        self.connection: serial.Serial | None = None
        self.reader_thread: threading.Thread | None = None
        self.reader_stop = threading.Event()
        self.incoming: queue.Queue[tuple[str, str]] = queue.Queue()
        self.running = False

        self.port_var = tk.StringVar()
        self.angle_var = tk.StringVar(value="5")
        self.ppr_var = tk.StringVar(value="10000")
        self.connection_var = tk.StringVar(value="Disconnected")
        self.motion_var = tk.StringVar(value="Idle")
        self.alarm_var = tk.StringVar(value="Alarm monitor disconnected")

        self._build_ui()
        self.refresh_ports()
        self._update_controls()
        self.root.after(50, self._drain_incoming)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.configure("Title.TLabel", font=("TkDefaultFont", 16, "bold"))
        style.configure("Status.TLabel", font=("TkDefaultFont", 10, "bold"))
        style.configure("AlarmNormal.TLabel", foreground="#2f5f3f")
        style.configure(
            "AlarmActive.TLabel",
            foreground="#a61b1b",
            font=("TkDefaultFont", 10, "bold"),
        )

        container = ttk.Frame(self.root, padding=16)
        container.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(4, weight=1)

        ttk.Label(container, text="HBT4248C Motor Controller", style="Title.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 12)
        )

        connection_frame = ttk.LabelFrame(container, text="Serial connection", padding=10)
        connection_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        connection_frame.columnconfigure(1, weight=1)

        ttk.Label(connection_frame, text="Port").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.port_box = ttk.Combobox(
            connection_frame, textvariable=self.port_var, state="readonly", width=28
        )
        self.port_box.grid(row=0, column=1, sticky="ew")
        self.refresh_button = ttk.Button(connection_frame, text="Refresh", command=self.refresh_ports)
        self.refresh_button.grid(row=0, column=2, padx=8)
        self.connect_button = ttk.Button(connection_frame, command=self.toggle_connection)
        self.connect_button.grid(row=0, column=3)

        movement_frame = ttk.LabelFrame(container, text="Movement sequence", padding=10)
        movement_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        movement_frame.columnconfigure(1, weight=1)

        ttk.Label(movement_frame, text="Increment (degrees)").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.angle_entry = ttk.Entry(movement_frame, textvariable=self.angle_var, width=14)
        self.angle_entry.grid(row=0, column=1, sticky="w")
        ttk.Label(movement_frame, text="Pulses per revolution").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=(8, 0)
        )
        self.ppr_entry = ttk.Entry(movement_frame, textvariable=self.ppr_var, width=14)
        self.ppr_entry.grid(row=1, column=1, sticky="w", pady=(8, 0))
        self.start_button = ttk.Button(movement_frame, text="Start 360° sequence", command=self.start)
        self.start_button.grid(row=0, column=2, rowspan=2, padx=8)
        self.stop_button = ttk.Button(movement_frame, text="Stop", command=self.stop)
        self.stop_button.grid(row=0, column=3, rowspan=2)

        status_frame = ttk.Frame(container)
        status_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        status_frame.columnconfigure(1, weight=1)
        ttk.Label(status_frame, textvariable=self.connection_var, style="Status.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 20)
        )
        ttk.Label(status_frame, textvariable=self.motion_var).grid(row=0, column=1, sticky="w")
        self.alarm_label = ttk.Label(
            status_frame,
            textvariable=self.alarm_var,
            style="AlarmNormal.TLabel",
        )
        self.alarm_label.grid(row=0, column=2, sticky="e")

        log_frame = ttk.LabelFrame(container, text="Device log", padding=8)
        log_frame.grid(row=4, column=0, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=14, state=tk.DISABLED)
        self.log.grid(row=0, column=0, sticky="nsew")

    def refresh_ports(self) -> None:
        ports = [port.device for port in list_ports.comports()]
        self.port_box["values"] = ports
        if self.port_var.get() not in ports:
            self.port_var.set(ports[0] if ports else "")

    def toggle_connection(self) -> None:
        if self.connection is None:
            self.connect()
        else:
            self.disconnect()

    def connect(self) -> None:
        port = self.port_var.get()
        if not port:
            messagebox.showerror("No serial port", "Select a serial port first.")
            return

        try:
            self.connection = serial.Serial(port, BAUD_RATE, timeout=0.2, write_timeout=1)
        except serial.SerialException as exc:
            messagebox.showerror("Connection failed", str(exc))
            return

        self.reader_stop.clear()
        self.reader_thread = threading.Thread(target=self._read_serial, daemon=True)
        self.reader_thread.start()
        self.connection_var.set(f"Connected: {port}")
        self.motion_var.set("Connected; ready to start")
        self._show_alarm_normal("No driver fault detected")
        self._append_log(f"Connected at {BAUD_RATE} baud")
        self._update_controls()

    def disconnect(self) -> None:
        self.reader_stop.set()
        connection = self.connection
        self.connection = None
        if connection is not None:
            try:
                connection.close()
            except serial.SerialException:
                pass
        if self.reader_thread is not None and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=0.5)
        self.reader_thread = None
        self.running = False
        self.connection_var.set("Disconnected")
        self.motion_var.set("Idle")
        self._show_alarm_normal("Alarm monitor disconnected")
        self._append_log("Disconnected")
        self._update_controls()

    def _read_serial(self) -> None:
        while not self.reader_stop.is_set():
            connection = self.connection
            if connection is None:
                return
            try:
                raw = connection.readline()
                if raw:
                    self.incoming.put(("line", raw.decode("utf-8", errors="replace").strip()))
            except (serial.SerialException, OSError) as exc:
                if not self.reader_stop.is_set():
                    self.incoming.put(("connection_error", str(exc)))
                return

    def _drain_incoming(self) -> None:
        try:
            while True:
                event, value = self.incoming.get_nowait()
                if event == "line":
                    self._handle_device_line(value)
                else:
                    self._append_log(f"Serial error: {value}")
                    messagebox.showerror("Serial connection lost", value)
                    self.disconnect()
        except queue.Empty:
            pass
        self.root.after(50, self._drain_incoming)

    def _handle_device_line(self, line: str) -> None:
        if not line:
            return
        self._append_log(f"RX  {line}")
        message = parse_device_message(line)

        if message.kind == "ready":
            self.running = False
            self.motion_var.set("Ready")
        elif message.kind == "started":
            self.running = True
            self.motion_var.set(
                f"Running: {message.value:g}° at "
                f"{message.pulses_per_revolution} PPR"
            )
            self._show_alarm_normal("No driver fault detected")
        elif message.kind == "segment_ok":
            self.motion_var.set(f"Completed {message.value:g}° segment; waiting 1 second")
        elif message.kind == "completed":
            self.running = False
            self.motion_var.set("360° sequence completed")
        elif message.kind == "stopped":
            self.running = False
            self.motion_var.set("Stopped")
        elif message.kind == "alert":
            self.running = False
            self._show_alarm_active("ALARM ACTIVE — driver fault")
            self.motion_var.set("Stopped by alarm")
        elif message.kind == "error":
            self.running = False
            self.motion_var.set(f"Device error: {message.value}")

        self._update_controls()

    def _show_alarm_normal(self, text: str) -> None:
        self.alarm_var.set(text)
        self.alarm_label.configure(style="AlarmNormal.TLabel")

    def _show_alarm_active(self, text: str) -> None:
        self.alarm_var.set(text)
        self.alarm_label.configure(style="AlarmActive.TLabel")

    def _send(self, payload: bytes) -> bool:
        connection = self.connection
        if connection is None:
            messagebox.showerror("Not connected", "Connect to the controller first.")
            return False
        try:
            connection.write(payload)
            connection.flush()
            self._append_log(f"TX  {payload.decode('ascii').strip()}")
            return True
        except (serial.SerialException, serial.SerialTimeoutException, OSError) as exc:
            messagebox.showerror("Send failed", str(exc))
            self.disconnect()
            return False

    def start(self) -> None:
        try:
            payload = format_start_command(
                float(self.angle_var.get()),
                int(self.ppr_var.get()),
            )
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Invalid movement settings", str(exc))
            self.angle_entry.focus_set()
            return
        if self._send(payload):
            self.running = True
            self.motion_var.set("Start requested")
            self._update_controls()

    def stop(self) -> None:
        if self._send(b"stop\n"):
            self.motion_var.set("Stop requested")

    def _append_log(self, text: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, f"[{timestamp}] {text}\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _update_controls(self) -> None:
        connected = self.connection is not None
        self.connect_button.configure(text="Disconnect" if connected else "Connect")
        self.port_box.configure(state=tk.DISABLED if connected else "readonly")
        self.refresh_button.configure(state=tk.DISABLED if connected else tk.NORMAL)
        self.angle_entry.configure(state=tk.DISABLED if self.running else tk.NORMAL)
        self.ppr_entry.configure(state=tk.DISABLED if self.running else tk.NORMAL)
        self.start_button.configure(state=tk.NORMAL if connected and not self.running else tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL if connected and self.running else tk.DISABLED)

    def close(self) -> None:
        if self.connection is not None:
            self.disconnect()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    MotorControllerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

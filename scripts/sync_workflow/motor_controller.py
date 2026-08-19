import time
import logging
import serial

logger = logging.getLogger(__name__)

class MotorController:
    """Wrapper for Marlin-based motor controller via serial."""
    
    def __init__(self, port="/dev/ttyACM0", baud=250000, timeout=2.0):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        logger.info(f"Connecting to motor on {self.port} at {self.baud} baud")
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
            time.sleep(2.0)  # Wait for board to boot
            self._drain_greeting()
        except serial.SerialException as e:
            logger.error(f"Failed to open serial port {self.port}: {e}")
            raise

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            logger.info("Disconnected from motor.")

    def _drain_greeting(self, seconds=1.5):
        end = time.time() + seconds
        while time.time() < end:
            raw = self.ser.readline()
            if raw:
                line = raw.decode(errors='replace').strip()
                if line:
                    logger.debug(f"Motor Greeting: {line}")

    def send_command(self, cmd, timeout_s=240, dry_run=False):
        """Sends a command and blocks until 'ok' is received."""
        logger.info(f"Motor CMD: {cmd}")
        if dry_run:
            return True

        if not self.ser or not self.ser.is_open:
            raise RuntimeError("Serial connection is not open.")

        self.ser.reset_input_buffer()
        self.ser.write((cmd + "\n").encode())
        self.ser.flush()

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue
            
            line = raw.decode(errors="replace").strip()
            if not line:
                continue
            
            low = line.lower()
            if low.startswith("echo:busy"):
                # Marlin is busy processing, wait.
                continue
                
            logger.debug(f"Motor RCV: {line}")
            
            if low.startswith("ok"):
                return True
            if low.startswith("error") or line.startswith("!!"):
                logger.error(f"Motor reported an error: {line}")
                return False

        raise TimeoutError(f"Command '{cmd}' timed out after {timeout_s}s")

    def home_all(self, dry_run=False):
        """
        Runs the homing sequence safely.
        Based on user instructions to avoid limit switch conflicts.
        """
        logger.info("Homing all axes...")
        sequence = [
            "G28 Z",
            "G28 A",
            "G28 X",
            "G1 X25 F1000",   # Move X to 0 to avoid limit switches
            # "G1 Y15 F1000",  # Move Y to 15 to avoid limit switches
            "G28 Y",
            "G1 Y40 F500",
            "G1 X200 Y0 F500"
        ]
        for cmd in sequence:
            success = self.send_command(cmd, dry_run=dry_run)
            if not success:
                raise RuntimeError(f"Homing failed at command: {cmd}")
        logger.info("Homing complete.")

    def move_y(self, angle, feedrate=1000, dry_run=False):
        """Move the Y axis to the specified angle and wait for completion."""
        cmd = f"G1 Y{angle:.2f} F{feedrate}"
        success = self.send_command(cmd, dry_run=dry_run)
        if not success:
            raise RuntimeError(f"Failed to move Y to {angle}")
            
        # Wait for the physical move to finish
        if not self.send_command("M400", dry_run=dry_run):
            raise RuntimeError("M400 failed while waiting for Y move to complete.")
            
    def move_x(self, pos, feedrate=1000, dry_run=False):
        """Move the X axis (translation) and wait for completion."""
        cmd = f"G1 X{pos:.2f} F{feedrate}"
        success = self.send_command(cmd, dry_run=dry_run)
        if not success:
            raise RuntimeError(f"Failed to move X to {pos}")
            
        # Wait for the physical move to finish
        if not self.send_command("M400", dry_run=dry_run):
            raise RuntimeError("M400 failed while waiting for X move to complete.")

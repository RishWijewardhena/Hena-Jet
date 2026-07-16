import serial
import time


# --- CONFIGURATION ---
SERIAL_PORT = '/dev/ttyACM0'  #ttyACM0
BAUD_RATE = 115200


# --- FUNCTION TO READ SERIAL DATA ---
def read_serial_data():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
        
        while True:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    try:
                        float(line)  # Validate if it's a float
                        print(f"Received angle: {line}°")
                    except ValueError:
                        print(f"Invalid data received: {line}")
            time.sleep(0.1)  # Small delay to prevent CPU overuse
    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial connection closed.")

if __name__ == "__main__":
    read_serial_data()
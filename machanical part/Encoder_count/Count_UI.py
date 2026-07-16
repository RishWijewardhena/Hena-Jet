import tkinter as tk
from tkinter import ttk
import serial
import threading

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/ttyACM0'  #ttyACM0
BAUD_RATE = 115200

class EncoderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hardware-Calculated Angle")
        self.root.geometry("400x250")
        
        self.current_angle = "0.00"
        self.ser = None  # We will store the serial connection here so the button can use it
        
        self.setup_ui()
        
        # Start background thread to read serial data
        self.serial_thread = threading.Thread(target=self.read_serial, daemon=True)
        self.serial_thread.start()
        
        self.update_ui_loop()
        
    def setup_ui(self):
        # Large angle display
        self.angle_label = ttk.Label(self.root, text="0.00°", font=("Helvetica", 54, "bold"))
        self.angle_label.pack(expand=True, pady=10)
        
        # --- NEW: Tare Button ---
        self.tare_button = ttk.Button(self.root, text="Tare (Hardware Zero)", command=self.send_tare_command)
        self.tare_button.pack(pady=15)
        
        # Connection Status Bar
        self.status_bar = ttk.Label(self.root, text="Connecting...", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def send_tare_command(self):
        # If the serial port is open, send the character 'T' to the ESP32
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(b'T')
            except Exception as e:
                print(f"Failed to send Tare command: {e}")

    def read_serial(self):
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            self.status_bar.config(text=f"Connected to {SERIAL_PORT}")
            
            while True:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode('utf-8').strip()
                    
                    if line:
                        try:
                            float(line)
                            self.current_angle = line
                        except ValueError:
                            pass 
                            
        except Exception as e:
            self.status_bar.config(text=f"Serial Error: Disconnected")

    def update_ui_loop(self):
        # Append degree symbol and display
        self.angle_label.config(text=f"{self.current_angle}°")
        
        self.root.after(40, self.update_ui_loop)

if __name__ == "__main__":
    window = tk.Tk()
    app = EncoderApp(window)
    window.mainloop()
#include <ESP32Encoder.h>
#include <math.h> 

ESP32Encoder encoder;

// XIAO ESP32S3 Pins
const int pinA = 3; 
const int pinB = 2;

const float PPR = 600.0;
const float COUNTS_PER_REV = PPR * 4.0;

void setup() {
  Serial.begin(115200);
  
  // Turn on internal pull-up resistors
  ESP32Encoder::useInternalWeakPullResistors = puType::up;
  
  encoder.attachFullQuad(pinA, pinB);
  encoder.clearCount();
}

void loop() {
  // 1. Check if Python sent any commands over Serial
  if (Serial.available() > 0) {
    char incomingCommand = Serial.read();
    
    // 2. If the command is 'T' or 't', reset the hardware counter to zero!
    if (incomingCommand == 'T' || incomingCommand == 't') {
      encoder.clearCount();
    }
  }

  // 3. Read raw count, calculate magnitude angle, and send to Python
  long currentCount = encoder.getCount();
  float continuousAngle = (currentCount / COUNTS_PER_REV) * 360.0;
  float magnitude = fabs(continuousAngle);
  
  Serial.println(magnitude, 2);
  
  delay(30);
}
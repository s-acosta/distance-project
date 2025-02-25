// test_miniservo
// Miniservo motor activation code:
// This code demonstrates is to test the activate servo function of the MiniServoMotor class.
// The user can enter a number in the serial monitor to activate a servo.
// When a servo is activated, it will go from rest position to active position and back to rest position

#include "MiniServoMotor.h"
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include "SharedPWM.h"

MiniServoMotor miniServos;

void setup() {
    Serial.begin(115200);
    initializePWM();
    miniServos.begin();
    
    Serial.println("Mini Servo Control");
    Serial.println("Enter [servo number (0-3)] to activate");
    Serial.println("Example: 0 activates servo 0");
}

void loop() {
    miniServos.update();
    
    if (Serial.available()) {
        int servoNum = Serial.parseInt();
        
        if (servoNum >= 0 && servoNum < 4) {
            miniServos.activate_servo(servoNum);
            Serial.print("Activating servo ");
            Serial.println(servoNum);
        }
    }
}
// test_motor.txt
// Motor Test Code:
// This code demonstrates how to control a stepper motor using the StepperMotor class.
// The motor will move to a specified position when a number is entered in the serial monitor.

#include "StepperMotor.h"
#include "Arduino.h"

#define STEP_PIN 11
#define DIR_PIN 10

StepperMotor motor(STEP_PIN, DIR_PIN);
String inputString = "";
bool stringComplete = false;

void setup() {
    Serial.begin(115200);
    Serial.println("Stepper Motor Control Ready");
    Serial.println("Enter position (e.g., '500'):");
}

void loop() {
    motor.update();  // Call this as frequently as possible
    
    // Read serial input
    while (Serial.available()) {
        char inChar = (char)Serial.read();
        if (inChar == '\n') {
            stringComplete = true;
        } else {
            inputString += inChar;
        }
    }
    
    // Process complete command
    if (stringComplete) {
        long newPosition = inputString.toInt();
        motor.moveTo(newPosition);
        
        Serial.print("Moving to position: ");
        Serial.println(newPosition);
        Serial.print("Current position: ");
        Serial.println(motor.getCurrentPosition());
        
        // Clear input
        inputString = "";
        stringComplete = false;
    }
    
    // Optional: Print position updates while moving
    static unsigned long lastPrintTime = 0;
    if (motor.isMoving() && millis() - lastPrintTime > 100) {  // Update every 200ms
        Serial.print("Current position: ");
        Serial.println(motor.getCurrentPosition());
        lastPrintTime = millis();
    }
}
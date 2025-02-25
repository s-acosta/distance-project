// servo_position_test.ino
// Simple test file for motor positioning
// Inputs a servo number (0-3) and:
//   1. Moves stepper to the position from TrialConfig.h
//   2. Activates the corresponding servo
//   3. Stays in position until new command
// Commands:
//   0-3   - Select and activate servo at its position
//   h     - Return to home position (deactivates all servos)

#include "ServoMotor.h"
#include "MiniServoMotor.h"
#include "StepperMotor.h"
#include "TrialConfig.h"
#include "SharedPWM.h"
#include <Wire.h>

// Motor instances
ServoMotor servos;
MiniServoMotor miniServos; // Not used but required for includes
StepperMotor stepper(11, 10);  // Using pins as in test_motor.ino (STEP_PIN, DIR_PIN)

// State tracking
int activeServo = -1;
bool goingHome = false;
String inputString = "";
bool stringComplete = false;

void setup() {
  Serial.begin(115200);
  
  // Initialize I2C and PWM controller
  Wire.begin();
  initializePWM();
  
  // Initialize motors
  servos.begin();
  stepper.moveTo(0); // Start at home position
  
  // Print welcome message
  Serial.println("=== Servo Position Test ===");
  Serial.println("Enter a servo ID (0-3) to move to position and activate");
  Serial.println("Enter 'h' to return to home position");
  Serial.println();
  
  // Print configured positions
  Serial.println("Configured positions in TrialConfig.h:");
  for (int i = 0; i < 4; i++) {
    Serial.print("  Servo ");
    Serial.print(i);
    Serial.print(": stepper pos = ");
    Serial.print(STEPPER_POSITIONS[i]);
    Serial.print(", servo rest = ");
    Serial.print(servos.servos[i].rest_pos);
    Serial.print(", active = ");
    Serial.println(servos.servos[i].active_pos);
  }
  
  Serial.println("\nReady for commands");
}

void loop() {
  // Update motors
  servos.update();
  stepper.update();
  
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
    processCommand(inputString);
    inputString = "";
    stringComplete = false;
  }
  
  // Removed printing when motors are in motion
}

void processCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;
  
  // Handle home command
  if (cmd.equals("h")) {
    // First deactivate any active servo
    if (activeServo >= 0) {
      servos.move_servo(activeServo, servos.servos[activeServo].rest_pos);
    }
    
    // Wait for servo to move to rest position
    while (servos.is_moving()) {
      servos.update();
      delay(10);
    }
    
    // Then go home
    stepper.moveTo(0);
    Serial.println("Returning to home position");
    activeServo = -1;
    return;
  }
  
  // Handle servo ID
  int servoNum = cmd.toInt();
  if (servoNum >= 0 && servoNum < 4) {
    // If a different servo is active, deactivate it first
    if (activeServo >= 0 && activeServo != servoNum) {
      servos.move_servo(activeServo, servos.servos[activeServo].rest_pos);
      
      // Wait for servo to move to rest position
      while (servos.is_moving()) {
        servos.update();
        delay(10);
      }
    }
    
    // Move stepper to the specified position from TrialConfig.h
    stepper.moveTo(STEPPER_POSITIONS[servoNum]);
    Serial.print("Moving to position for servo ");
    Serial.print(servoNum);
    Serial.print(" (");
    Serial.print(STEPPER_POSITIONS[servoNum]);
    Serial.println(")");
    
    // Wait for stepper to reach position
    while (stepper.isMoving()) {
      stepper.update();
      delay(10);
    }
    
    // Activate the servo
    servos.move_servo(servoNum, servos.servos[servoNum].active_pos);
    Serial.print("Activating servo ");
    Serial.println(servoNum);
    activeServo = servoNum;
    
    return;
  }
  
  // If we get here, command wasn't recognized
  Serial.println("Unknown command. Use 0-3 for servos or 'h' for home");
}
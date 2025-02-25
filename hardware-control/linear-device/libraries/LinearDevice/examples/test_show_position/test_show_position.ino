#include "ServoMotor.h"
#include "MiniServoMotor.h"
#include "StepperMotor.h"
#include "SharedPWM.h"
#include <Wire.h>

// Motor instances
ServoMotor servos;
MiniServoMotor miniServos;
StepperMotor stepper(11, 10);  // Using pins from test_motor.ino

// Constants
const unsigned long REPORT_INTERVAL = 50;  // 50ms between reports
const long STEPPER_POSITIONS[] = {100, 200, 300, 400};  // From TrialConfig.h
const unsigned long SERVO_WAIT_TIME = 1000;  // 1 second wait

// Global variables
unsigned long startTime = 0;
unsigned long lastReportTime = 0;
int currentId = -1;
int previousId = -1;

// State machine for servo control
enum ControlState {
  IDLE,
  MOVING_STEPPER,
  ACTIVATING_SERVO,
  ACTIVATING_MINI_SERVO,
  WAITING_MINI_SERVO
};

ControlState currentState = IDLE;
unsigned long stateStartTime = 0;

void setup() {
  Serial.begin(115200);
  initializePWM();
  servos.begin();
  miniServos.begin();
  
  startTime = millis();
  
  Serial.println("Servo Control Demo");
  Serial.println("Enter servo ID (0-3) to select and activate a servo");
  Serial.println("Example: '0' selects and activates servo 0");
}

void sendReport() {
  unsigned long currentTime = millis();
  
  if (currentTime - lastReportTime >= REPORT_INTERVAL) {
    // Format: <mXXXiYaZtWWWW>
    // m: stepper position
    // i: current servo ID (or -1 if none)
    // a: mini servo active (1) or idle (0)
    // t: time since start in ms
    Serial.print("<m");
    Serial.print(stepper.getCurrentPosition());
    Serial.print("i");
    Serial.print(currentId);
    Serial.print("a");
    Serial.print(miniServos.is_moving() ? "1" : "0");
    Serial.print("t");
    Serial.print(currentTime - startTime);
    Serial.println(">");
    
    lastReportTime = currentTime;
  }
}

void processInput(String input) {
  if (input.length() > 0) {
    int id = input.toInt();
    if (id >= 0 && id < 4) {
      if (currentState == IDLE) {
        currentId = id;
        
        // When a new servo is selected, reset any previous servo to rest position
        if (previousId != -1 && previousId != currentId) {
          servos.move_servo(previousId, servos.servos[previousId].rest_pos);
          // Wait for servo to move to rest
          while (servos.is_moving()) {
            servos.update();
          }
        }
        
        // Start the sequence by moving the stepper to position
        stepper.moveTo(STEPPER_POSITIONS[id]);
        currentState = MOVING_STEPPER;
        stateStartTime = millis();
        Serial.print("Selected servo ID: ");
        Serial.println(id);
        Serial.print("Moving stepper to position: ");
        Serial.println(STEPPER_POSITIONS[id]);
      } else {
        Serial.println("Already processing a servo. Please wait.");
      }
    } else {
      Serial.println("Invalid servo ID. Please enter a number between 0-3.");
    }
  }
}

void loop() {
  // Update all motors
  servos.update();
  miniServos.update();
  stepper.update();
  
  // State machine to handle servo control sequence
  switch (currentState) {
    case IDLE:
      // Do nothing, waiting for input
      break;
      
    case MOVING_STEPPER:
      if (!stepper.isMoving()) {
        // Stepper has reached position, now activate the servo
        servos.move_servo(currentId, servos.servos[currentId].active_pos);
        currentState = ACTIVATING_SERVO;
        Serial.print("Activating servo ");
        Serial.println(currentId);
      }
      break;
      
    case ACTIVATING_SERVO:
      if (!servos.is_moving()) {
        // Servo has reached active position, now activate mini servo
        miniServos.activate_servo(currentId);
        currentState = ACTIVATING_MINI_SERVO;
        Serial.print("Activating mini-servo ");
        Serial.println(currentId);
      }
      break;
      
    case ACTIVATING_MINI_SERVO:
      if (!miniServos.is_moving()) {
        // Mini servo has completed its activation sequence
        stateStartTime = millis();
        currentState = WAITING_MINI_SERVO;
        Serial.println("Mini-servo activation complete. Waiting...");
      }
      break;
      
    case WAITING_MINI_SERVO:
      if (millis() - stateStartTime >= SERVO_WAIT_TIME) {
        // Wait complete, return to idle state
        previousId = currentId;  // Remember this ID for next time
        currentState = IDLE;
        Serial.println("Sequence complete. Ready for next servo ID.");
      }
      break;
  }
  
  // Send periodic status report
  sendReport();
  
  // Check for user input
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    processInput(input);
  }
}
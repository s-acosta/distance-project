/*
  Radial Device Control - Command Reference
  
  ============================================
  STEPPER MOTOR COMMANDS
  ============================================
  
  move <angle>
    - Move to target angle using current speed
    - Example: move -30
    - Response: <Moving to -30 at 20000>
    - Completion: <Movement done: -30>
  
  move <angle> <speed>
    - Move to target angle and set speed (microseconds delay)
    - Example: move 60 28000
    - Response: <Moving to 60 at 28000>
    - Completion: <Movement done: 60>
  
  velocity <speed>
    - Set stepper speed (5000-100000 microseconds)
    - Example: velocity 15000
    - Response: <Velocity set to 15000>
  
  speed <preset>
    - Set speed preset (1=slow/50000, 2=medium/20000, 3=fast/8000)
    - Example: speed 2
    - Response: <Speed preset 2>
  
  ============================================
  SERVO MOTOR COMMANDS
  ============================================
  
  raise <id>
    - Raise servo to active position (default time)
    - Example: raise 2
    - Response: <Raising servo 2>
    - Completion: <Servo done>
  
  raise <id> <time>
    - Raise servo to active position with custom time (milliseconds)
    - Example: raise 2 500
    - Response: <Raising servo 2 in 500 ms>
    - Completion: <Servo done>
  
  lower <id>
    - Lower servo to rest position (default time)
    - Example: lower 2
    - Response: <Lowering servo 2>
    - Completion: <Servo done>
  
  lower <id> <time>
    - Lower servo to rest position with custom time (milliseconds)
    - Example: lower 2 250
    - Response: <Lowering servo 2 in 250 ms>
    - Completion: <Servo done>
  
  servo <id> <pos>
    - Move servo to specific position (default time)
    - Example: servo 2 300
    - Response: <Moving servo 2 to 300>
    - Completion: <Servo done>
  
  servo <id> <pos> <time>
    - Move servo to specific position with custom time (milliseconds)
    - Example: servo 2 300 1000
    - Response: <Moving servo 2 to 300 in 1000 ms>
    - Completion: <Servo done>
  
  ============================================
  SYSTEM COMMANDS
  ============================================
  
  status
    - Get current system status
    - Response: <Status: angle=-30, target=-30, velocity=20000, servo_moving=no>
  
  reset
    - Reset encoder, stepper position, and all servos to rest
    - Response: <Reset complete>
  
  stop
    - Emergency stop stepper motor
    - Response: <Stopped>
  
  ============================================
  SERVO CONFIGURATION (in setup())
  ============================================
  Servo 0: Pin 12, Rest=400, Active=180
  Servo 1: Pin 8,  Rest=400, Active=180
  Servo 2: Pin 4,  Rest=400, Active=180
  Servo 3: Pin 0,  Rest=400, Active=180
  
  ============================================
  NOTES
  ============================================
  - All commands must end with newline character (\n)
  - Angle tolerance: ±1 degree
  - Valid servo IDs: 0-3
  - Stepper speed range: 5000-100000 microseconds
  - All responses wrapped in < > brackets
  - Encoder reports current angle continuously during movements
  
  Created: 2025
*/


#include <Arduino.h>
#include "Encoder.h"
#include "StepperMotor.h"
#include "ServoMotor.h"
#include "SharedPWM.h"

// === Pin definitions ===
#define STEP_PIN   11
#define DIR_PIN    12
#define ENCODER_A  18
#define ENCODER_B  19

// === Objects ===
StepperMotor stepper(STEP_PIN, DIR_PIN);
Encoder encoder(ENCODER_A, ENCODER_B, false);
ServoMotor servos(500);

// === Variables ===
int targetEncoderAngle = 0;
int currentEncoderAngle = 0;
const int ANGLE_TOLERANCE = 1;

bool waitingForCommand = true;
bool servoWasMoving = false;

// Velocity control
unsigned long currentStepDelay = 20000;
const unsigned long MIN_STEP_DELAY = 5000;
const unsigned long MAX_STEP_DELAY = 100000;

// === Serial input handling ===
String inputString = "";
bool stringComplete = false;

void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}
  
  // Initialize encoder
  encoder.begin();
  encoder.resetCounter();
  
  // Initialize stepper
  stepper.setStepDelay(currentStepDelay);
  stepper.resetPosition();
  
  // Initialize PWM and servos
  initializePWM();
  servos.setServoConfig(0, 12, 400, 180);
  servos.setServoConfig(1,  8, 400, 180);
  servos.setServoConfig(2,  4, 400, 180);
  servos.setServoConfig(3,  0, 420, 180);
  servos.begin();
  
  Serial.println("<Arduino is ready>");
}

void handleCommand(String cmd) {
  cmd.trim();
  
  if (cmd == "reset") {
    stepper.stopContinuousMove();
    stepper.resetPosition();
    encoder.resetCounter();
    waitingForCommand = true;
    for (int i = 0; i < servos.NUM_SERVOS; i++) {
      servos.move_servo(i, servos.servos[i].rest_pos);
    }
    Serial.println("<Reset complete>");
    
  } else if (cmd == "status") {
    Serial.print("<Status: angle=");
    Serial.print(currentEncoderAngle);
    Serial.print(", target=");
    Serial.print(targetEncoderAngle);
    Serial.print(", velocity=");
    Serial.print(currentStepDelay);
    Serial.print(", servo_moving=");
    Serial.print(servos.is_moving() ? "yes" : "no");
    Serial.println(">");
    
  } else if (cmd == "stop") {
    stepper.stopContinuousMove();
    waitingForCommand = true;
    Serial.println("<Stopped>");
    
  } else if (cmd.startsWith("velocity ")) {
    unsigned long newDelay = cmd.substring(9).toInt();
    if (newDelay >= MIN_STEP_DELAY && newDelay <= MAX_STEP_DELAY) {
      currentStepDelay = newDelay;
      stepper.setStepDelay(currentStepDelay);
      Serial.print("<Velocity set to ");
      Serial.print(currentStepDelay);
      Serial.println(">");
    } else {
      Serial.println("<Invalid velocity>");
    }
    
  } else if (cmd.startsWith("speed ")) {
    int preset = cmd.substring(6).toInt();
    switch(preset) {
      case 1:
        currentStepDelay = 50000;
        break;
      case 2:
        currentStepDelay = 20000;
        break;
      case 3:
        currentStepDelay = 8000;
        break;
      default:
        Serial.println("<Invalid speed preset>");
        return;
    }
    stepper.setStepDelay(currentStepDelay);
    Serial.print("<Speed preset ");
    Serial.print(preset);
    Serial.println(">");
    
  } else if (cmd.startsWith("move ")) {
    // Format: move <angle> OR move <angle> <speed>
    String data = cmd.substring(5);
    int space = data.indexOf(' ');
    
    if (space == -1) {
      // Format: move <angle> - use current speed
      int angle = data.toInt();
      targetEncoderAngle = angle;
      waitingForCommand = false;
      Serial.print("<Moving to ");
      Serial.print(targetEncoderAngle);
      Serial.print(" at ");
      Serial.print(currentStepDelay);
      Serial.println(">");
    } else {
      // Format: move <angle> <speed>
      int angle = data.substring(0, space).toInt();
      unsigned long speed = data.substring(space + 1).toInt();
      
      if (speed >= MIN_STEP_DELAY && speed <= MAX_STEP_DELAY) {
        currentStepDelay = speed;
        stepper.setStepDelay(currentStepDelay);
        targetEncoderAngle = angle;
        waitingForCommand = false;
        Serial.print("<Moving to ");
        Serial.print(targetEncoderAngle);
        Serial.print(" at ");
        Serial.print(currentStepDelay);
        Serial.println(">");
      } else {
        Serial.println("<Invalid speed>");
      }
    }
    
  } else if (cmd.startsWith("raise ")) {
    // Format: raise <id> OR raise <id> <time>
    String data = cmd.substring(6);
    int space = data.indexOf(' ');
    
    if (space == -1) {
      // Format: raise <id> - use active position with default time
      int id = data.toInt();
      if (id >= 0 && id < servos.NUM_SERVOS) {
        servos.move_servo(id, servos.servos[id].active_pos);
        Serial.print("<Raising servo ");
        Serial.print(id);
        Serial.println(">");
      } else {
        Serial.println("<Invalid servo ID>");
      }
    } else {
      // Format: raise <id> <time>
      int id = data.substring(0, space).toInt();
      int time_ms = data.substring(space + 1).toInt();
      
      if (id >= 0 && id < servos.NUM_SERVOS) {
        servos.move_servo(id, servos.servos[id].active_pos, time_ms);
        Serial.print("<Raising servo ");
        Serial.print(id);
        Serial.print(" in ");
        Serial.print(time_ms);
        Serial.println(" ms>");
      } else {
        Serial.println("<Invalid servo ID>");
      }
    }
    
  } else if (cmd.startsWith("lower ")) {
    // Format: lower <id> OR lower <id> <time>
    String data = cmd.substring(6);
    int space = data.indexOf(' ');
    
    if (space == -1) {
      // Format: lower <id> - use rest position with default time
      int id = data.toInt();
      if (id >= 0 && id < servos.NUM_SERVOS) {
        servos.move_servo(id, servos.servos[id].rest_pos);
        Serial.print("<Lowering servo ");
        Serial.print(id);
        Serial.println(">");
      } else {
        Serial.println("<Invalid servo ID>");
      }
    } else {
      // Format: lower <id> <time>
      int id = data.substring(0, space).toInt();
      int time_ms = data.substring(space + 1).toInt();
      
      if (id >= 0 && id < servos.NUM_SERVOS) {
        servos.move_servo(id, servos.servos[id].rest_pos, time_ms);
        Serial.print("<Lowering servo ");
        Serial.print(id);
        Serial.print(" in ");
        Serial.print(time_ms);
        Serial.println(" ms>");
      } else {
        Serial.println("<Invalid servo ID>");
      }
    }
    
  } else if (cmd.startsWith("servo ")) {
    String data = cmd.substring(6);
    
    // Parse: servo <id> <pos> OR servo <id> <pos> <time>
    int firstSpace = data.indexOf(' ');
    if (firstSpace == -1) {
      Serial.println("<Error: Use 'servo <id> <pos>' or 'servo <id> <pos> <time>'>");
      return;
    }
    
    int id = data.substring(0, firstSpace).toInt();
    int secondSpace = data.indexOf(' ', firstSpace + 1);
    
    if (secondSpace == -1) {
      // Format: servo <id> <pos>
      int pos = data.substring(firstSpace + 1).toInt();
      if (id >= 0 && id < servos.NUM_SERVOS) {
        servos.move_servo(id, pos);
        Serial.print("<Moving servo ");
        Serial.print(id);
        Serial.print(" to ");
        Serial.print(pos);
        Serial.println(">");
      } else {
        Serial.println("<Invalid servo ID>");
      }
    } else {
      // Format: servo <id> <pos> <time>
      int pos = data.substring(firstSpace + 1, secondSpace).toInt();
      int movement_time = data.substring(secondSpace + 1).toInt();
      if (id >= 0 && id < servos.NUM_SERVOS) {
        servos.move_servo(id, pos, movement_time);
        Serial.print("<Moving servo ");
        Serial.print(id);
        Serial.print(" to ");
        Serial.print(pos);
        Serial.print(" in ");
        Serial.print(movement_time);
        Serial.println(" ms>");
      } else {
        Serial.println("<Invalid servo ID>");
      }
    }
    
  } else {
    Serial.println("<Unknown command>");
  }
}

void loop() {
  // Update components
  stepper.update();
  servos.update();
  
  // Read current encoder position
  currentEncoderAngle = encoder.readAngle();
  
  // Check if servo just finished moving
  if (servoWasMoving && !servos.is_moving()) {
    Serial.println("<Servo done>");
    servoWasMoving = false;
  }
  
  // Track if servo is moving
  if (servos.is_moving()) {
    servoWasMoving = true;
  }
  
  // Control loop for stepper
  if (!waitingForCommand) {
    int error = targetEncoderAngle - currentEncoderAngle;
    
    if (abs(error) < ANGLE_TOLERANCE) {
      if (stepper.isContinuousMode()) {
        stepper.stopContinuousMove();
        Serial.print("<Movement done: ");
        Serial.print(currentEncoderAngle);
        Serial.println(">");
        waitingForCommand = true;
      }
    } else {
      if (!stepper.isContinuousMode()) {
        if (error > 0) {
          stepper.startContinuousMove(true);
        } else {
          stepper.startContinuousMove(false);
        }
      } else {
        bool currentDir = (error > 0);
        if ((currentDir && error < 0) || (!currentDir && error > 0)) {
          stepper.stopContinuousMove();
        }
      }
    }
  }
  
  // Handle serial commands
  serialEvent();
  if (stringComplete) {
    handleCommand(inputString);
    inputString = "";
    stringComplete = false;
  }
}
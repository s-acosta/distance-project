#include "ServoMotor.h"
#include "MiniServoMotor.h"
#include "StepperMotor.h"
#include "SharedPWM.h"
#include <Wire.h>

ServoMotor servos;
MiniServoMotor miniServos;
StepperMotor stepper(11, 10);  // Using pins from test_motor.txt

void printCurrentState() {
    // Print stepper position
    Serial.print("Stepper Position: ");
    Serial.println(stepper.getCurrentPosition());
    
    // Print servo positions
    for(int i = 0; i < 4; i++) {
        Serial.print("Servo ");
        Serial.print(i);
        Serial.print(": ");
        Serial.println(servos.servos[i].current_pos);
    }
    
    // Print mini servo positions
    for(int i = 0; i < 4; i++) {
        Serial.print("Mini-servo ");
        Serial.print(i);
        Serial.print(": ");
        Serial.print(miniServos.mini_servos[i].current_pos);
        Serial.println("");
    }
    Serial.println("-------------------");
}

void setup() {
    Serial.begin(115200);
    initializePWM();
    servos.begin();
    miniServos.begin();

    Serial.println("Motor Calibration Program");
    Serial.println("Commands:");
    Serial.println("  mXXX  - Move stepper to position XXX");
    Serial.println("  sNXXX - Move servo N to position XXX");
    Serial.println("  nNXXX - Move mini-servo N to position XXX");
    Serial.println("  p     - Print current positions");
    Serial.println("Example: m200 moves stepper to position 200");
    Serial.println("         s0467 moves servo 0 to position 467");
    Serial.println("         n0300 moves mini-servo 0 to position 300");
}

void loop() {
    servos.update();
    miniServos.update();
    stepper.update();
    
    if (Serial.available()) {
        String input = Serial.readStringUntil('\n');
        input.trim();
        
        if (input.length() > 0) {
            char command = input.charAt(0);
            
            switch(command) {
                case 'm': {  // Stepper motor command
                    long position = input.substring(1).toInt();
                    stepper.moveTo(position);
                    Serial.print("Moving stepper to position: ");
                    Serial.println(position);
                    break;
                }
                
                case 's': {  // Servo command
                    if (input.length() >= 2) {
                        int servoNum = input.charAt(1) - '0';
                        int position = input.substring(2).toInt();
                        if (servoNum >= 0 && servoNum < 4) {
                            servos.move_servo(servoNum, position);
                            Serial.print("Moving servo ");
                            Serial.print(servoNum);
                            Serial.print(" to position: ");
                            Serial.println(position);
                        }
                    }
                    break;
                }
                
                case 'n': {  // Mini-servo command
                    if (input.length() >= 2) {
                        int servoNum = input.charAt(1) - '0';
                        int position = input.substring(2).toInt();
                        if (servoNum >= 0 && servoNum < 4) {
                            miniServos.move_to_position(servoNum, position);
                            Serial.print("Moving mini-servo ");
                            Serial.print(servoNum);
                            Serial.print(" to position: ");
                            Serial.println(position);
                        }
                    }
                    break;
                }
                
                case 'p': {  // Print current state
                    printCurrentState();
                    break;
                }
                
                default:
                    Serial.println("Invalid command");
                    break;
            }
        }
    }
}
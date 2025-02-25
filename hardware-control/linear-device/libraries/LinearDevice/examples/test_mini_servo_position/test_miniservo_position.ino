// test_miniservo_position.txt
// Mini-Servo motor control code:
// This code demonstrates how to control a mini-servo motor using the Adafruit_PWMServoDriver library 
// and the ServoConfig.h file. The motor will move to a specified position when a number is entered

#include "MiniServoMotor.h"
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include "SharedPWM.h"

MiniServoMotor miniServos;

void setup() {
   Serial.begin(115200);
   initializePWM();
   miniServos.begin();

   Serial.println("Mini Servo Position Control");
   Serial.println("Enter [servo number (0-3)][position]"); 
   Serial.println("Example: 0500 moves servo 0 to position 500");
}

void loop() {
   miniServos.update();
   
   if (Serial.available()) {
       String input = Serial.readStringUntil('\n');

       if (input.length() > 1) {
           int servoNum = input.substring(0, 1).toInt();
           int position = input.substring(1).toInt();
           
           if (servoNum >= 0 && servoNum < 4) {
            miniServos.move_to_position(servoNum, position);
            Serial.print("Mini Servo ");
            Serial.print(servoNum);
            Serial.print(" to pos ");
            Serial.println(position);
        }
       }
   }
}
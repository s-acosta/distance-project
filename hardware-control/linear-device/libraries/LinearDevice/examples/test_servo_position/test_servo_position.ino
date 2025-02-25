// test_servo_position.txt
// Servo motor control code:
// This code demonstrates how to control a servo motor using the Adafruit_PWMServoDriver library 
// and the ServoConfig.h file. The motor will move to a specified position when a number is entered

#include "ServoConfig.h"
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include "SharedPWM.h"

void setup()
{
    Serial.begin(115200);
    initializePWM();

    Serial.println("Servo Position Control");
    Serial.println("Enter [servo number (0-3)][position]");
    Serial.println("Example: 0500 moves servo 0 to position 500");
}

void loop()
{
    if (Serial.available())
    {
        String input = Serial.readStringUntil('\n');

        if (input.length() > 1) {  // Make sure we have at least 2 characters
            int servoNum = input.substring(0, 1).toInt();
            int position = input.substring(1).toInt();
            
            if (servoNum >= 0 && servoNum < 4) {
                pwm.setPWM(servos[servoNum].pin, 0, position);
                Serial.print("Servo ");
                Serial.print(servoNum);
                Serial.print(" to pos ");
                Serial.println(position);
            }
        }
    }
}
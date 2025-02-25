#include "TrialHistory.h"
#include "SharedPWM.h"
#include <Wire.h>

ServoMotor servos;
MiniServoMotor miniServos;
StepperMotor stepper(11, 10);
TrialHistory trial(servos, miniServos, stepper);

void setup() {
    Serial.begin(115200);
    Wire.begin();
    initializePWM();
    servos.begin();
    miniServos.begin();
    trial.begin();
}

void loop() {
    servos.update();
    miniServos.update();
    stepper.update();
    trial.update();
    
    if (Serial.available()) {
        String input = Serial.readStringUntil('\n');
        trial.processInput(input);
    }
}
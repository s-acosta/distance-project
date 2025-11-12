#include <Arduino.h>
#include "StepperMotor.h"

# define STEP_PIN 11
# define DIR_PIN 12

StepperMotor stepper(STEP_PIN, DIR_PIN);

String inputString = "";
bool stringComplete = false;

void serialEvent() {
	// Read serial input
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
	Serial.println("StepperMotor example");
	stepper.setDirection(1);
	stepper.setStepDelay(15000);
}

void loop() {

	stepper.update();
	serialEvent();
	if (stringComplete) {
		
		Serial.println("Current position: " + String(stepper.getCurrentPosition()));
		Serial.println("Moving to position: " + inputString);

		long newPosition = inputString.toInt();
		stepper.moveTo(newPosition);
		inputString = "";
		stringComplete = false;
	}
}

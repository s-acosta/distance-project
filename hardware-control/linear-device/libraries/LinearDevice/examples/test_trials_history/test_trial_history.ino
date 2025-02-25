#include "ServoMotor.h"
#include "MiniServoMotor.h"
#include "StepperMotor.h"
#include "SharedPWM.h"
#include <Wire.h>

// Motor instances
ServoMotor servos;
MiniServoMotor miniServos;
StepperMotor stepper(11, 10); // From test_motor.txt

// Sequence parameters
const long STEPPER_FINAL_POS = 500;
const unsigned long FINAL_WAIT_TIME = 200;
const long STEPPER_POSITIONS[] = {100, 200, 300, 400};
const unsigned long POSITION_WAIT_TIME = 2000;
String input;

enum class SequenceState
{
    WAITING_FOR_INPUT,
    SERVO_TO_ACTIVE,
    STEPPER_TO_FINAL,
    FINAL_WAIT,
    STEPPER_TO_POSITION,
    POSITION_WAIT,
    MINI_SERVO_ACTIVATE,
    SERVO_TO_REST,
    STEPPER_HOME,
    COMPLETE
};

SequenceState currentState = SequenceState::WAITING_FOR_INPUT;
int currentId = -1;
unsigned long waitStartTime = 0;

void setup()
{
    Serial.begin(115200); // Matches test files
    Wire.begin();
    initializePWM();
    servos.begin();
    miniServos.begin();

    Serial.println("Motor Sequence Control Ready");
    Serial.println("Enter servo ID (0-3) to start sequence");
}

void loop()
{
    servos.update();
    miniServos.update();
    stepper.update();

    switch (currentState)
    {
    case SequenceState::WAITING_FOR_INPUT:
        if (Serial.available())
        {
            // Adding block scope
            input = Serial.readStringUntil('\n');
            if (input.length() > 0)
            {
                currentId = input.toInt();
                if (currentId >= 0 && currentId < 4)
                {
                    servos.move_servo(currentId, servos.servos[currentId].active_pos);
                    Serial.print("Starting sequence with ID: ");
                    Serial.println(currentId);
                    currentState = SequenceState::SERVO_TO_ACTIVE;
                }
            }
        }
        break;

    case SequenceState::SERVO_TO_ACTIVE:
        if (!servos.is_moving())
        {
            stepper.moveTo(STEPPER_FINAL_POS);
            currentState = SequenceState::STEPPER_TO_FINAL;
        }
        break;

    case SequenceState::STEPPER_TO_FINAL:
        if (!stepper.isMoving())
        {
            waitStartTime = millis();
            currentState = SequenceState::FINAL_WAIT;
        }
        break;

    case SequenceState::FINAL_WAIT:
        if (millis() - waitStartTime >= FINAL_WAIT_TIME)
        {
            stepper.moveTo(STEPPER_POSITIONS[currentId]);
            currentState = SequenceState::STEPPER_TO_POSITION;
        }
        break;

    case SequenceState::STEPPER_TO_POSITION:
        if (!stepper.isMoving())
        {
            waitStartTime = millis();
            currentState = SequenceState::POSITION_WAIT;
        }
        break;

    case SequenceState::POSITION_WAIT:
        if (millis() - waitStartTime >= POSITION_WAIT_TIME)
        {
            miniServos.activate_servo(currentId);
            currentState = SequenceState::MINI_SERVO_ACTIVATE;
        }
        break;

    case SequenceState::MINI_SERVO_ACTIVATE:
        if (!miniServos.is_moving())
        {
            servos.move_servo(currentId, servos.servos[currentId].rest_pos);
            currentState = SequenceState::SERVO_TO_REST;
        }
        break;

    case SequenceState::SERVO_TO_REST:
        if (!servos.is_moving())
        {
            stepper.moveTo(0);
            currentState = SequenceState::STEPPER_HOME;
        }
        break;

    case SequenceState::STEPPER_HOME:
        if (!stepper.isMoving())
        {
            currentState = SequenceState::COMPLETE;
            Serial.println("Sequence complete");
        }
        break;

    case SequenceState::COMPLETE:
        currentState = SequenceState::WAITING_FOR_INPUT;
        break;
    }
}
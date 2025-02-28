#include "TrialHistory.h"

TrialHistory::TrialHistory(ServoMotor &servos, MiniServoMotor &miniServos, StepperMotor &stepper)
    : servos(servos), miniServos(miniServos), stepper(stepper), currentState(HistoryState::WAITING_FOR_INPUT), currentId(-1), waitStartTime(0)
{
}

void TrialHistory::begin()
{
    currentState = HistoryState::WAITING_FOR_INPUT;
    currentId = -1;
    Serial.println("Enter servo ID (0-3)");
}

void TrialHistory::processInput(const String &input)
{
    if (currentState != HistoryState::WAITING_FOR_INPUT)
        return;

    if (input.length() > 0)
    {
        currentId = input.toInt();
        if (currentId >= 0 && currentId < 4)
        {
            // Set different starting positions based on servo ID
            if (currentId == 0 || currentId == 1) {
                // For servo 0 and 1, start at position 0
                stepper.moveTo(150);
            } else {
                // For servo 2 and 3, start at position 100
                stepper.moveTo(0);
            }
            
            // Change state to a new state that waits for stepper to reach position
            currentState = HistoryState::STEPPER_TO_START;
            
            Serial.print("Moving to start position for ID: ");
            Serial.println(currentId);
        }
    }
}

void TrialHistory::update()
{
    switch (currentState)
    {
    case HistoryState::WAITING_FOR_INPUT:
        // Handled by processInput()
        break;

    case HistoryState::STEPPER_TO_START:
        // Wait for stepper to reach the initial position (0 or 100)
        if (!stepper.isMoving())
        {
            // Now that stepper is in position, activate the servo
            servos.move_servo(currentId, servos.servos[currentId].active_pos);
            currentState = HistoryState::SERVO_TO_ACTIVE;
        }
        break;

    case HistoryState::SERVO_TO_ACTIVE:
        if (!servos.is_moving())
        {
            stepper.moveTo(STEPPER_FINAL_POS);
            currentState = HistoryState::STEPPER_TO_FINAL;
        }
        break;

    case HistoryState::STEPPER_TO_FINAL:
        if (!stepper.isMoving())
        {
            waitStartTime = millis();
            currentState = HistoryState::FINAL_WAIT;
        }
        break;

    case HistoryState::FINAL_WAIT:
        if (millis() - waitStartTime >= FINAL_WAIT_TIME)
        {
            stepper.moveTo(STEPPER_POSITIONS[currentId]);
            currentState = HistoryState::STEPPER_TO_POSITION;
        }
        break;

    case HistoryState::STEPPER_TO_POSITION:
        if (!stepper.isMoving())
        {
            waitStartTime = millis();
            currentState = HistoryState::POSITION_WAIT;
        }
        break;

    case HistoryState::POSITION_WAIT:
        if (millis() - waitStartTime >= POSITION_WAIT_TIME)
        {
            miniServos.activate_servo(currentId);
            currentState = HistoryState::MINI_SERVO_ACTIVATE;
        }
        break;

    case HistoryState::MINI_SERVO_ACTIVATE:
        if (!miniServos.is_moving())
        {
            waitStartTime = millis();
            currentState = HistoryState::MINI_SERVO_HOLD;
        }
        break;

    case HistoryState::MINI_SERVO_HOLD:
        if (millis() - waitStartTime >= MINI_SERVO_HOLD_TIME)
        {
            // Add additional wait time (200ms) after mini servo activation
            waitStartTime = millis();
            currentState = HistoryState::POST_ACTIVATION_WAIT;
        }
        break;
        
    case HistoryState::POST_ACTIVATION_WAIT:
        if (millis() - waitStartTime >= POST_ACTIVATION_WAIT_TIME) // 200ms wait
        {
            servos.move_servo(currentId, servos.servos[currentId].rest_pos);
            currentState = HistoryState::SERVO_TO_REST;
        }
        break;

    case HistoryState::SERVO_TO_REST:
        if (!servos.is_moving())
        {
            // Change from original: Use position 150 instead of 0
            stepper.moveTo(0); // Home position is now 150
            currentState = HistoryState::STEPPER_HOME;
        }
        break;

    case HistoryState::STEPPER_HOME:
        if (!stepper.isMoving())
        {
            currentState = HistoryState::COMPLETE;
            Serial.println("Sequence complete");
            Serial.println("Enter servo ID (0-3)");
        }
        break;

    case HistoryState::COMPLETE:
        currentState = HistoryState::WAITING_FOR_INPUT;
        break;
    }
}

bool TrialHistory::isComplete() const
{
    return currentState == HistoryState::COMPLETE;
}
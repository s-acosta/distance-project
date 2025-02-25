#include "TrialNoHistory.h"

TrialNoHistory::TrialNoHistory(ServoMotor& servos, MiniServoMotor& miniServos, StepperMotor& stepper)
    : servos(servos)
    , miniServos(miniServos)
    , stepper(stepper)
    , currentState(NoHistoryState::WAITING_FOR_INPUT)
    , currentId(-1)
    , waitStartTime(0)
{
}

void TrialNoHistory::begin() {
    currentState = NoHistoryState::WAITING_FOR_INPUT;
    currentId = -1;
}

void TrialNoHistory::processInput(const String& input) {
    if (currentState != NoHistoryState::WAITING_FOR_INPUT) return;
    
    if (input.length() > 0) {
        currentId = input.toInt();
        if (currentId >= 0 && currentId < 4) {
            stepper.moveTo(STEPPER_POSITIONS[currentId]);
            currentState = NoHistoryState::STEPPER_TO_POSITION;
        }
    }
}

void TrialNoHistory::update() {
    switch (currentState) {
    case NoHistoryState::WAITING_FOR_INPUT:
        // Handled by processInput()
        break;

    case NoHistoryState::STEPPER_TO_POSITION:
        if (!stepper.isMoving()) {
            servos.move_servo(currentId, servos.servos[currentId].active_pos);
            currentState = NoHistoryState::SERVO_TO_ACTIVE;
        }
        break;

    case NoHistoryState::SERVO_TO_ACTIVE:
        if (!servos.is_moving()) {
            waitStartTime = millis();
            currentState = NoHistoryState::SERVO_WAIT;
        }
        break;

    case NoHistoryState::SERVO_WAIT:
        if (millis() - waitStartTime >= SERVO_WAIT_TIME) {
            miniServos.activate_servo(currentId);
            currentState = NoHistoryState::MINI_SERVO_ACTIVATE;
        }
        break;

    // In the state machine, replace the MINI_SERVO_ACTIVATE case with:
    case NoHistoryState::MINI_SERVO_ACTIVATE:
        if (!miniServos.is_moving())
        {
            waitStartTime = millis();
            currentState = NoHistoryState::MINI_SERVO_HOLD;
        }
        break;

    case NoHistoryState::MINI_SERVO_HOLD:
        if (millis() - waitStartTime >= MINI_SERVO_HOLD_TIME)  // Using MiniServoMotor's HOLD_TIME
        {
            servos.move_servo(currentId, servos.servos[currentId].rest_pos);
            currentState = NoHistoryState::SERVO_TO_REST;
        }
        break;

    case NoHistoryState::SERVO_TO_REST:
        if (!servos.is_moving()) {
            stepper.moveTo(0);
            currentState = NoHistoryState::STEPPER_HOME;
        }
        break;

    case NoHistoryState::STEPPER_HOME:
        if (!stepper.isMoving()) {
            currentState = NoHistoryState::COMPLETE;
            Serial.println("Ready for next sequence");
        }
        break;

    case NoHistoryState::COMPLETE:
        currentState = NoHistoryState::WAITING_FOR_INPUT;
        break;
    }
}

bool TrialNoHistory::isComplete() const {
    return currentState == NoHistoryState::COMPLETE;
}
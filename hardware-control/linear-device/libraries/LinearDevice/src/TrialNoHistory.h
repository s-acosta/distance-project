#ifndef TRIAL_NO_HISTORY_H
#define TRIAL_NO_HISTORY_H

#include "ServoMotor.h"
#include "MiniServoMotor.h"
#include "StepperMotor.h"
#include "TrialConfig.h"

enum class NoHistoryState {
    WAITING_FOR_INPUT,
    STEPPER_TO_POSITION,
    SERVO_TO_ACTIVE,
    MINI_SERVO_ACTIVATE,
    MINI_SERVO_HOLD,   
    SERVO_WAIT,
    SERVO_TO_REST,
    STEPPER_HOME,
    COMPLETE
};

class TrialNoHistory {
public:
    TrialNoHistory(ServoMotor& servos, MiniServoMotor& miniServos, StepperMotor& stepper);
    void begin();
    void update();
    void processInput(const String& input);
    bool isComplete() const;

private:
    ServoMotor& servos;
    MiniServoMotor& miniServos;
    StepperMotor& stepper;

    static const unsigned long MINI_SERVO_HOLD_TIME = 100;
    
    NoHistoryState currentState;
    int currentId;
    unsigned long waitStartTime;
};

#endif
#ifndef TRIAL_HISTORY_H
#define TRIAL_HISTORY_H

#include "ServoMotor.h"
#include "MiniServoMotor.h"
#include "StepperMotor.h"
#include "TrialConfig.h"

enum class HistoryState {
    WAITING_FOR_INPUT,
    SERVO_TO_ACTIVE,
    STEPPER_TO_FINAL,
    FINAL_WAIT,
    STEPPER_TO_POSITION,
    POSITION_WAIT,
    MINI_SERVO_ACTIVATE,
    MINI_SERVO_HOLD,
    SERVO_TO_REST,
    STEPPER_HOME,
    COMPLETE
};

class TrialHistory {
public:
    TrialHistory(ServoMotor& servos, MiniServoMotor& miniServos, StepperMotor& stepper);
    void begin();
    void update();
    void processInput(const String& input);
    bool isComplete() const;

private:
    static const long STEPPER_FINAL_POS = 500;
    static const unsigned long FINAL_WAIT_TIME = 200;
    static const unsigned long MINI_SERVO_HOLD_TIME = 100;
    static const long START_POSITION = 200; // Added new constant for start position

    ServoMotor& servos;
    MiniServoMotor& miniServos;
    StepperMotor& stepper;
    
    HistoryState currentState;
    int currentId;
    unsigned long waitStartTime;
};

#endif
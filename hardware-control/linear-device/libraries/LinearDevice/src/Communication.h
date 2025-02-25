#ifndef COMMUNICATION_H
#define COMMUNICATION_H

#include "ServoMotor.h"
#include "MiniServoMotor.h"
#include "StepperMotor.h"
#include "TrialHistory.h"
#include "TrialNoHistory.h"

// Reports status to MATLAB
void sendReport(StepperMotor& stepper, MiniServoMotor& miniServos, 
               int currentId, unsigned long startTime, unsigned long& lastReportTime);

// Processes input from MATLAB
void processInput(String input, int& currentId, bool& useHistory, 
                 TrialHistory& historyTrial, TrialNoHistory& noHistoryTrial);

#endif
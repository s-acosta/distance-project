#include "Communication.h"

const unsigned long REPORT_INTERVAL = 50;  // 50ms between reports

void sendReport(StepperMotor& stepper, MiniServoMotor& miniServos, 
               int currentId, unsigned long startTime, unsigned long& lastReportTime) 
{
    unsigned long currentTime = millis();
    
    if (currentTime - lastReportTime >= REPORT_INTERVAL) {
        // Format: <mXXXiYaZtWWWW>
        // m: stepper position
        // i: current servo ID (or -1 if none)
        // a: mini servo active (1) or idle (0)
        // t: time since start in ms
        Serial.print("<m");
        Serial.print(stepper.getCurrentPosition());
        Serial.print("i");
        Serial.print(currentId);
        Serial.print("a");
        Serial.print(miniServos.is_moving() ? "1" : "0");
        Serial.print("t");
        Serial.print(currentTime - startTime);
        Serial.println(">");
        
        lastReportTime = currentTime;
    }
}

void processInput(String input, int& currentId, bool& useHistory, 
                 TrialHistory& historyTrial, TrialNoHistory& noHistoryTrial) 
{
    // Expected format: <ID,type>
    // Example: <2,h> for servo 2 with history
    // Example: <1,n> for servo 1 without history
    
    // Check for valid markers
    if (input.charAt(0) != '<' || input.charAt(input.length()-1) != '>') {
        return;  // Invalid format, ignore command
    }
    
    // Remove markers and get actual command
    String command = input.substring(1, input.length()-1);
    
    int commaIndex = command.indexOf(',');
    if (commaIndex != -1) {
        int id = input.substring(0, commaIndex).toInt();
        char type = input.charAt(commaIndex + 1);
        
        if (id >= 0 && id < 4) {
            currentId = id;
            useHistory = (type == 'h');
            
            // Convert input format for trial modules
            String trialInput = String(id);
            
            if (useHistory) {
                historyTrial.processInput(trialInput);
            } else {
                noHistoryTrial.processInput(trialInput);
            }
        }
    }
}
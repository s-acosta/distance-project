#include <Communication.h>
#include <MiniServoMotor.h>
#include <ServoMotor.h>
#include <SharedPWM.h>
#include <StepperMotor.h>
#include <TrialConfig.h>
#include <TrialHistory.h>
#include <TrialNoHistory.h>

#include "TrialHistory.h"
#include "TrialNoHistory.h"
#include "SharedPWM.h"

// Motor instances
ServoMotor servos;
MiniServoMotor miniServos;
StepperMotor stepper(11, 10);

// Trial instances
TrialHistory historyTrial(servos, miniServos, stepper);
TrialNoHistory noHistoryTrial(servos, miniServos, stepper);

// Constants
const int TRIGGER_PIN = 9;
const unsigned long REPORT_INTERVAL = 50;  // 50ms between reports

// Global variables
unsigned long startTime = 0;
unsigned long lastReportTime = 0;
bool isTriggered = false;
bool triggerEnabled = false;
bool useHistory = false;
int currentId = -1;

void setup() {

    Serial.begin(115200);
    initializePWM();
    servos.begin();
    miniServos.begin();
    historyTrial.begin();
    noHistoryTrial.begin();
    
    pinMode(TRIGGER_PIN, INPUT);

    // Check initial trigger pin state
    if (digitalRead(TRIGGER_PIN) == HIGH) {
        Serial.println("Trigger pin is already HIGH - trigger mode disabled");
        triggerEnabled = false;
        isTriggered = true;  // Auto-trigger since we're not using trigger mode
        startTime = millis();
    } else {
        Serial.println("Trigger pin is LOW - trigger mode enabled");
        triggerEnabled = true;
    }


    startTime = millis();  // Initial start time

    // Print that Arduino is ready
    Serial.println("Arduino is ready");
}

void sendReport() {
    unsigned long currentTime = millis();
    if (currentTime - lastReportTime >= REPORT_INTERVAL) {
        if (!isTriggered) {
            // Just print b - MATLAB will ignore this since it doesn't start with <>
            Serial.println("b");

        } else {
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
            Serial.print(">");
            Serial.println();
        }
        lastReportTime = currentTime;
    }
}

void processInput(String input) {
    // Expected format: "ID,type"
    // Example: "2,h" for servo 2 with history
    // Example: "1,n" for servo 1 without history
    
    int commaIndex = input.indexOf(',');
    if (commaIndex != -1) {
        int id = input.substring(0, commaIndex).toInt();
        char type = input.charAt(commaIndex + 1);
        
        if (id >= 0 && id < 4) {
            // If not triggered yet, use first command as trigger
            if (!isTriggered) {
                isTriggered = true;
                startTime = millis();
            }
            
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

void loop() {
    
    if (triggerEnabled && !isTriggered && digitalRead(TRIGGER_PIN) == HIGH) {
        isTriggered = true;
        startTime = millis();
    }
    
    // Update motors
    servos.update();
    miniServos.update();
    stepper.update();
    
    // Update appropriate trial
    if (useHistory) {
        historyTrial.update();
    } else {
        noHistoryTrial.update();
    }
    
    // Send periodic report
    sendReport();
    
    // Process any incoming commands
    if (Serial.available()) {
        String input = Serial.readStringUntil('\n');
        processInput(input);
    }
}
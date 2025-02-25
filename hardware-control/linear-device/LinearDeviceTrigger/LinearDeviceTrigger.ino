/*
 * LinearDeviceTrigger
 * 
 * A simple sketch to trigger the linearDeviceControl system via digital pin
 * 
 * Commands:
 * - 't' or 'trigger': Sets the trigger pin HIGH for a configurable duration
 * - 's' or 'status': Reports the current trigger pin status
 * - 'h' or 'help': Shows available commands
 */

// Configuration
const int TRIGGER_PIN = 2;        // Digital pin used for triggering the main Arduino
const int TRIGGER_DURATION = 5000;   // Duration in ms to keep the trigger HIGH (0 = stay HIGH until reset)
const int LED_PIN = 13;           // Built-in LED for visual feedback

// Variables
bool isTriggered = false;
unsigned long triggerStartTime = 0;

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  
  // Configure pins
  pinMode(TRIGGER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  
  // Initial states
  digitalWrite(TRIGGER_PIN, LOW);
  digitalWrite(LED_PIN, LOW);
  
  // Print welcome message and help
  Serial.println("\nLinearDeviceTrigger Ready");
  printHelp();
}

void loop() {
  // Check if we need to reset the trigger after the duration
  if (isTriggered && TRIGGER_DURATION > 0 && (millis() - triggerStartTime >= TRIGGER_DURATION)) {
    resetTrigger();
  }
  
  // Check for serial commands
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "t" || command == "trigger") {
      activateTrigger();
    }
    else if (command == "r" || command == "reset") {
      resetTrigger();
    }
    else if (command == "s" || command == "status") {
      printStatus();
    }
    else if (command == "h" || command == "help") {
      printHelp();
    }
    else {
      Serial.println("Unknown command. Type 'h' for help.");
    }
  }
}

void activateTrigger() {
  digitalWrite(TRIGGER_PIN, HIGH);
  digitalWrite(LED_PIN, HIGH);
  triggerStartTime = millis();
  isTriggered = true;
  Serial.println("Trigger ACTIVATED");
  
  if (TRIGGER_DURATION > 0) {
    Serial.print("Trigger will automatically reset after ");
    Serial.print(TRIGGER_DURATION);
    Serial.println(" ms");
  }
}

void resetTrigger() {
  digitalWrite(TRIGGER_PIN, LOW);
  digitalWrite(LED_PIN, LOW);
  isTriggered = false;
  Serial.println("Trigger RESET");
}

void printStatus() {
  Serial.print("Trigger pin (");
  Serial.print(TRIGGER_PIN);
  Serial.print(") is ");
  Serial.println(isTriggered ? "HIGH" : "LOW");
  
  if (isTriggered && TRIGGER_DURATION > 0) {
    unsigned long elapsed = millis() - triggerStartTime;
    unsigned long remaining = (elapsed > TRIGGER_DURATION) ? 0 : (TRIGGER_DURATION - elapsed);
    Serial.print("Time remaining: ");
    Serial.print(remaining);
    Serial.println(" ms");
  }
}

void printHelp() {
  Serial.println("Available commands:");
  Serial.println("  t or trigger - Activate trigger (set pin HIGH)");
  Serial.println("  r or reset   - Reset trigger (set pin LOW)");
  Serial.println("  s or status  - Show current trigger status");
  Serial.println("  h or help    - Show this help message");
}
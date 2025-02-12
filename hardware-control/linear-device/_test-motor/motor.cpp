#include "motor.h"
#include "Arduino.h"

int steps = 0;

void init_stepper() {
    // Set the pin modes for the stepper motor control pins
    pinMode(DIR_PIN, OUTPUT);
    pinMode(STEP_PIN, OUTPUT);
    
    // Initialize pins to a known state
    digitalWrite(DIR_PIN, LOW);
    digitalWrite(STEP_PIN, LOW);
}

void move_stepper_steps(int steps_to_move) {
    // Determine direction based on whether steps is positive or negative
    bool direction = steps_to_move >= 0;
    set_direction(direction);
    
    // Convert negative steps to positive for the loop
    int absolute_steps = abs(steps_to_move);
    
    // Move the specified number of steps
    for(int i = 0; i < absolute_steps; i++) {
        // Generate one step pulse
        digitalWrite(STEP_PIN, HIGH);
        delayMicroseconds(SPEED_TIME);
        digitalWrite(STEP_PIN, LOW);
        delayMicroseconds(SPEED_TIME);
        
        // Optional: Add a small delay between steps
        delayMicroseconds(MOTOR_PERIOD);
    }
    
    // Update the global step counter
    steps = steps_to_move;
}

void set_direction(bool direction) {
    // Set the direction pin
    // Typically, HIGH = clockwise, LOW = counterclockwise
    // (This might need to be reversed depending on your motor setup)
    digitalWrite(DIR_PIN, direction ? HIGH : LOW);
    // Small delay to ensure direction is set before moving
    delayMicroseconds(MOTOR_PERIOD);
}

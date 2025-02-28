// StepperMotor.h
#ifndef STEPPER_MOTOR_H
#define STEPPER_MOTOR_H

class StepperMotor {
public:
    // Constructor: pins for step and direction control
    StepperMotor(int stepPin, int dirPin);
    
    // Main state machine update function - call this in loop()
    void update();
    
    // Command to move to absolute position
    void moveTo(long position);
    
    // Get current position
    long getCurrentPosition() const { return currentPosition; }
    
    // Check if motor is currently moving
    bool isMoving() const { return state != IDLE; }

private:
    // Motor states
    enum State {
        IDLE,
        MOVING,
        STEP_HIGH,
        STEP_LOW
    };
    
    // Pin definitions
    const int stepPin;
    const int dirPin;
    
    // Position tracking
    long currentPosition;
    long targetPosition;
    
    // Timing variables
    unsigned long lastStepTime;
    const unsigned long stepDelay = 4800; // Microseconds between steps
    const unsigned long pulseWidth = 10;  // Microseconds for step pulse
    
    // State machine variable
    State state;
    
    // Direction control
    void setDirection(bool clockwise);
};

#endif
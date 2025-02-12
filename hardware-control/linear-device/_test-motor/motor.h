#ifndef MOTOR_H
#define MOTOR_H

// Constants for motor control
constexpr int DIR_PIN = 10;
constexpr int STEP_PIN = 1;
constexpr int SPEED_TIME = 1000;
constexpr int MOTOR_PERIOD = 2;

extern int steps;

void init_stepper();  // Function declaration
void move_stepper_steps(int steps);  // Move the stepper motor a specific number of steps
void set_direction(bool direction); // Set direction: true = forward, false = backward

#endif
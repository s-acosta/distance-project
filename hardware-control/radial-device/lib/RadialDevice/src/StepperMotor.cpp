// StepperMotor.cpp
#include "StepperMotor.h"
#include "Arduino.h"

StepperMotor::StepperMotor(int stepPin, int dirPin)
    : stepPin(stepPin),
      dirPin(dirPin),
      currentPosition(0),
      targetPosition(0),
      lastStepTime(0),
      stepDelay(4800),
      state(IDLE)
{
    pinMode(stepPin, OUTPUT);
    pinMode(dirPin, OUTPUT);
    digitalWrite(stepPin, LOW);
    digitalWrite(dirPin, LOW);
}

void StepperMotor::setStepDelay(unsigned long new_stepDelay)
{
    stepDelay = new_stepDelay;
}

void StepperMotor::setDirection(bool clockwise)
{
    digitalWrite(dirPin, clockwise ? HIGH : LOW);
}

void StepperMotor::moveTo(long position)
{
    targetPosition = position;
}

void StepperMotor::update()
{
    unsigned long currentTime = micros();

    switch (state)
    {
    case IDLE:
        if (currentPosition != targetPosition)
        {
            setDirection(targetPosition > currentPosition);
            state = MOVING;
        }
        break;

    case MOVING:
        if (currentTime - lastStepTime >= stepDelay)
        {
            digitalWrite(stepPin, HIGH);
            lastStepTime = currentTime;
            state = STEP_HIGH;
        }
        break;

    case STEP_HIGH:
        if (currentTime - lastStepTime >= pulseWidth)
        {
            digitalWrite(stepPin, LOW);
            currentPosition += (targetPosition > currentPosition) ? 1 : -1;
            state = STEP_LOW;
        }
        break;

    case STEP_LOW:
        if (currentTime - lastStepTime >= stepDelay)
        {
            if (currentPosition == targetPosition)
            {
                state = IDLE;
            }
            else
            {
                state = MOVING;
            }
        }
        break;
    }
}






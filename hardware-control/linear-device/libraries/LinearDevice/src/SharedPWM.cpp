#include "SharedPWM.h"

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

void initializePWM() {
    pwm.begin();
    pwm.setOscillatorFrequency(27000000);
    pwm.setPWMFreq(50);
}
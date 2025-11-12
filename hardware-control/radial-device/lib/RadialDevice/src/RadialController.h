#ifndef RADIAL_CONTROLLER_H
#define RADIAL_CONTROLLER_H

#include <Arduino.h>

class RadialController {
public:
	RadialController(uint8_t stepPin, uint8_t dirPin);
	void begin();
	void step(int32_t steps);
	void setSpeedHz(uint32_t stepsPerSecond);
	void update();

private:
	uint8_t _stepPin;
	uint8_t _dirPin;
	uint32_t _stepsPerSecond;
	uint32_t _stepIntervalMicros;
	int32_t _remainingSteps;
	unsigned long _lastStepMicros;
};

#endif

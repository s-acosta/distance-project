#include "RadialController.h"

RadialController::RadialController(uint8_t stepPin, uint8_t dirPin)
	: _stepPin(stepPin), _dirPin(dirPin), _stepsPerSecond(200), _stepIntervalMicros(5000), _remainingSteps(0), _lastStepMicros(0) {}

void RadialController::begin() {
	pinMode(_stepPin, OUTPUT);
	pinMode(_dirPin, OUTPUT);
	setSpeedHz(_stepsPerSecond);
}

void RadialController::setSpeedHz(uint32_t stepsPerSecond) {
	_stepsPerSecond = stepsPerSecond == 0 ? 1 : stepsPerSecond;
	_stepIntervalMicros = 1000000UL / _stepsPerSecond;
}

void RadialController::step(int32_t steps) {
	digitalWrite(_dirPin, steps >= 0 ? HIGH : LOW);
	_remainingSteps = abs(steps);
}

void RadialController::update() {
	if (_remainingSteps <= 0) return;
	unsigned long now = micros();
	if (now - _lastStepMicros < _stepIntervalMicros) return;
	_lastStepMicros = now;
	digitalWrite(_stepPin, HIGH);
	// short pulse
	delayMicroseconds(2);
	digitalWrite(_stepPin, LOW);
	_remainingSteps--;
}

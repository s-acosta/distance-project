# LinearDevice Library

An Arduino library for controlling a linear device with multiple servo motors and a stepper motor. This library provides precise control and coordination of movements for research apparatus requiring accurate motor control and timing.

## Features

- Control of multiple servo motors (both standard and mini)
- Stepper motor control with precise positioning
- Trial sequencing with and without history
- Coordinated movement sequences
- Real-time status reporting
- Calibration capabilities

## Hardware Requirements

- Arduino board
- PCA9685 PWM controller
- Servo motors (standard and mini)
- Stepper motor
- Stepper motor driver

## Dependencies

- [Adafruit PWM Servo Driver Library](https://github.com/adafruit/Adafruit-PWM-Servo-Driver-Library)
- Wire Library (built into Arduino IDE)

## Installation

1. Download this repository as ZIP
2. In Arduino IDE: Sketch -> Include Library -> Add .ZIP Library
3. Select the downloaded ZIP file

Or copy the library files to your Arduino libraries folder:
- Windows: Documents\Arduino\libraries\LinearDevice
- Mac: ~/Documents/Arduino/libraries/LinearDevice
- Linux: ~/Arduino/libraries/LinearDevice

## Usage

Basic example:

```cpp
#include <LinearDevice.h>

ServoMotor servos;
MiniServoMotor miniServos;
StepperMotor stepper(11, 10);  // Step pin: 11, Dir pin: 10

void setup() {
    Serial.begin(115200);
    initializePWM();
    servos.begin();
    miniServos.begin();
}

void loop() {
    servos.update();
    miniServos.update();
    stepper.update();
}
```

Check the examples folder for more detailed usage scenarios:
- test_servo: Basic servo control
- test_miniservo: Mini servo control
- test_motor: Stepper motor control
- calibration: Motor calibration
- test_trial_history: Trial sequencing with history
- test_trial_no_history: Trial sequencing without history

## Pin Configuration

Default pin assignments:
- Stepper Motor:
  - STEP: Pin 11
  - DIR: Pin 10
- Servo Motors (via PCA9685):
  - Servo 0: Pin 0
  - Servo 1: Pin 4
  - Servo 2: Pin 8
  - Servo 3: Pin 12
- Mini Servos (via PCA9685):
  - Mini Servo 0: Pin 3
  - Mini Servo 1: Pin 7
  - Mini Servo 2: Pin 11
  - Mini Servo 3: Pin 15

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- [Your Name] - *Initial work*

## Acknowledgments

- Thanks to Adafruit for their PWM Servo Driver Library
- [Add any other acknowledgments here]
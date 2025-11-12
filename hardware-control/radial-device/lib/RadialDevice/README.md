# RadialDevice Library

An Arduino library for controlling a radial device (e.g., rotary stage) with coordinated motor control and experiment-friendly helpers. Mirrors the structure of `LinearDevice` for a consistent API.

## Features
- Simple facade header `RadialDevice.h` including public classes
- `src/` for implementation files
- `examples/` with a runnable starter sketch

## Installation
- Place this folder under your Arduino `libraries` directory or add as ZIP in the IDE.

## Usage
```cpp
#include <RadialDevice.h>

void setup() {
  Serial.begin(115200);
}

void loop() {
}
```

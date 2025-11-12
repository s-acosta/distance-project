# Radial Device - Command Reference

## Connection Settings
- **Baud Rate**: 115200
- **Line Ending**: Newline (\n)
- **Response Format**: All responses wrapped in `< >` brackets

---

## Stepper Motor Commands

### Move to Angle
```
move <angle>
```
Move to target angle using current speed setting.
- **Example**: `move -30`
- **Response**: `<Moving to -30 at 20000>`
- **Completion**: `<Movement done: -30>`

```
move <angle> <speed>
```
Move to target angle and set new speed (microseconds delay between steps).
- **Example**: `move 60 28000`
- **Response**: `<Moving to 60 at 28000>`
- **Valid Speed Range**: 5000-100000 microseconds

### Set Velocity
```
velocity <speed>
```
Set stepper motor speed without moving.
- **Example**: `velocity 15000`
- **Response**: `<Velocity set to 15000>`
- **Range**: 5000-100000 microseconds (lower = faster)

### Speed Presets
```
speed <preset>
```
Use predefined speed settings.
- **Example**: `speed 2`
- **Response**: `<Speed preset 2>`
- **Presets**:
  - `1` = Slow (50000 µs)
  - `2` = Medium (20000 µs)
  - `3` = Fast (8000 µs)

### Stop Motor
```
stop
```
Emergency stop for stepper motor.
- **Response**: `<Stopped>`

---

## Servo Motor Commands

### Raise Servo
```
raise <id>
```
Raise servo to active position using default movement time.
- **Example**: `raise 2`
- **Response**: `<Raising servo 2>`
- **Completion**: `<Servo done>`
- **Valid IDs**: 0-3

```
raise <id> <time>
```
Raise servo with custom movement time (milliseconds).
- **Example**: `raise 2 500`
- **Response**: `<Raising servo 2 in 500 ms>`

### Lower Servo
```
lower <id>
```
Lower servo to rest position using default movement time.
- **Example**: `lower 2`
- **Response**: `<Lowering servo 2>`
- **Completion**: `<Servo done>`

```
lower <id> <time>
```
Lower servo with custom movement time (milliseconds).
- **Example**: `lower 2 250`
- **Response**: `<Lowering servo 2 in 250 ms>`

### Custom Servo Position
```
servo <id> <pos>
```
Move servo to specific position using default time.
- **Example**: `servo 2 300`
- **Response**: `<Moving servo 2 to 300>`
- **Position Range**: Typically 150-600 (depends on servo config)

```
servo <id> <pos> <time>
```
Move servo to position with custom time (milliseconds).
- **Example**: `servo 2 300 1000`
- **Response**: `<Moving servo 2 to 300 in 1000 ms>`

---

## System Commands

### Status
```
status
```
Get current system status.
- **Response**: `<Status: angle=-30, target=-30, velocity=20000, servo_moving=no>`

### Reset
```
reset
```
Reset encoder counter, stepper position, and all servos to rest position.
- **Response**: `<Reset complete>`

---

## Current Configuration

### Servo Setup
| Servo ID | PWM Pin | Rest Position | Active Position |
|----------|---------|---------------|-----------------|
| 0        | 12      | 400           | 180             |
| 1        | 8       | 400           | 180             |
| 2        | 4       | 400           | 180             |
| 3        | 0       | 400           | 180             |

### Stepper Limits
- **Minimum**: -150 degrees
- **Maximum**: +230 degrees
- **Tolerance**: ±1 degree

### Default Settings
- **Stepper Speed**: 20000 microseconds
- **Servo Movement Time**: 500 milliseconds

---

## Example Usage

### Basic Experiment Sequence
```
reset                    # Initialize system
move -30                 # Move to starting position
raise 2                  # Raise servo 2
move 60                  # Move to end position
lower 2                  # Lower servo 2
move -30                 # Return to start
```

### Speed Variations
```
speed 3                  # Use fast preset
move 45                  # Quick movement

velocity 12000           # Custom speed
move -20

speed 1                  # Use slow preset
move 0                   # Precise movement
```

### Servo Timing Control
```
raise 2 1000            # Slow raise (1 second)
lower 2 200             # Fast lower (200 ms)
servo 2 300 750         # Custom position, custom time
```

---

## Notes

- All commands must end with newline character (\n)
- Wait for completion messages before sending next command
- Encoder continuously reports angle during movements
- Invalid commands are ignored with no response
- Servo movements are non-blocking (can queue next command)
- Stepper movements block until completion

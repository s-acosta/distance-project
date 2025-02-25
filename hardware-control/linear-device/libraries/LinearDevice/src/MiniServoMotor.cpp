#include "MiniServoMotor.h"
#include "SharedPWM.h"

// Helper function to create slower acceleration at start
int easeInPosition(unsigned long elapsed, unsigned long total_time, int start_pos, int end_pos) {
    // Convert elapsed time to a 0-1 progress value
    float t = static_cast<float>(elapsed) / total_time;
    
    // Apply easing - this creates a slower start (t squared)
    t = t * t;
    
    // Convert back to position
    int delta = end_pos - start_pos;
    return start_pos + (delta * t);
}

MiniServoMotor::MiniServoMotor() {
    mini_servos[0] = {.pin = 3,  .rest_pos = 230, .active_pos = 415, .current_pos = 230, .state = ServoState::IDLE};
    mini_servos[1] = {.pin = 7,  .rest_pos = 446, .active_pos = 267, .current_pos = 446, .state = ServoState::IDLE};
    mini_servos[2] = {.pin = 11, .rest_pos = 432, .active_pos = 250, .current_pos = 435, .state = ServoState::IDLE};
    mini_servos[3] = {.pin = 15, .rest_pos = 440, .active_pos = 245, .current_pos = 435, .state = ServoState::IDLE};
    active_servo = -1;
}

void MiniServoMotor::begin() {
    for(int i = 0; i < NUM_SERVOS; i++) {
        pwm.setPWM(mini_servos[i].pin, 0, mini_servos[i].rest_pos);
        mini_servos[i].current_pos = mini_servos[i].rest_pos;
        mini_servos[i].state = ServoState::IDLE;
    }
}

bool MiniServoMotor::move_to_position(int id, int position) {
    if(id < 0 || id >= NUM_SERVOS) {
        return false;
    }


    if (position < 150 || position > 450) {
        return false;
    }

    pwm.setPWM(mini_servos[id].pin, 0, position);
    mini_servos[id].current_pos = position;
    return true;
}

bool MiniServoMotor::activate_servo(int id) {
    if(id < 0 || id >= NUM_SERVOS) {
        Serial.println("Invalid servo ID");
        return false;
    }
    
    // Check if any servo is active
    for(int i = 0; i < NUM_SERVOS; i++) {
        if(i != id && mini_servos[i].state != ServoState::IDLE) {
            Serial.println("Another servo is already active");
            return false;
        }
    }
    
    mini_servos[id].start_time = millis();
    mini_servos[id].state = ServoState::MOVING_TO_ACTIVE;
    active_servo = id;
    return true;
}

void MiniServoMotor::update() {
    if (active_servo == -1) return;
    
    unsigned long elapsed = millis() - mini_servos[active_servo].start_time;
    
    switch(mini_servos[active_servo].state) {
        case ServoState::MOVING_TO_ACTIVE:
            if (elapsed >= MOVEMENT_TIME) {
                mini_servos[active_servo].current_pos = mini_servos[active_servo].active_pos;
                mini_servos[active_servo].state = ServoState::HOLDING_ACTIVE;
                mini_servos[active_servo].start_time = millis();
            } else {
                // Use easing function for smoother start
                mini_servos[active_servo].current_pos = easeInPosition(
                    elapsed, MOVEMENT_TIME,
                    mini_servos[active_servo].rest_pos,
                    mini_servos[active_servo].active_pos
                );
            }
            break;
            
        case ServoState::HOLDING_ACTIVE:
            if (elapsed >= HOLD_TIME) {
                mini_servos[active_servo].state = ServoState::MOVING_TO_REST;
                mini_servos[active_servo].start_time = millis();
            }
            break;
            
        case ServoState::MOVING_TO_REST:
            if (elapsed >= MOVEMENT_TIME) {
                mini_servos[active_servo].current_pos = mini_servos[active_servo].rest_pos;
                mini_servos[active_servo].state = ServoState::IDLE;
                active_servo = -1;
            } else {
                mini_servos[active_servo].current_pos = easeInPosition(
                    elapsed, MOVEMENT_TIME,
                    mini_servos[active_servo].active_pos,
                    mini_servos[active_servo].rest_pos
                );
            }
            break;
            
        default:
            break;
    }
    
    if (active_servo != -1) {
        pwm.setPWM(mini_servos[active_servo].pin, 0, mini_servos[active_servo].current_pos);
    }
}

bool MiniServoMotor::is_moving() {
    return active_servo != -1;
}
#include "ServoMotor.h"
#include "SharedPWM.h"


ServoMotor::ServoMotor() {
    servos[0] = {.pin = 0,  .rest_pos = 170, .active_pos = 435, .current_pos = 170, .target_pos = 170};
    servos[1] = {.pin = 4,  .rest_pos = 416, .active_pos = 162, .current_pos = 416, .target_pos = 416};
    servos[2] = {.pin = 8,  .rest_pos = 415, .active_pos = 184, .current_pos = 418, .target_pos = 418};
    servos[3] = {.pin = 12, .rest_pos = 180, .active_pos = 428, .current_pos = 180, .target_pos = 180};
    active_servo = -1;
}

void ServoMotor::begin() {
    
    for(int i = 0; i < NUM_SERVOS; i++) {
        pwm.setPWM(servos[i].pin, 0, servos[i].rest_pos);
        servos[i].current_pos = servos[i].rest_pos;
        servos[i].target_pos = servos[i].rest_pos;
    }
}

bool ServoMotor::move_servo(int id, int pos) {
    
    if(id < 0 || id >= NUM_SERVOS) {
        Serial.println("Invalid servo ID");
        return false;
    }
    
    // Check if any servo is in active position
    bool another_active = false;
    for(int i = 0; i < NUM_SERVOS; i++) {
        if(i != id && servos[i].current_pos == servos[i].active_pos) {
            another_active = true;
            break;
        }
    }
    
    // If moving to active and another servo is active, prevent movement
    if(pos == servos[id].active_pos && another_active) {
        Serial.println("Another servo is already in active position");
        return false;
    }
    
    int min_pos = min(servos[id].rest_pos, servos[id].active_pos);
    int max_pos = max(servos[id].rest_pos, servos[id].active_pos);
    
    if(pos < min_pos || pos > max_pos) {
        Serial.println("Position out of bounds");
        return false;
    }
    
    servos[id].target_pos = pos;
    servos[id].start_time = millis();
    servos[id].start_pos = servos[id].current_pos;
    active_servo = id;
    
    return true;
}

void ServoMotor::update() {

    if (active_servo == -1) return;
    if(servos[active_servo].current_pos == servos[active_servo].target_pos) {
        active_servo = -1; 
        return;
        }

    unsigned long elapsed = millis() - servos[active_servo].start_time;

    if (elapsed > MOVEMENT_TIME) {
        elapsed = MOVEMENT_TIME;
    }
    
    servos[active_servo].current_pos = map(elapsed, 0, MOVEMENT_TIME,
        servos[active_servo].start_pos, servos[active_servo].target_pos);

    pwm.setPWM(servos[active_servo].pin, 0, servos[active_servo].current_pos);
    
}

bool ServoMotor::is_moving() {
    return active_servo != -1;
}
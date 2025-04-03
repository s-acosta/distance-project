#ifndef SERVO_MOTOR_H
#define SERVO_MOTOR_H

#include <Adafruit_PWMServoDriver.h>

struct Servo {
    int pin;
    int rest_pos;
    int active_pos;
    int current_pos;
    int target_pos;
    unsigned long start_time;
    int start_pos;
};

class ServoMotor {
    public:
        ServoMotor();
        Servo servos[4];
        void begin();
        bool move_servo(int id, int pos);
        void update();
        bool is_moving();
        int NUM_SERVOS = 4;
        unsigned MOVEMENT_TIME = 100;
        
    private:
        unsigned long last_update;
        int active_servo;
};

#endif
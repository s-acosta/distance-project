#ifndef MINI_SERVO_MOTOR_H
#define MINI_SERVO_MOTOR_H

enum class ServoState {
    IDLE,
    MOVING_TO_ACTIVE,
    HOLDING_ACTIVE,
    MOVING_TO_REST
};

struct MiniServo {
    int pin;
    int rest_pos;
    int active_pos;
    int current_pos;
    ServoState state;
    unsigned long start_time;
    float exact_pos;  // Added to store precise position
};

class MiniServoMotor {
    public:
        MiniServoMotor();
        void begin();
        bool move_to_position(int id, int position);
        bool activate_servo(int id);
        void update();
        bool is_moving();
        
        // New method to check if servo is moving to active position
        bool is_moving_to_active();
        
        static const int NUM_SERVOS = 4;
        MiniServo mini_servos[NUM_SERVOS];
        
    private:
        static const unsigned long MOVEMENT_TO_ACTIVE_TIME = 3000;  // 2 seconds to move to active position
        static const unsigned long MOVEMENT_TO_REST_TIME = 100;     // 0.5 seconds to return to rest position
        static const unsigned long HOLD_TIME = 100;                 // 100ms hold time
        
        int active_servo;
        unsigned long last_update_time;  // Track last update time for dt calculation
};

#endif
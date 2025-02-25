//test_servo.txt
// Moving servo to active and rest positions

#include "ServoMotor.h"
#include "SharedPWM.h"

ServoMotor servos;

void setup() {

    Serial.begin(115200);
    initializePWM();
    servos.begin();

    Serial.println("Servo Rest/Active Position Control");
    Serial.println("Commands: r[id] - Move to rest position"); 
    Serial.println("         a[id] - Move to active position");
    Serial.println("Example: r0 moves servo 0 to rest position");

}

void loop() {
    servos.update();
    
    if(Serial.available() >= 2) {
        unsigned long start = millis();
        char cmd = Serial.read();
        char id = Serial.read() - '0'; // Convert char to int
        
        if((cmd == 'r' || cmd == 'a') && id >= 0 && id < 4) {
            servos.move_servo(id, cmd == 'r' ? servos.servos[id].rest_pos : servos.servos[id].active_pos);
        }
    }
 }
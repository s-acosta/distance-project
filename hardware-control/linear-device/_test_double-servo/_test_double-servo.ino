// zoomkat 12-13-11 serial servo (2) test
// for writeMicroseconds, use a value like 1500
// for IDE 1.0
// Powering a servo from the arduino usually DOES NOT WORK.
// two servo setup with two servo commands
// send eight character string like 15001500 or 14501550
// use serial monitor to test
#include <Servo.h>

#define servo_num 5
Servo servo;
Servo micro_servo;
int startAngle;
int stopAngle;
String readString, pos_string, id_string;
int servo_pin = 2;
int micro_servo_pin = 3;

unsigned long experiment_start_time;

bool new_order = false;

int active_pos = 600;
int rest_pos = 1800;
int pos;
// int card_pos[servo_num] = {33, 51, 41, 58, 120, 60 };

void setup() {
  Serial.begin(9600);
  servo.attach(servo_pin);
  micro_servo.attach(micro_servo_pin);
  servo.writeMicroseconds(rest_pos);

  // delay(2000);

  // for (pos = 1800; pos >= 600; pos -= 1) {  // goes from 0 degrees to 180 degrees
  //   // in steps of 1 degree
  //   servo.writeMicroseconds(pos);  // tell servo to go to position in variable 'pos'
  //   delay(5);                      // waits 15ms for the servo to reach the position
  // }
}

void loop() {

  // for (pos = 0; pos <= 180; pos += 1) {  // goes from 0 degrees to 180 degrees
  //   // in steps of 1 degree
  //   micro_servo.write(pos);  // tell servo to go to position in variable 'pos'
  //   delay(15);               // waits 15ms for the servo to reach the position
  // }

  // for (pos = 180; pos >= 0; pos -= 1) {  // goes from 180 degrees to 0 degrees
  //   micro_servo.write(pos);              // tell servo to go to position in variable 'pos'
  //   delay(15);                           // waits 15ms for the servo to reach the position
  // }
}
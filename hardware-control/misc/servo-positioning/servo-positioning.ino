// zoomkat 12-13-11 serial servo (2) test
// for writeMicroseconds, use a value like 1500
// for IDE 1.0
// Powering a servo from the arduino usually DOES NOT WORK.
// two servo setup with two servo commands
// send eight character string like 15001500 or 14501550
// use serial monitor to test

#include <Servo.h> 
 
#define servo_num 5
Servo servo[servo_num];
int startAngle ;
int stopAngle;
String readString, pos_string, id_string;
int servo_pin[servo_num] = {2, 3, 4, 5};
int active_pos[servo_num] = {980, 985, 980, 980, 980};
int rest_pos[servo_num] = { 170, 180, 170, 170, 170 };
// int card_pos[servo_num] = {33, 51, 41, 58, 120, 60 };

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < servo_num; i++) {
    servo[i].attach(servo_pin[i]);
  }

  servo[0].writeMicroseconds(500);
  servo[1].writeMicroseconds(600);
  servo[2].writeMicroseconds(2500);
  servo[3].writeMicroseconds(620);
}

void loop() {
  
  while (Serial.available()) {
    delay(5);  //delay to allow buffer to fill
    if (Serial.available() > 0) {
      char c = Serial.read();  //gets one byte from serial buffer
      readString += c;         //makes the string readString
    }
  }

  if (readString.length() > 0) {

    // expect a string like 07002100 containing the two servo positions
    id_string = readString.substring(0, 1);   //get the first four characters
    pos_string = readString.substring(1, 5);  //get the next four characters

    Serial.println(readString);

    int id = id_string.toInt();
    int pos = pos_string.toInt();
    id = id - 1;

    servo[id].writeMicroseconds(pos); //set servo position

    readString = "";

  }

  }
  

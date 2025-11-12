// For Serial Communication
const byte numChars = 32;
char receivedChars[numChars];  // an array to store the received data
char receivedChars_1[numChars];
char tempChars[numChars];  // temporary array for use when parsing
float angle;
boolean newData = false;

// variables to hold the parsed data
char messageFromPC[numChars] = { 0 };
int stimulus_number;
int counter_goal;

boolean newOrder = false;
boolean distanceUp = false;
boolean abanicoUp = false;
boolean direction = false;
boolean reached90 = false;

// Stepper motor set-up
int step_pin = 12;
int dir_pin = 13;
int speed_time = 1000;

int counter;  // new for this version
int previous_counter;
int order;
int dataNumber = 0;
unsigned long start_time;
unsigned long order_time;
unsigned long last_ms;
unsigned long moving_start_time;
unsigned long moving_end_time;
unsigned long experiment_start_time;

#define SLOW_MOTOR_PERIOD 16
#define FAST_MOTOR_PERIOD 5

// Servo Motor Set-up
#include <Servo.h>
#define servo_num 5
Servo servo[servo_num];
int servo_pin[servo_num] = {2, 3, 4, 5, 6};
int active_pos[servo_num] = {1000, 1000, 1000, 1000, 1000};
int rest_pos[servo_num] = {2200, 2200, 2200, 2200};

int servo_step = 5;

void setup() {

  // Stepper Motor Set-Up
  pinMode(dir_pin, OUTPUT);
  pinMode(step_pin, OUTPUT);
  digitalWrite(step_pin, LOW);
  digitalWrite(dir_pin, HIGH);

  // Serial set-up
  Serial.begin(115200);
  Serial1.begin(115200);
  Serial.print("<Arduino is ready>");
  Serial.print('\n');
  last_ms = millis();

  // Servo set-up
  for (int i = 0; i < servo_num; i++) {
    servo[i].attach(servo_pin[i]);
    servo[i].writeMicroseconds(rest_pos[i]);
  }
}

void loop() {

  readMatlab();
  readCounter();

  // Order received
  if (newOrder == true) {

    if (distanceUp == false) {
      experiment_start_time = order_time + 500;
    }

    // Servo goes up
    if (reached90 == false && millis() > experiment_start_time && distanceUp == false) {

      unsigned long now = millis()-experiment_start_time;
      unsigned long rise_duration = 2000;
      unsigned long time_into_cycle = now % rise_duration;
      int angle;

      for (int i = 0; i < servo_num; i++){
        if (i != stimulus_number) {
          servo[i].write(rest_pos[i]);
        } 
        else {
            if (time_into_cycle < 1950){
              angle = map(time_into_cycle, 0, 1950, rest_pos[i], active_pos[i]);
              servo[i].write(angle);
            }
            else{
              distanceUp = true;
              moving_start_time = millis();
            }
            }
        }
      }
    }

  // Movement starts (to 90)
  if (reached90 == false && distanceUp == true && millis() > moving_start_time + 50) {

    digitalWrite(dir_pin, HIGH);

    if (millis() - last_ms > SLOW_MOTOR_PERIOD) {
      moveSteps(1);
      last_ms = millis();
    }

    if (abs(counter - 160) < 1) {
      moving_end_time = millis();
      for (int i = 0; i < servo_num; i++) {
        servo[i].write(rest_pos[i]);
      }
      distanceUp = false;
      reached90 = true;
      Serial.print("<Movement Done>");
      Serial.print('\n');
    }
  }

  // Arm comes back (to 0)
  if (reached90 == true && distanceUp == false && millis() > moving_end_time + 100) {

    digitalWrite(dir_pin, LOW);

    if (millis() - last_ms > FAST_MOTOR_PERIOD) {
      moveSteps(1);
      last_ms = millis();
    }

    if (counter < (-80)) {
      moving_end_time = millis();
      newOrder = false;
      reached90 = false;
      Serial.print("<Movement Done>");
      Serial.print('\n');
    }
  }


}
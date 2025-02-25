// Goal of the phase 3:
// 1. Stage 1: Device receives order
// 2. Stage 2: Stepper goes to the end of the range
// 3. Stage 3: Stepper comes back
// 4. Stage 4: It stops at target position for 5 seconds
// 5. Stage 5: It continues to initial position

// Experiment timing variables 
int rise_duration = 1000;
int still_duration = 500;
int off_time = 2000;
int spin_duration = 6000;

// 1. Stepper Motor variables
int dir_pin = 12;
int step_pin = 11;
int target_servo;
int steps;
int current_steps;
int current_position = 0;
#define SPEED_TIME 1000      // this is in microsends, between steps
#define SLOW_MOTOR_PERIOD 2  // this is miliseconds
#define FAST_MOTOR_PERIOD 2

// Boolean variables for the linear movement & servo raising
boolean is_stage_1 = false;  // True when has a new order
boolean is_stage_2 = false;  // True when distance is raised
boolean is_stage_3 = false;  // True when it has gone to the end
boolean is_stage_4 = false;  // True when it has stopped at desired position
boolean is_stage_5 = false;  // True when it has stopped moving the mini servo
boolean is_stage_6 = false;   // True when it has come back to 0
boolean is_stage_7 = false; 

boolean reached_target = false;
boolean new_order = false;
boolean forward_movement;

// 2. Time-distance reporter variables
#define PERIOD 10  // Period in ms
unsigned long experiment_start_time;
unsigned long order_start_time;
unsigned long current_time;
unsigned long last_ms;
unsigned long start_time;
int i = 0;
long int matlab_last_ms;
char start_marker = '<';
char end_marker = '>';
char time_marker = 't';
char counter_marker = 'c';
boolean send_data = false;
int data_number = 0;
#define MATLAB_PERIOD 50

// 3. Trigger variables
int trigger_pin = 3;
boolean is_triggered = false;

// 4. Servo variables
#include <Servo.h>
#define servo_num 2
Servo servo[servo_num];
Servo micro_servo[servo_num];
int servo_pin[servo_num] = {2,4};
int microservo_pin[servo_num] = {3,5};
int servo_active_pos[servo_num] = {600, 600};
int servo_rest_pos[servo_num] = {1800, 1600};
int microservo_active_pos[servo_num] = {500, 600};

// 5. Motor variables
int start_position[servo_num] = {1250, 2000};
int end_position[servo_num] = {2950, 300};
int middle_position = 1500;

int startAngle;
int stopAngle;

int pos;

// 5. Matlab communication variables
String read_string, distance_string, id_string;
const byte num_chars = 32;
char received_chars[num_chars];  // an array to store the received data
char received_chars_1[num_chars];
char temp_chars[num_chars];  // temporary array for use when parsing
boolean new_data = false;

char message_from_PC[num_chars] = { 0 };
int stimulus_number;
int counter_goal;

void setup() {

  Serial.begin(115200);

  for (int i = 0; i < servo_num; i++) {
      servo[i].attach(servo_pin[i]);
      micro_servo[i].attach(microservo_pin[i]);
    }

  for (int i = 0; i < servo_num; i++) {
      servo[i].writeMicroseconds(servo_rest_pos[i]);
      micro_servo[i].writeMicroseconds(microservo_active_pos[i]);
    }

  last_ms = millis();
  current_position = 0;
  is_triggered = false;
  matlab_last_ms = millis();

  pinMode(dir_pin, OUTPUT);
  pinMode(step_pin, OUTPUT);
  digitalWrite(dir_pin, LOW);

  // setDirection(500);
  // boolean is_in_position = false;
  // while (is_in_position == false){
  //   is_in_position = moveToPosition(500);
  // }
  
  Serial.print("<Arduino is ready>");
  Serial.print('\n');

  start_time = millis();
}

void loop() {

  readMatlab();

  // Report time and position
  current_time = millis();
  if (current_time - matlab_last_ms > MATLAB_PERIOD && is_triggered == true) {
    reportCounter();
    matlab_last_ms = millis();
  }

  if (new_order == true && is_stage_1 == false) {

    if (is_triggered == false) {
      experiment_start_time = millis();
      is_triggered = true;
    }

    start_time = millis();
    is_stage_1 = true;
    setDirection(int(start_position[target_servo]/3));

  }

  // Move device to start of trial
  if (is_stage_1 == true && is_stage_2 == false) {
    is_stage_2 = moveToPosition(int(start_position[target_servo]/3));

    if (is_stage_2 == true) {
      start_time = millis();
    }
  }

  // Move big servo up
  if (is_stage_2 == true && is_stage_3 == false) {

    unsigned long now = millis() - start_time;
    unsigned long time_into_cycle = now % rise_duration;
    int angle;

    if (time_into_cycle < rise_duration - 50) {
      angle = map(time_into_cycle, 0, rise_duration - 50, servo_rest_pos[target_servo], servo_active_pos[target_servo]);
      servo[target_servo].writeMicroseconds(angle);
    } else {
      setDirection(int(end_position[target_servo]/3));
      start_time = millis();
      is_stage_3 = true;
    }
  }

  // Move to end position (other servo is good at this position)
    if (is_stage_3 == true && is_stage_4 == false && millis() > start_time + still_duration ) {
      is_stage_4 = moveToPosition(int(end_position[target_servo]/3));

      if (is_stage_4 == true) {
        start_time = millis();
        setDirection(int(start_position[target_servo]/3));
      }
    }

  // Move to start of trial (actual testing position)
    if (is_stage_4 == true && is_stage_5 == false && millis() > start_time + still_duration ) {
      is_stage_5 = moveToPosition(int(start_position[target_servo]/3));

      if (is_stage_5 == true) {
        start_time = millis();
      }
    }

  // Move microservo
  if (is_stage_5 == true && is_stage_6 == false && millis() > start_time + off_time) {

    unsigned long now = millis() - start_time - off_time;
    unsigned long time_into_cycle = now % 6000;
    int angle;

    if (time_into_cycle < 2950) {
      angle = map(time_into_cycle, 0, 2950, 500, 2500);
      micro_servo[target_servo].writeMicroseconds(angle);
    }
    else if (time_into_cycle < 5950) {
      angle = map(time_into_cycle, 2950, 5950, 2500, microservo_active_pos[i]);
      micro_servo[target_servo].writeMicroseconds(angle);
    }
    
    else {
      is_stage_6 = true;
      start_time = millis();
      setDirection(int(middle_position/3));
    }
  }

  // bring servo down and go back to middle point
  if (is_stage_6 == true && is_stage_7 == false && millis() > start_time + still_duration) {

    servo[target_servo].writeMicroseconds(servo_rest_pos[target_servo]);
    is_stage_7 = moveToPosition(int(middle_position/3));

    if (is_stage_7 == true) {
      is_stage_1 = false;
      is_stage_2 = false;
      is_stage_3 = false;
      is_stage_4 = false;
      is_stage_5 = false;
      is_stage_6 = false;
      is_stage_7 = false;

      new_order = false;
      Serial.print("<Movement Done>");
      Serial.print('\n');
    }
  }
}
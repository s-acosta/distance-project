// Goal of the phase 2:
// 1. Stage 1: Device receives order
// 2. Stage 2: Stepper goes to the end of the range
// 3. Stage 3: Stepper comes back
// 4. Stage 4: It stops at target position for 5 seconds
// 5. Stage 5: It continues to initial position

// 1. Stepper Motor variables
int dir_pin = 13;
int step_pin = 12;
int target_position;
int steps;
int current_steps;
int current_position = 0;

#define SPEED_TIME 1000      // this is in microsends, between steps
#define SLOW_MOTOR_PERIOD 7  // this is miliseconds
#define FAST_MOTOR_PERIOD 5

String read_string, distance_string, id_string;

boolean is_stage_1 = false;  // True when has a new order
boolean is_stage_2 = false;  // True when has gone to the end
boolean is_stage_3 = false;  // True when has stopped at desired position
boolean is_stage_4 = false;  // True when has come back to initial position

boolean reached_target = false;
boolean new_order = false;
boolean is_moving = false;
boolean is_distance_up = false;
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

// 5. Matlab communication variables
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

  last_ms = millis();
  current_position = 0;
  is_triggered = false;
  matlab_last_ms = millis();

  pinMode(dir_pin, OUTPUT);
  pinMode(step_pin, OUTPUT);
  digitalWrite(dir_pin, LOW);

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

    start_time = millis();

      if (is_triggered == false) {
      experiment_start_time = millis();
      is_triggered = true;
    }
    setDirection(700);
    is_stage_1 = true;
  }

  if (is_stage_1 == true && is_stage_2 == false && millis() > start_time + 100) {
    is_stage_2 = moveToPosition(700);

    if (is_stage_2 == true) {
      start_time = millis();
      setDirection(target_position);
    }
  }

  if (is_stage_2 == true && is_stage_3 == false && millis() > start_time + 100) {
    is_stage_3 = moveToPosition(target_position);

    if (is_stage_3 == true){
      start_time = millis();
      setDirection(0);
    }
  }

  if (is_stage_3 == true && is_stage_4 == false && millis() > start_time + 5000){
    is_stage_4 = moveToPosition(0);

    if (is_stage_4 == true) {
      is_stage_1 = false;
      is_stage_2 = false;
      is_stage_3 = false;
      is_stage_4 = false;

      new_order = false;
      Serial.print("<Movement Done>");
      Serial.print('\n');
    }


  }

}
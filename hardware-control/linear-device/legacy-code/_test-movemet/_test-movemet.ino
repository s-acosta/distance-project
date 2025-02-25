// 1. Stepper Motor variables
int dir_pin = 10;
int step_pin = 11;

int target_position;
int steps;
int current_steps;
int current_position=0;

#define SPEED_TIME 1000 // this is in microsends, between steps
#define SLOW_MOTOR_PERIOD 2 // this is miliseconds
#define FAST_MOTOR_PERIOD 5

String read_string, distance_string, id_string;

boolean reached_target = false;
boolean new_order = false;
boolean is_moving = false;
boolean is_distance_up = false;
boolean forward_movement;

// 2. Time-distance reporter variables
#define PERIOD 10  // Period in ms
unsigned long start_time;
unsigned long current_time;
unsigned long last_ms;
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
boolean is_triggered=false;

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
    if (current_time-matlab_last_ms > MATLAB_PERIOD){
      reportCounter();
      matlab_last_ms = millis();
    }

  if (new_order == true) {

    if (is_triggered == false){
      start_time = millis();
      is_triggered = true;
    }

    if (is_moving == false && millis()>(start_time + 50)) {

      int delta_position = target_position - current_position;

      if (delta_position < 0) {
        forward_movement = false;
        digitalWrite(dir_pin, LOW);
      } 
      else if (delta_position >= 0) {
        forward_movement = true;
        digitalWrite(dir_pin, HIGH);
      }
      
      is_moving = true;
    }

    if (current_position != target_position && is_moving == true){

      if (millis()-last_ms > SLOW_MOTOR_PERIOD) { 
        moveSteps(1);
        
        last_ms = millis();

        if (forward_movement == true){
          current_position += 1;
        }
        else if (forward_movement == false){
          current_position -= 1;
        }
      }
    }

    else if (current_position == target_position){
      is_moving = false;
      new_order = false;
      Serial.print("<Movement Done>");
      Serial.print('\n');
    }

  }
}
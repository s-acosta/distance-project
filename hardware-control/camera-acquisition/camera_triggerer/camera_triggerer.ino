int trigger_pin = 13;
int led_pin = 2;
unsigned long start_time;
unsigned long camera_time;
int frame_time = 100; // ms

// the setup function runs once when you press reset or power the board
void setup() {
  // initialize digital pin LED_BUILTIN as an output.
  pinMode(trigger_pin, INPUT);
  pinMode(led_pin, OUTPUT);
  
  digitalWrite(led_pin, LOW);

  Serial.begin(115200);
  // Serial.print("<Arduino is ready>");
  // Serial.print('\n');

  while (digitalRead(trigger_pin) == LOW){
    Serial.print("0");
    Serial.print('\n');
  }

  digitalWrite(led_pin, HIGH);

  start_time = millis();

}

// the loop function runs over and over again forever
void loop() {

  if (millis()>camera_time + 100){
    camera_time = millis();
    Serial.print("1");
    Serial.print('\n');
  }
  else{
    Serial.print("0");
    Serial.print('\n');
  }

}

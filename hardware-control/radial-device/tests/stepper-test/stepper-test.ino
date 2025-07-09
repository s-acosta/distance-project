int dir_pin = 13;
int step_pin = 12;
int steps = 217;
int counter = 0;
int speed_time = 11500;

void setup() {

  Serial.begin(9600);
  pinMode(dir_pin, OUTPUT);
  pinMode(step_pin, OUTPUT);
  digitalWrite(dir_pin, LOW);
}

void loop() {

  while (Serial.available() == 0) {
  }

  if (Serial.read() > 0) {

    digitalWrite(dir_pin, HIGH);

    for (int i = 0; i < abs(steps); i++) {
      digitalWrite(step_pin, HIGH);
      delayMicroseconds(speed_time);
      digitalWrite(step_pin, LOW);
      delayMicroseconds(speed_time);
    }

    delay(1000);

    digitalWrite(dir_pin, LOW);

    for (int i = 0; i < abs(steps); i++) {
      digitalWrite(step_pin, HIGH);
      delayMicroseconds(speed_time);
      digitalWrite(step_pin, LOW);
      delayMicroseconds(speed_time);
    }

  }

}

void moveSteps(int num_steps){

  for (int i = 0; i < num_steps ; i++) {
  digitalWrite(step_pin, HIGH);
  delayMicroseconds(speed_time);
  digitalWrite(step_pin, LOW);
  delayMicroseconds(speed_time);
  }

}
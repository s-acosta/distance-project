void setDirection(int desired_position) {

  int delta_position = desired_position - current_position;

  if (delta_position < 0) {
    forward_movement = false;
    digitalWrite(dir_pin, LOW);
  } else if (delta_position >= 0) {
    forward_movement = true;
    digitalWrite(dir_pin, HIGH);
  }
}
void reportCounter() {
  Serial.print(counter_marker);
  Serial.print(current_position);
  Serial.print(time_marker);
  int long now_time = millis()-start_time;
  Serial.print(now_time);
  Serial.print('\n');
}
void reportTime() {
  Serial.print(time_marker);
  int long now_time = millis()-start_time;
  // Serial.print(now_time/1000);
  // Serial.print('.');
  // Serial.print(now_time%1000);
  Serial.print(now_time);
  Serial.print('\n');
}
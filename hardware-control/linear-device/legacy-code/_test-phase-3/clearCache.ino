void clearCache() {
  while (Serial.available() > 0) {
    Serial.read();
  }
}
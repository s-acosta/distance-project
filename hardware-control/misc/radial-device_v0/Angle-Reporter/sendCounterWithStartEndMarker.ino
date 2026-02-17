void sendCounterWithStartEndMarker() {
  mySerial.print(startMarker);
  mySerial.print(counter);
  mySerial.print(endMarker);
}
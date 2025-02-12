void receiveFromMatlab() {
  static boolean recv_in_progress = false;
  static byte ndx = 0;
  char start_marker = '<';
  char end_marker = '>';
  char rc;

  while (Serial.available() > 0 && new_data == false) {
    rc = Serial.read();

    if (recv_in_progress == true) {
      if (rc != end_marker) {
        received_chars_1[ndx] = rc;
        ndx++;
        if (ndx >= num_chars) {
          ndx = num_chars - 1;
        }
      } else {
        received_chars_1[ndx] = '\0';  // terminate the string
        recv_in_progress = false;
        ndx = 0;
        new_data = true;
      }
    }

    else if (rc == start_marker) {
      recv_in_progress = true;
    }

    else {
      while (Serial.available() > 0) {
        Serial.read();
      }
    }
  }
}
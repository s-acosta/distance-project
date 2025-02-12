void readMatlab() {

  receiveFromMatlab();
  
    if (new_data == true) {

      target_servo = atoi(received_chars_1);
      target_servo = target_servo - 1;

      // id_string = readString.substring(0, 1);   //get the first four characters
      // pos_string = readString.substring(1, 5);  //get the next four characters
        
      new_data = false;
      new_order = true;

    }
}
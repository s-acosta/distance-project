void readMatlab() {

  receiveFromMatlab();
  
    if (newData == true) {

      stimulus_number = atoi(receivedChars_1);

      if (stimulus_number == 0){
      stimulus_number = 5;
      }

      stimulus_number = stimulus_number - 1;
        
      newData = false;
      order_time = millis();
      newOrder = true;

    }
}
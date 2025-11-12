void parseData() {      // split the data into its parts

    char * strtokIndx; // this is used by strtok() as an index

    strtokIndx = strtok(tempChars,",");      // get the first part - the string
    strcpy(messageFromPC, strtokIndx); // copy it to messageFromPC
    stimulus_number = atoi(messageFromPC);

    if (stimulus_number == 0){
      stimulus_number = 5;
    }

    stimulus_number = stimulus_number - 1;
 
    strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
    counter_goal = atoi(strtokIndx);     // convert this part to an integer

}
void readCounter() {

  receiveFromReporter();

    if (newData == true) {
        counter = atoi(receivedChars);
        newData = false;
    }
}
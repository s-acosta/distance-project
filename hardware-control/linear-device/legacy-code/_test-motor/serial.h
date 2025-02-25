#ifndef SERIAL_COMMAND_H
#define SERIAL_COMMAND_H

class SerialCommand{
  public:
    SerialCommand(long baudRate = 9600);
    void begin();
    void update();
    bool hasNewSteps();
    long getSteps();
    
  private:
    static const int BUFFER_SIZE = 32;
    char buffer[BUFFER_SIZE];
    int bufferIndex;
    bool newCommand;
    long steps;
    
    void processBuffer();
    void clearBuffer();
};

#endif
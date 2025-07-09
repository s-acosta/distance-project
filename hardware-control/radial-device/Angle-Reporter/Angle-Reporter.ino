#include <SoftwareSerial.h>
#define rxPin 2
#define txPin 3
#define ledPin 11
#define triggerPin 13

const byte numChars = 32;
char receivedChars[numChars];   // an array to store the received data
char startMarker = '<';
char endMarker = '>';
char timeMarker = 't';
char counterMarker = 'c';
boolean sendData = false;
boolean newData = false;
boolean isTriggered = false;
int dataNumber = 0; 

// Optical encoder setup
#define outputA 5
#define outputB 4
int counter = 0;
int aState;
int aLastState;
#define MATLAB_PERIOD 100  // Period in ms
#define MOTOR_PERIOD 10
long int matlab_last_ms;
long int motor_last_ms;
long int start_time;
long int current_time;

// Set up a new SoftwareSerial object
SoftwareSerial mySerial =  SoftwareSerial(rxPin, txPin);

void setup()  {
    // Define pin modes for TX and RX
    pinMode(rxPin, INPUT);
    pinMode(txPin, OUTPUT);
    pinMode(ledPin, OUTPUT);
    pinMode(triggerPin, INPUT);
    digitalWrite(ledPin, LOW);
    
    pinMode(outputA, INPUT);
    pinMode(outputB, INPUT);
    aLastState = digitalRead(outputA);
    readAngle();
    matlab_last_ms = millis();
    motor_last_ms = millis();
    
    // Set the baud rate for the SoftwareSerial object
    mySerial.begin(115200);
    Serial.begin(115200);

    Serial.print("<Arduino is ready>");
    Serial.print('\n');
    start_time = millis();
}

void loop() {

    readAngle();

    if (isTriggered == false && digitalRead(triggerPin) == HIGH){
      start_time = millis();
      digitalWrite(ledPin, HIGH);
      isTriggered = true;
    }

    current_time = millis();
    if (current_time-matlab_last_ms > MATLAB_PERIOD){
      reportCounter();
      reportTime();
      // clearCache();
      matlab_last_ms = millis();
    }

    current_time = millis();
    if (current_time-motor_last_ms > MOTOR_PERIOD){
      sendCounterWithStartEndMarker();
      // clearCache();
      motor_last_ms = millis();
    }

}
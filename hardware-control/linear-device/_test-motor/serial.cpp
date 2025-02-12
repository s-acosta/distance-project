#include "serial.h"
#include "Arduino.h"


SerialManager::SerialManager() 
    : serialBuffer{0}, receivedData(""), dataAvailable(false) {}

void SerialManager::init() {
    Serial.begin(SERIAL_BAUD_RATE);
    // Optional: Add any initialization message
    Serial.println("Serial Manager Initialized");
}

bool SerialManager::hasData() {
    readSerialData();
    return dataAvailable;
}

String SerialManager::getData() {
    return receivedData;
}

void SerialManager::clearData() {
    receivedData = "";
    dataAvailable = false;
}

void SerialManager::readSerialData() {
    // Check if data is available
    if (Serial.available() > 0) {
        // Read the incoming bytes
        int bytesRead = Serial.readBytesUntil('\n', serialBuffer, BUFFER_SIZE - 1);
        
        // Null-terminate the buffer
        serialBuffer[bytesRead] = '\0';
        
        // Convert to String and process
        receivedData = String(serialBuffer);
        
        // Trim whitespace
        receivedData.trim();
        
        // Mark data as available
        if (!receivedData.isEmpty()) {
            dataAvailable = true;
            processSerialData();
        }
    }
}

void SerialManager::processSerialData() {
    // Basic processing - can be customized based on your specific needs
    // For example, you might want to validate the format or parse specific commands
    
    // For now, just a simple print for debugging
    Serial.print("Received: ");
    Serial.println(receivedData);
}

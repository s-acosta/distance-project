// Start trigger for radialDeviceControl_v3 (trigger "arduino")
//
// The scope's frame clock goes to TRIGGER_PIN (BNC centre to D2, shield to GND). MATLAB sends 'a'
// to arm; the first rising edge after that sends "<Start>" once, and MATLAB starts its clock then.
// Edges before arming (focus / live view) are ignored. Later frames are not counted.
//
// An interrupt catches the edge, since frame-clock pulses can be microseconds long. The internal
// pull-up keeps an unplugged pin high, so noise can't make a rising edge.

const byte TRIGGER_PIN = 2;  // interrupt pin (Uno / Nano: 2 or 3)
volatile bool armed = false;
volatile bool fired = false;

void onEdge() {
  if (armed) {
    fired = true;
    armed = false;
  }
}

void setup() {
  pinMode(TRIGGER_PIN, INPUT_PULLUP);
  Serial.begin(115200);
  attachInterrupt(digitalPinToInterrupt(TRIGGER_PIN), onEdge, RISING);
  Serial.println("<Trigger ready>");
}

void loop() {
  if (fired) {
    fired = false;
    Serial.println("<Start>");
  }
  if (Serial.available() && Serial.read() == 'a') {
    noInterrupts();
    fired = false;
    armed = true;
    interrupts();
    Serial.println("<Armed>");
  }
}

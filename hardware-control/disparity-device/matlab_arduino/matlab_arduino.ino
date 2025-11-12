#include <Arduino.h>
#include <SPI.h>
#include <Adafruit_GC9A01A.h>

#define TFT_CS  10
#define TFT_RST  8
#define TFT_DC   9

Adafruit_GC9A01A tft(TFT_CS, TFT_DC, TFT_RST);
const uint32_t BAUD = 921600;

bool readBytesExact(uint8_t* buf, size_t n, uint32_t timeout_ms=4000) {
  size_t got = 0; uint32_t t0 = millis();
  while (got < n) {
    if (Serial.available())
      got += Serial.readBytes((char*)(buf + got), n - got); // <-- cast fix
    if (millis() - t0 > timeout_ms) return false;
  }
  return true;
}

void flushUntilHeader() {
  while (Serial.available()) {
    if (Serial.peek() == 'I') return;
    Serial.read();
  }
}

void setup() {
  Serial.begin(BAUD);
  while (!Serial) {}
  tft.begin();
  tft.setRotation(0);
  tft.fillScreen(0x0000);
  Serial.println("GC9A01A RX ready");
}

void loop() {
  if (!Serial.available()) return;
  flushUntilHeader();
  if (Serial.available() < 4) return;

  uint8_t hdr[4];
  if (!readBytesExact(hdr, 4)) return;
  if (hdr[0]!='I' || hdr[1]!='M' || hdr[2]!='G' || hdr[3]!='F') return;

  uint8_t wh[4];
  if (!readBytesExact(wh, 4)) return;
  uint16_t w = (uint16_t)wh[0] | ((uint16_t)wh[1] << 8);
  uint16_t h = (uint16_t)wh[2] | ((uint16_t)wh[3] << 8);
  if (!w || !h || w>240 || h>240) return;

  const size_t lineBytes = (size_t)w * 2;
  static uint16_t linebuf[240];
  static uint8_t  rxbuf[480];

  for (uint16_t y = 0; y < h; y++) {
    if (!readBytesExact(rxbuf, lineBytes)) return;
    for (uint16_t x = 0; x < w; x++) {
      uint8_t lo = rxbuf[2*x + 0];
      uint8_t hi = rxbuf[2*x + 1];
      linebuf[x] = (uint16_t)hi << 8 | lo;
    }
    tft.drawRGBBitmap(0, y, linebuf, w, 1);
  }
  Serial.write("OK", 2);
}

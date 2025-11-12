// Teensy 4.0 + GC9A01A — full-frame upload in one push
#include <Adafruit_GFX.h>
#include <Adafruit_GC9A01A.h>
#include <SPI.h>

// Teensy 4.0 SPI pins: SCK=13, MOSI=11
#define TFT_CS  10
#define TFT_DC  8
#define TFT_RST 9

// #define SCK 13
// #define MOSI 11

// Max panel size (adjust if your panel differs)
#define MAX_W 240
#define MAX_H 240

Adafruit_GC9A01A tft(TFT_CS, TFT_DC, TFT_RST);

// Full-frame buffer in RAM2 to avoid stack/DTCM pressure (~115 KB)
DMAMEM static uint16_t framebuf[MAX_W * MAX_H];

// Read exactly n bytes into buf; uses bulk reads for speed
bool readExact(void* buf, size_t n) {
  uint8_t* p = static_cast<uint8_t*>(buf);
  size_t off = 0;
  unsigned long start = millis();
  while (off < n) {
    // readBytes takes a char* on many Arduino cores
    size_t got = Serial.readBytes(reinterpret_cast<char*>(p + off), n - off);
    if (got) {
      off += got;
      start = millis(); // progress made; refresh idle timer
    } else if (millis() - start > 2000) { // 2s of no progress -> fail
      return false;
    }
  }
  return true;
}

static inline void fillColor(uint8_t r, uint8_t g, uint8_t b) {
  uint16_t c = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3);
  tft.fillScreen(c);
}

void setup() {
  Serial.begin(115200);          // baud ignored for USB CDC on Teensy
  Serial.setTimeout(40);         // short per-read wait; outer 2s guards hangs
  tft.begin();
  tft.setSPISpeed(60000000);     // 60 MHz SPI (GC9A01A is fine with this)
  tft.setRotation(0);
  tft.invertDisplay(true);
  fillColor(0, 0, 255);          // blue splash
}

void loop() {
  if (!Serial.available()) return;

  // Commands are "COLOR r g b" or "IMG W H"
  String line = Serial.readStringUntil('\n');
  line.trim();

  if (line.startsWith("COLOR")) {
    int r, g, b;
    if (sscanf(line.c_str(), "COLOR %d %d %d", &r, &g, &b) == 3) {
      fillColor((uint8_t)constrain(r,0,255),
                (uint8_t)constrain(g,0,255),
                (uint8_t)constrain(b,0,255));
      // Optional ack:
      // Serial.println("OK");
    } else {
      // Serial.println("ERR");
    }
    return;
  }

  if (line.startsWith("IMG")) {
    int w, h;
    if (sscanf(line.c_str(), "IMG %d %d", &w, &h) != 2) {
      // Serial.println("ERR");
      return;
    }
    if (w <= 0 || h <= 0 || w > MAX_W || h > MAX_H) {
      // Serial.println("ERR");
      return;
    }

    const uint32_t pixels = (uint32_t)w * (uint32_t)h;
    const uint32_t nbytes = pixels * 2;

    // Read the entire frame into framebuf
    if (!readExact(framebuf, nbytes)) {
      // Serial.println("ERR");
      return;
    }

    // If your host sends BIG-endian, uncomment to swap once for the whole frame:
    // for (uint32_t i = 0; i < pixels; ++i) framebuf[i] = __builtin_bswap16(framebuf[i]);

    // One-shot push
    tft.startWrite();
    tft.setAddrWindow(0, 0, w, h);               // set window
    tft.writePixels(framebuf, pixels, false);     // push entire frame
    tft.endWrite();

    // Optional ack:
    // Serial.println("OK");
  }
}

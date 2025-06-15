#include <SoftwareSerial.h>
#include <Adafruit_NeoPixel.h>
#include <math.h>

#define LED_COUNT 10
#define BRIGHTNESS_SCALE 0.3  // 降低亮度至原来的 30%

// 初始化 5 个灯带对象
Adafruit_NeoPixel rgb_display_7(LED_COUNT);
Adafruit_NeoPixel rgb_display_8(LED_COUNT);
Adafruit_NeoPixel rgb_display_9(LED_COUNT);
Adafruit_NeoPixel rgb_display_10(LED_COUNT);
Adafruit_NeoPixel rgb_display_11(LED_COUNT);

SoftwareSerial BTSerial(2, 3);

// 状态记录：每个灯带当前颜色值
int rVals[5] = {0, 0, 0, 0, 0};
int gVals[5] = {0, 0, 0, 0, 0};
int bVals[5] = {0, 0, 0, 0, 0};

// 获取灯带对象指针
Adafruit_NeoPixel* getStrip(int pin) {
  switch (pin) {
    case 7: return &rgb_display_7;
    case 8: return &rgb_display_8;
    case 9: return &rgb_display_9;
    case 10: return &rgb_display_10;
    case 11: return &rgb_display_11;
    default: return nullptr;
  }
}

// 将 PIN 7~11 映射到数组索引 0~4
int getIndexFromPin(int pin) {
  return pin - 7;
}

// 获取亮度波动因子（0.2 ~ 1.0）
float getBrightnessModulation() {
  float t = millis() / 1000.0;
  return 0.6 + 0.4 * sin(t * 2.0 * PI / 2.0);  // 2 秒周期，0.2~1.0
}

void setup() {
  Serial.begin(9600);
  BTSerial.begin(9600);

  rgb_display_7.begin();  rgb_display_7.setPin(7);   rgb_display_7.clear();  rgb_display_7.show();
  rgb_display_8.begin();  rgb_display_8.setPin(8);   rgb_display_8.clear();  rgb_display_8.show();
  rgb_display_9.begin();  rgb_display_9.setPin(9);   rgb_display_9.clear();  rgb_display_9.show();
  rgb_display_10.begin(); rgb_display_10.setPin(10); rgb_display_10.clear(); rgb_display_10.show();
  rgb_display_11.begin(); rgb_display_11.setPin(11); rgb_display_11.clear(); rgb_display_11.show();

  Serial.println("Arduino 启动：每个引脚接收 PINx:R,G,B 并以类正弦方式显示光照");
}

void loop() {
  // 蓝牙接收新指令
  if (BTSerial.available()) {
    String cmd = BTSerial.readStringUntil('\n');
    cmd.trim();
    Serial.println("收到指令：" + cmd);

    if (cmd.startsWith("PIN")) {
      int colonIndex = cmd.indexOf(':');
      int pin = cmd.substring(3, colonIndex).toInt();

      String rgbStr = cmd.substring(colonIndex + 1);
      int c1 = rgbStr.indexOf(',');
      int c2 = rgbStr.lastIndexOf(',');
      if (c1 == -1 || c2 == -1 || c1 == c2) return;

      int r = rgbStr.substring(0, c1).toInt();
      int g = rgbStr.substring(c1 + 1, c2).toInt();
      int b = rgbStr.substring(c2 + 1).toInt();

      int index = getIndexFromPin(pin);
      if (index >= 0 && index < 5) {
        rVals[index] = r;
        gVals[index] = g;
        bVals[index] = b;
        Serial.println("设置 PIN" + String(pin) + " -> R:" + r + " G:" + g + " B:" + b);
      }
    }
  }

  // 更新灯光效果
  float modulation = getBrightnessModulation();

  for (int pin = 7; pin <= 11; pin++) {
    int index = getIndexFromPin(pin);
    Adafruit_NeoPixel* strip = getStrip(pin);
    if (!strip) continue;

    int r = (int)(rVals[index] * BRIGHTNESS_SCALE * modulation);
    int g = (int)(gVals[index] * BRIGHTNESS_SCALE * modulation);
    int b = (int)(bVals[index] * BRIGHTNESS_SCALE * modulation);

    // 确保最小亮度至少为 1
    r = max(1, r);
    g = max(1, g);
    b = max(1, b);

    for (int i = 0; i < LED_COUNT; i++) {
      strip->setPixelColor(i, strip->Color(r, g, b));
    }
    strip->show();
  }

  delay(30); // 控制更新速度，防止过快闪烁
}

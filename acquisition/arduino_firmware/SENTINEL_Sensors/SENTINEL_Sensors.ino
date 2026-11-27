#include <DHT.h>

// SENTINEL firmware for Arduino Uno
// DHT22 DATA -> D2; LM35 VOUT -> A0; LDR module AO -> A1
#define DHT_PIN 2
#define DHT_TYPE DHT22
#define LM35_PIN A0
#define LDR_PIN A1

DHT dht(DHT_PIN, DHT_TYPE);
const unsigned long SENSOR_INTERVAL_MS = 2000;
unsigned long previousMillis = 0;

void setup() {
  Serial.begin(9600);
  dht.begin();
  pinMode(LDR_PIN, INPUT);
  delay(1000);
  Serial.println("SENTINEL_READY");
}

void loop() {
  const unsigned long now = millis();
  if (now - previousMillis < SENSOR_INTERVAL_MS) return;
  previousMillis = now;

  const float temperature = dht.readTemperature();
  const float humidity = dht.readHumidity();
  const int lm35Raw = analogRead(LM35_PIN); // retained for future diagnostics
  const int lightADC = analogRead(LDR_PIN);
  (void)lm35Raw;

  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("DHT22_ERROR");
    return;
  }

  // CSV protocol consumed by app.py: temperature,humidity,lightADC
  Serial.print(temperature, 1);
  Serial.print(',');
  Serial.print(humidity, 1);
  Serial.print(',');
  Serial.println(lightADC);
}

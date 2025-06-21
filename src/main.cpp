//==================================
//ESP32 WebSocket Server: Toggle LED
//by: Ulas Dikme
//==================================
#include <WiFi.h>
#include <WebServer.h>
#include <WebSocketsServer.h>
#include <Arduino.h>
#include <ArduinoJson.h>
//-----------------------------------------------
const char* ssid = "OPTUS_264100N";
const char* password = "eagle36379hf";
//-----------------------------------------------
#define LED1 26
#define LED2 25
#define LED3 33
#define LED4 32
#define BUTTON_PIN1 14
#define BUTTON_PIN2 4
#define BUTTON_PIN3 2
#define BUTTON_PIN4 15

//-----------------------------------------------
WebServer server(80);
WebSocketsServer webSocket = WebSocketsServer(81);
//-----------------------------------------------
boolean LEDonoff = false; String JSONtxt;
// Declare extern globals here
int sequence[4] = { 0 , 1 , 2, 3};
String mode = "IDLE";
//-----------------------------------------------
#include "html_page.h"
#include "functions.h"
//====================================================================

const int LED_PINS[4] = {LED1, LED2, LED3, LED4};
const int BUTTON_PINS[4] = {BUTTON_PIN1, BUTTON_PIN2, BUTTON_PIN3, BUTTON_PIN4};


static bool buttonLastStates[4] = {HIGH, HIGH, HIGH, HIGH};

// ================ HELPER FUNCTIONS ========================

void display_sequence(int sequence[], int length_t) {
  for (int i = 0; i < length_t; i++) {
    digitalWrite(LED_PINS[sequence[i]], HIGH);
    delay(500);
    digitalWrite(LED_PINS[sequence[i]], LOW);
    delay(300);
  }
}

void check_player_input() {
  for (int i = 0; i < 4; i++) {
    bool buttonState = digitalRead(BUTTON_PINS[i]);

    if (buttonState == LOW && buttonLastStates[i] == HIGH) {
      Serial.printf("Button %d Pressed!\n", i + 1);

      String msg = "{\"button\":" + String(i + 1) + "}";
      webSocket.broadcastTXT(msg);

      digitalWrite(LED_PINS[i], HIGH);
      delay(100);
      digitalWrite(LED_PINS[i], LOW);
    }
    buttonLastStates[i] = buttonState;
  }
}
// ================ MAIN FUNCTIONS ========================

void setup() {
  Serial.begin(115200);
  mode = "IDLE";

  for (int i = 0; i < 4; i++) {
    pinMode(LED_PINS[i], OUTPUT);
    pinMode(BUTTON_PINS[i], INPUT_PULLUP);
  }

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { Serial.print("."); delay(500); }
  WiFi.mode(WIFI_STA);
  Serial.println();
  Serial.print("Local IP: ");
  Serial.println(WiFi.localIP());
  delay(500);

  server.on("/", webpage);
  server.begin();
  webSocket.begin();
  webSocket.onEvent(webSocketEvent);
}

void loop() {
  webSocket.loop();
  server.handleClient();

  if (mode == "BEGIN") {
    display_sequence(sequence, 4);
    mode = "PLAY";
  }
  else if (mode == "PLAY") {
    Serial.print("Current sequence: ");
    for (int i = 0; i < 4; i++) {
      Serial.print(sequence[i]);
      Serial.print(" ");
    }
    Serial.println();

    check_player_input();
  }
  else if (mode == "END"){
    mode = "IDLE";
  }
  else if (mode == "IDLE") {
    Serial.println("IDLE");
  }

  delay(100);
}

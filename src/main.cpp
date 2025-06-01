//==================================
//ESP32 WebSocket Server: Toggle LED
//by: Ulas Dikme
//==================================
#include <WiFi.h>
#include <WebServer.h>
#include <WebSocketsServer.h>
//-----------------------------------------------
const char* ssid = "OPTUS_264100N";
const char* password = "eagle36379hf";
//-----------------------------------------------
#define LED 2
//-----------------------------------------------
WebServer server(80);
WebSocketsServer webSocket = WebSocketsServer(81);
//-----------------------------------------------
boolean LEDonoff=false; String JSONtxt;
//-----------------------------------------------
#include "html_page.h"
#include "functions.h"
//====================================================================
void setup()
{
  Serial.begin(115200); pinMode(LED, OUTPUT);
  //-----------------------------------------------
  WiFi.begin(ssid, password);
  while(WiFi.status() != WL_CONNECTED){Serial.print("."); delay(500);}
  WiFi.mode(WIFI_STA);
  Serial.println();
  Serial.print("Local IP: ");
  Serial.println(WiFi.localIP());
  //-----------------------------------------------
  server.on("/", webpage);
  //-----------------------------------------------
  server.begin(); webSocket.begin();
  webSocket.onEvent(webSocketEvent);
}
//====================================================================
void loop()
{
  webSocket.loop(); server.handleClient();
  //-----------------------------------------------
  if(LEDonoff == false) digitalWrite(LED, LOW);
  else digitalWrite(LED, HIGH);
  //-----------------------------------------------
  String LEDstatus = "OFF";
  if(LEDonoff == true) LEDstatus = "ON";
  JSONtxt = "{\"LEDonoff\":\""+LEDstatus+"\"}";
  webSocket.broadcastTXT(JSONtxt);
}
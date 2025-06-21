//=======================================
//handle function: send webpage to client
//=======================================
void webpage()
{
  server.send(200,"text/html", webpageCode);
}
//=====================================================
//function process event: new data received from client
//=====================================================
void webSocketEvent(uint8_t num, WStype_t type, uint8_t *payload, size_t welength)
{
    if(type == WStype_TEXT) // Receive text from client
    {
        Serial.println("Received message: ");
        Serial.println((char *)payload);

        JsonDocument doc;
        DeserializationError error = deserializeJson(doc, payload);

        if (error) {
            Serial.print("deserializeJson() failed: ");
            Serial.println(error.f_str());
            return;
        }

        String command = doc["command"];
        Serial.print("Command: ");
        Serial.println(command);

        // Optional: if your Python sends a sequence number
        

        if (command == "LEDonoff") {
            String value = doc["value"];
            LEDonoff = (value == "ON");
            Serial.print("LEDonoff set to: ");
            Serial.println(LEDonoff ? "ON" : "OFF");
        }
        else if (command == "begin") {
            // Handle "begin" command here
            Serial.println("Begin command received.");
            // Add your logic here
            mode = "BEGIN";  // OK now because enum declared in functions.h
            if (doc["seq"].is<JsonArray>()) {
                JsonArray seqArray = doc["seq"];
                int index = 0;
                for (int val : seqArray) {
                    if (index < 4) sequence[index] = val;
                    index++;
                }
            }
        }
        else if (command == "end") {
          Serial.println("End command received");
          mode = "END";
        }
    }
}

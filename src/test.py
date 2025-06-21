import asyncio
import websockets

ESP32_IP = "192.168.0.53"  # Replace with your ESP32's IP
PORT = 81                   # Your ESP32 WebSocket port

async def toggle_led(state):
    uri = f"ws://{ESP32_IP}:{PORT}"
    async with websockets.connect(uri) as websocket:
        # Send ON or OFF message
        message = f"LEDonoff={state}"
        await websocket.send(message)
        print(f"> Sent: {message}")
        
        # Wait for status response from ESP32
        response = await websocket.recv()
        print(f"< Received: {response}")


async def read():
    uri = f"ws://{ESP32_IP}:{PORT}"
    async with websockets.connect(uri) as websocket:
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=2)
            print(f"Hello < Received: {response}")
        except asyncio.TimeoutError:
            print("< No response received")

        await asyncio.sleep(3)



# Run the command: choose 'ON' or 'OFF'
asyncio.run(toggle_led("OFF"))  # or "OFF"
asyncio.run(read())


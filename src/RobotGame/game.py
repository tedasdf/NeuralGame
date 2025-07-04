import random
import time
import json
import websocket
import threading

class Game:
    def __init__(self, url, sequence_length):

        self.num_sequences = sequence_length
        self.sequence = []
        self.player_sequence = []

        self.active = False
        
        self.url = url
        self.connected = False

        self.reward = None
        
        
        self.ws = websocket.WebSocketApp(
            url,
            on_message=lambda ws, msg: self.on_message(ws, msg),
            on_open=lambda ws: self.on_open(ws),
            on_error=lambda ws, err: self.on_error(ws, err),
            on_close=lambda ws: self.on_close(ws)
        )

    def start_seq(self):
        # Start WebSocket listener in a separate thread
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()
        # Wait until connected (max wait time: 5 seconds)
        timeout = 5
        waited = 0
        while not game.connected and waited < timeout:
            print("Waiting for WebSocket connection...")
            time.sleep(1)
            waited += 1

    def reset(self):
        self.sequence = []
        self.player_sequence = []
        self.active = False
        self.starttime = None
        print("Game reset.")

        self.beginning_seq()

    def beginning_seq(self):
        self.sequence = random.sample(range(self.num_sequences), self.num_sequences)
        self.player_progress = 0
        self.active = True
        print(f"Generated sequence: {self.sequence}")
        

    def check_sequence(self):
        info = {}
        done = False
        reward = None
        cam_cap = None

        if not self.active:
            return "Game not active."
        else:    
            if self.player_sequence == self.sequence[:len(self.player_sequence)]:
                if len(self.player_sequence) == 4:
                        return "Win"
            
                return "Correct"
            else:
                self.active = False

                return "Incorrect"


    def on_message(self, ws, message):
        print(f"Received from ESP32: {message}")
        data = json.loads(message)

        if 'button' in data:
            print("Here the button")
            print(data['button'])
            button = data['button'] - 1  # ESP32 sends 1-indexed buttons
            self.player_sequence.append(button)
            print(f"Check player_sequence: {self.player_sequence}")

            # if result == "Win":
            #     ws.send(json.dumps({"command": "END", "result": "win"}))
            # elif result == "Incorrect":
            #     ws.send(json.dumps({"command": "END", "result": "incorrect"}))
            # else:
            #     pass  # Correct so far

    def on_open(self,ws):
        print("Connected to ESP32 WebSocket server.")
        self.connected = True
        seq = self.beginning_seq()
        self.ws.send(json.dumps({"command": "begin" , "seq" : seq}))

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def on_close(self, ws):
        self.connected = False
        print("WebSocket connection closed.")
    
    def send_command(self, command_dict):
        if self.ws.sock and self.ws.sock.connected:
            self.ws.send(json.dumps(command_dict))
            print(f"Sent: {command_dict}")
        else:
            print("WebSocket not connected")

    


if __name__ == "__main__":
    ESP32_IP = "192.168.0.100"  # Replace with your ESP32's IP
    PORT = 81                   # Your ESP32 WebSocket port
    url = f"ws://{ESP32_IP}:{PORT}"

    game = Game(url)

    # Start WebSocket listening thread
    game.start_seq()



    if game.connected:
        print("WebSocket connected successfully!")
        # You can now interact here (you could add input() to send custom commands)
        while True:
            if game.active == False:
                game.reset()
            print(game.player_sequence)
            print(game.sequence)
            print(game.check_sequence())
            time.sleep(1)
            # if game.player_sequence !== prev_sequence :
            #     print(game.player_sequence)


            # prev_sequence = game.player_sequence        
            

    else:
        print("WebSocket not connected.")






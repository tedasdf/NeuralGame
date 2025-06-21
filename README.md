Reinforcement Learning Robot with Bi-Directional Communication
Overview
This project combines robot design, reinforcement learning, and real-time communication to build an interactive system where a neural network controls a robot through a game-like environment.

The robot is programmed to replicate LED light sequences using four physical buttons. A custom reinforcement learning environment has been developed to train the agent to complete the task successfully. Communication between the robot (powered by an ESP32 microcontroller) and a Python-based training engine is fully bi-directional using WebSocket protocols.

Features
✅ Custom-built robot design (Fusion 360 CAD)

✅ Four-button control interface for interacting with LED sequences

✅ Reinforcement learning environment for training the robot to reproduce sequences

✅ Real-time, bi-directional communication between ESP32 and Python

✅ WebSocket-based communication channel for fast and reliable signal exchange

System Architecture
plaintext
Copy
Edit
[Python RL Agent]  <== WebSocket ==>  [ESP32 Microcontroller]  <==>  [Robot Hardware]
Python sends action commands to ESP32

ESP32 updates robot hardware (LEDs, buttons) and sends sensor states back to Python

Reinforcement Learning model is trained based on state-action feedback loop

Technologies Used
Python: Game engine, RL training environment

ESP32: Microcontroller handling robot hardware logic

WebSocket: Real-time, two-way communication

Fusion 360: Robot design and CAD files

Stable-Baselines3 (or your RL library): Reinforcement learning framework

Getting Started
Prerequisites
Python 3.x

ESP32 development environment (e.g. Arduino IDE or PlatformIO)

WebSocket Python library (websocket-client)

Run the Python Controller
bash
Copy
Edit
pip install websocket-client stable-baselines3
python controller.py
Flash ESP32 Code
Upload the ESP32 firmware via your preferred platform.

Folder Structure
plaintext
Copy
Edit
/
├── cad/                  # Fusion 360, STL, and STEP files
├── esp32/                # ESP32 firmware code
├── python/               # Python RL agent and environment
├── docs/                 # Design notes, schematics, documentation
└── README.md
Project Status
Robot design: ✅ Nearly complete

RL environment: ✅ Completed

Communication: ✅ Established and tested

Training: 🔄 In progress

Future Work
Add reward shaping to improve learning speed

Develop more complex button-LED sequences

Refine robot chassis and movement control

Test with different communication protocols (e.g., MQTT)

License
This project is open-source and available under the MIT License.

Contact
For questions or collaboration opportunities, feel free to reach out to me on LinkedIn or open an issue in this repository.


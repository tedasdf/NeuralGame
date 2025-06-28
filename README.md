# Reinforcement Learning Robot with Bi-Directional Communication

## Overview
This project combines **robot design, reinforcement learning, and real-time communication** to build an interactive system where a neural network controls a robot through a game-like environment.

The robot is programmed to replicate LED light sequences using **four physical buttons**. A **custom reinforcement learning environment** has been developed to train the agent to complete the task successfully. Communication between the robot (powered by an ESP32 microcontroller) and a Python-based training engine is fully **bi-directional** using WebSocket protocols.

---

## Features
- []  Custom-built robot design (Fusion 360 CAD)
- ✅ Four-button control interface for interacting with LED sequences
- ✅ Reinforcement learning environment for training the robot to reproduce sequences
- ✅ Real-time, bi-directional communication between ESP32 and Python
- ✅ WebSocket-based communication channel for fast and reliable signal exchange

---

## System Architecture
[Python RL Agent]  <== WebSocket ==>  [ESP32 Microcontroller]  <==>  [Robot Hardware]


## Technologies Used

* **Python**: Game engine, RL training environment  
* **ESP32**: Microcontroller handling robot hardware logic  
* **WebSocket**: Real-time, two-way communication  
* **Fusion 360**: Robot design and CAD files  
* **Stable-Baselines3 (or your RL library)**: Reinforcement learning framework  

---

## Getting Started

### Prerequisites

* Python 3.x  
* ESP32 development environment (e.g., Arduino IDE or PlatformIO)  
* WebSocket Python library (`websocket-client`)  

### Run the Python Controller

```bash
pip install websocket-client stable-baselines3
python controller.py
```

``` bash
├── docker-compose.yaml
├── rabbitMQ.Dockerfile
├── rabbit_init.sh
├── worker/
│   ├── Dockerfile
│   └── upload_worker.py
├── pacman/
│   ├── train.py
│   └── env/
│       └── pacman_env.py
├── redis.conf
└── .env



https://download.stereolabs.com/zedsdk/5.0/l4t35.4/ZED_SDK_L4T35.4_JETSON_XAVIER.run

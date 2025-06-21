# Reinforcement Learning Robot with Bi-Directional Communication

## Overview
This project combines **robot design, reinforcement learning, and real-time communication** to build an interactive system where a neural network controls a robot through a game-like environment.

The robot is programmed to replicate LED light sequences using **four physical buttons**. A **custom reinforcement learning environment** has been developed to train the agent to complete the task successfully. Communication between the robot (powered by an ESP32 microcontroller) and a Python-based training engine is fully **bi-directional** using WebSocket protocols.

---

## Features
- ✅ Custom-built robot design (Fusion 360 CAD)
- ✅ Four-button control interface for interacting with LED sequences
- ✅ Reinforcement learning environment for training the robot to reproduce sequences
- ✅ Real-time, bi-directional communication between ESP32 and Python
- ✅ WebSocket-based communication channel for fast and reliable signal exchange

---

## System Architecture
```plaintext
[Python RL Agent]  <== WebSocket ==>  [ESP32 Microcontroller]  <==>  [Robot Hardware]

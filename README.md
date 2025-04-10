Real-Time Traffic Light Timing Optimisation Using Reinforcement Learning
Presented at IEEE SCEECS’25, MANIT Bhopal – January 2025  
Read our IEEE paper - https://ieeexplore.ieee.org/document/10940668

Project Overview
This project explores the idea of optimizing traffic light timings using Reinforcement Learning (RL) to reduce traffic congestion and vehicle waiting times, as opposed to the traditional manually configured or fixed-timing traffic systems.
Our solution demonstrates how AI-driven traffic signal control can outperform static systems by adapting in real-time to actual traffic conditions.

Project Objectives
Show the effectiveness of RL-based traffic control versus fixed-timing methods.
Simulate and visualize real-world traffic scenarios.
Provide a web interface for users to interact with the simulation and observe performance comparisons.

Models Implemented
We trained and tested two models:
Fixed Model - Uses pre-defined, static signal timings.
RL Model - Learns optimal timing patterns using Reinforcement Learning.
Both models were developed in Python using TraCI (SUMO-Python interface) and trained for multiple episodes.
⚠️ We recommend training the RL model for at least 50 epochs for stable performance.

🛠️ Tech Stack
🎮 Simulation & Model Training
Python
TraCI (SUMO interface)
SUMO (Simulation of Urban MObility) – for visual simulation
🌐 Website
Frontend: Next.js + React.js
Backend: Node.js (API handling via Postman for testing)

🔄 Flow of the Application
Train the RL Model
Input training parameters (suggested: 50 epochs).
View training progress via a generated graph.
Compare Models
Run simulations for both the RL model and Fixed model.
Choose between:
With GUI (SUMO visual interface)
Without GUI (CLI-based simulation)
Compare total waiting times and observe improvements.

📊 Why It Matters
Traffic congestion is a major urban issue. Our project demonstrates that adopting a data-driven, adaptive approach to traffic control can significantly improve traffic flow, reduce fuel waste, and enhance commuter experience.









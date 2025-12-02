# Vertical Farming AI

Small, local AI platform for a vertical farming prototype:
- A simple sensor-based classifier that predicts plant health (`healthy` / `stressed`) from basic sensor readings (temperature, humidity, light hours, EC, pH).
- FastAPI service (`/predict`) that serves the model for local network access (sensors or controllers can POST JSON and get a response).
- Designed to be run locally for prototyping and on an edge device (Raspberry Pi) for field usage.

## Repo structure


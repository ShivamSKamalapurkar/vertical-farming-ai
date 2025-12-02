# 🌱 Vertical Farming AI  
A lightweight, local AI system designed for **smart vertical farming**, capable of predicting plant health from sensor readings and exposing the model through a **FastAPI** service for IoT devices and controllers.

---

# 🚀 Features

### ✅ Machine Learning Model  
- Classifies plant status as **healthy** or **stressed**  
- Uses simple sensor readings:  
  - Temperature  
  - Humidity  
  - Light hours  
  - EC (nutrient concentration)  
  - pH  

### ✅ FastAPI Backend  
- `/predict` endpoint returns health prediction + probabilities  
- Accessible from controllers (ESP32, Raspberry Pi, PC, sensors)

### ✅ Local & Edge Deployment  
- Works on Windows, Linux, Raspberry Pi  
- Very lightweight (scikit-learn + FastAPI)

---

# 📁 Directory Structure


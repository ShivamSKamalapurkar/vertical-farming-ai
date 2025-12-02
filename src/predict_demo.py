# src/predict_demo.py
import joblib
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, "models")

model = joblib.load(os.path.join(MODEL_DIR, "plant_health_model.joblib"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))

def predict_plant_status(temperature, humidity, light_hours, ec, pH):
    features = np.array([[temperature, humidity, light_hours, ec, pH]])
    pred = model.predict(features)[0]
    label = label_encoder.inverse_transform([pred])[0]
    return label

if __name__ == "__main__":
    temp = float(input("Enter temperature (°C): "))
    hum = float(input("Enter humidity (%): "))
    light = float(input("Enter light hours: "))
    ec = float(input("Enter EC: "))
    ph = float(input("Enter pH: "))
    result = predict_plant_status(temp, hum, light, ec, ph)
    print(f"\nPredicted plant status: {result}")

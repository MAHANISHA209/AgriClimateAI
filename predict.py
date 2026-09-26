import numpy as np
import joblib

# Load trained model
model = joblib.load("crop_model.pkl")

print("🌱 AgriClimate AI - Crop Recommendation")
print("----------------------------------------")

N = float(input("Enter Nitrogen (N): "))
P = float(input("Enter Phosphorus (P): "))
K = float(input("Enter Potassium (K): "))
temperature = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
ph = float(input("Enter Soil pH: "))
rainfall = float(input("Enter Rainfall: "))

input_data = np.array([[
    N, P, K,
    temperature,
    humidity,
    ph,
    rainfall
]])

prediction = model.predict(input_data)6

print("\n🌾 Recommended Crop:", prediction[0])
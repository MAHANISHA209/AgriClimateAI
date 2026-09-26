from flask import Flask, render_template, request
import numpy as np
import joblib
import requests

app = Flask(__name__)

# Load trained AI model
model = joblib.load("crop_model.pkl")


# -----------------------------------------
# Crop-Specific Information
# -----------------------------------------
crop_info = {

    "rice": {
        "water": "Rice generally requires sufficient water. Monitor soil moisture and avoid unnecessary irrigation during rainfall.",
        "climate": "Warm and humid conditions are generally suitable for rice cultivation.",
        "tip": "Maintain proper water management and regularly monitor the crop."
    },

    "maize": {
        "water": "Provide irrigation according to soil moisture and rainfall conditions.",
        "climate": "Maize grows well under warm conditions with adequate sunlight and moisture.",
        "tip": "Monitor soil moisture and avoid both excessive dryness and waterlogging."
    },

    "wheat": {
        "water": "Provide irrigation when soil moisture is insufficient, especially during important growth stages.",
        "climate": "Wheat generally performs better under cooler growing conditions.",
        "tip": "Monitor temperature and soil moisture regularly."
    },

    "cotton": {
        "water": "Irrigation should be adjusted according to rainfall and soil moisture.",
        "climate": "Cotton generally prefers warm conditions with suitable moisture availability.",
        "tip": "Monitor moisture and regularly inspect the crop."
    },

    "sugarcane": {
        "water": "Sugarcane requires adequate water. Irrigation should be managed according to rainfall and soil moisture.",
        "climate": "Warm conditions with sufficient moisture are generally suitable for sugarcane.",
        "tip": "Maintain proper irrigation and monitor crop growth."
    },

    "banana": {
        "water": "Banana requires regular moisture. Avoid excessive waterlogging around the roots.",
        "climate": "Warm and humid conditions are generally suitable for banana cultivation.",
        "tip": "Maintain soil moisture and provide proper drainage."
    },

    "chickpea": {
        "water": "Avoid excessive irrigation and monitor soil moisture carefully.",
        "climate": "Chickpea generally prefers relatively cool and dry growing conditions.",
        "tip": "Avoid waterlogging and monitor crop growth regularly."
    },

    "kidneybeans": {
        "water": "Provide moderate irrigation according to soil moisture and rainfall.",
        "climate": "Moderate temperatures and suitable moisture support kidney bean growth.",
        "tip": "Avoid excessive moisture and monitor the crop regularly."
    },

    "pigeonpeas": {
        "water": "Moderate irrigation is generally sufficient. Avoid prolonged waterlogging.",
        "climate": "Pigeon pea can grow under warm conditions with suitable moisture.",
        "tip": "Monitor rainfall and soil moisture throughout the crop cycle."
    },

    "mothbeans": {
        "water": "Moth beans are relatively drought tolerant, but soil moisture should still be monitored.",
        "climate": "Warm and relatively dry conditions can support moth bean cultivation.",
        "tip": "Avoid excessive irrigation and monitor soil moisture."
    },

    "mungbean": {
        "water": "Provide moderate irrigation and avoid excessive water accumulation.",
        "climate": "Warm conditions with suitable moisture are generally favourable.",
        "tip": "Monitor soil moisture and crop health regularly."
    },

    "blackgram": {
        "water": "Moderate irrigation should be provided according to rainfall and soil moisture.",
        "climate": "Warm conditions with suitable moisture can support blackgram growth.",
        "tip": "Avoid waterlogging and monitor the crop regularly."
    },

    "lentil": {
        "water": "Lentil generally requires moderate moisture and should not be over-irrigated.",
        "climate": "Cooler conditions are generally suitable for lentil cultivation.",
        "tip": "Avoid excessive irrigation and monitor soil moisture."
    },

    "apple": {
        "water": "Provide water according to soil moisture and local weather conditions.",
        "climate": "Apple cultivation generally requires suitable cooler climatic conditions.",
        "tip": "Monitor temperature and soil moisture carefully."
    },

    "orange": {
        "water": "Provide irrigation according to soil moisture and rainfall conditions.",
        "climate": "Warm conditions with suitable moisture can support orange cultivation.",
        "tip": "Maintain balanced soil moisture and monitor plant health."
    },

    "papaya": {
        "water": "Provide regular but controlled irrigation and ensure good drainage.",
        "climate": "Papaya generally prefers warm conditions and adequate moisture.",
        "tip": "Avoid waterlogging and monitor plant health."
    },

    "coconut": {
        "water": "Provide sufficient moisture during dry periods and monitor rainfall.",
        "climate": "Warm and humid conditions are generally suitable for coconut.",
        "tip": "Maintain soil moisture and proper drainage."
    },

    "grapes": {
        "water": "Irrigation should be managed according to soil moisture and weather conditions.",
        "climate": "Grapes generally require suitable temperature and controlled moisture.",
        "tip": "Monitor moisture and regularly inspect the plants."
    },

    "watermelon": {
        "water": "Provide sufficient moisture during growth but avoid excessive waterlogging.",
        "climate": "Warm conditions with adequate sunlight are generally suitable.",
        "tip": "Maintain balanced irrigation and monitor soil moisture."
    },

    "muskmelon": {
        "water": "Provide controlled irrigation and avoid excessive moisture.",
        "climate": "Warm and relatively dry conditions can support muskmelon cultivation.",
        "tip": "Avoid over-irrigation and monitor soil moisture."
    },

    "pomegranate": {
        "water": "Irrigation should be managed carefully according to soil moisture and rainfall.",
        "climate": "Pomegranate generally performs well under warm conditions.",
        "tip": "Avoid excessive irrigation and monitor plant health."
    },

    "coffee": {
        "water": "Maintain suitable soil moisture and provide water during dry periods.",
        "climate": "Coffee generally prefers suitable warm and humid conditions.",
        "tip": "Monitor moisture, temperature and plant health regularly."
    }

}


# -----------------------------------------
# Get Weather Information
# -----------------------------------------
def get_weather(location):

    try:

        # Step 1: Convert location name to coordinates
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"

        geo_params = {
            "name": location,
            "count": 1,
            "language": "en",
            "format": "json"
        }

        geo_response = requests.get(
            geo_url,
            params=geo_params,
            timeout=10
        )

        geo_response.raise_for_status()

        geo_data = geo_response.json()

        if "results" not in geo_data or not geo_data["results"]:
            return None

        latitude = geo_data["results"][0]["latitude"]
        longitude = geo_data["results"][0]["longitude"]
        place_name = geo_data["results"][0]["name"]

        # Step 2: Get current weather
        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation",
            "temperature_unit": "celsius",
            "precipitation_unit": "mm",
            "timezone": "auto"
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        )

        weather_response.raise_for_status()

        weather_data = weather_response.json()
        current = weather_data["current"]

        return {
            "location": place_name,
            "temperature": current["temperature_2m"],
            "humidity": current["relative_humidity_2m"],
            "precipitation": current["precipitation"]
        }

    except requests.exceptions.RequestException as e:

        print("Weather API Error:", e)

        return None

    except Exception as e:

        print("Weather Processing Error:", e)

        return None


# -----------------------------------------
# Home Page
# -----------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    irrigation = None
    climate_risk = None
    disease_risk = None
    action = None

    weather = None
    weather_error = None
    weather_advice = None

    # Crop information variables
    crop_water = None
    crop_climate = None
    crop_tip = None

    # Error message
    error_message = None

    if request.method == "POST":

        try:

            # -----------------------------------------
            # Farm Location
            # -----------------------------------------
            location = request.form.get("location", "").strip()

            # -----------------------------------------
            # Check Required Fields
            # -----------------------------------------
            required_fields = [
                "N",
                "P",
                "K",
                "temperature",
                "humidity",
                "ph",
                "rainfall"
            ]

            for field in required_fields:

                if request.form.get(field, "").strip() == "":

                    raise ValueError(
                        "Please fill all farm input fields."
                    )

            # -----------------------------------------
            # Convert Inputs to Numbers
            # -----------------------------------------
            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])

            # -----------------------------------------
            # Basic Validation
            # -----------------------------------------
            if N < 0 or P < 0 or K < 0:

                raise ValueError(
                    "N, P and K values cannot be negative."
                )

            if humidity < 0 or humidity > 100:

                raise ValueError(
                    "Humidity must be between 0 and 100."
                )

            if ph < 0 or ph > 14:

                raise ValueError(
                    "pH value must be between 0 and 14."
                )

            if rainfall < 0:

                raise ValueError(
                    "Rainfall cannot be negative."
                )

            # -----------------------------------------
            # Get Current Weather
            # -----------------------------------------
            if location:

                weather = get_weather(location)

                if weather is None:

                    weather_error = (
                        "Weather information could not be retrieved "
                        "for this location. You can still continue "
                        "with the farm analysis."
                    )

            # -----------------------------------------
            # Live Weather Advisory
            # -----------------------------------------
            if weather:

                live_temperature = weather["temperature"]
                live_humidity = weather["humidity"]
                live_rainfall = weather["precipitation"]

                if live_temperature > 35:

                    weather_advice = (
                        "High temperature detected. "
                        "Monitor heat stress and provide sufficient irrigation."
                    )

                elif live_temperature > 30 and live_rainfall == 0:

                    weather_advice = (
                        "Warm and dry conditions detected. "
                        "Monitor soil moisture and provide irrigation when required."
                    )

                elif live_rainfall > 5:

                    weather_advice = (
                        "Rainfall detected. "
                        "Avoid unnecessary irrigation and monitor waterlogging."
                    )

                elif live_humidity > 80:

                    weather_advice = (
                        "High humidity detected. "
                        "Monitor the crop regularly for disease risk."
                    )

                else:

                    weather_advice = (
                        "Current weather conditions are relatively stable. "
                        "Continue regular crop monitoring."
                    )

            # -----------------------------------------
            # AI Crop Prediction
            # -----------------------------------------
            input_data = np.array([[
                N,
                P,
                K,
                temperature,
                humidity,
                ph,
                rainfall
            ]])

            prediction = model.predict(input_data)[0]

            # -----------------------------------------
            # AI Confidence
            # -----------------------------------------
            confidence = round(
                max(model.predict_proba(input_data)[0]) * 100,
                2
            )

            # -----------------------------------------
            # Crop-Specific Information
            # -----------------------------------------
            selected_crop = prediction.lower()

            if selected_crop in crop_info:

                crop_water = crop_info[selected_crop]["water"]
                crop_climate = crop_info[selected_crop]["climate"]
                crop_tip = crop_info[selected_crop]["tip"]

            else:

                crop_water = (
                    "Manage irrigation according to rainfall "
                    "and soil moisture."
                )

                crop_climate = (
                    "Monitor local temperature, humidity "
                    "and weather conditions."
                )

                crop_tip = (
                    "Regularly monitor crop health and "
                    "soil conditions."
                )

            # -----------------------------------------
            # Irrigation Advice
            # -----------------------------------------
            if rainfall < 50:

                irrigation = (
                    "Low rainfall detected. "
                    "Regular irrigation is recommended."
                )

            elif rainfall < 150:

                irrigation = (
                    "Moderate rainfall. "
                    "Irrigation should be monitored."
                )

            else:

                irrigation = (
                    "Good rainfall level. "
                    "Reduce unnecessary irrigation."
                )

            # -----------------------------------------
            # Climate Risk
            # -----------------------------------------
            if temperature > 35:

                climate_risk = "High temperature risk."

            elif temperature > 30:

                climate_risk = "Moderate temperature risk."

            else:

                climate_risk = "Low temperature risk."

            # -----------------------------------------
            # Disease Risk Indicator
            # -----------------------------------------
            if humidity > 80 and temperature > 25:

                disease_risk = (
                    "High crop disease risk. "
                    "Monitor the crop regularly."
                )

            elif humidity > 70 and temperature > 20:

                disease_risk = (
                    "Moderate crop disease risk. "
                    "Regular monitoring is recommended."
                )

            else:

                disease_risk = "Low crop disease risk."

            # -----------------------------------------
            # Suggested Action
            # -----------------------------------------
            if rainfall < 50 and temperature > 30:

                action = (
                    "Provide sufficient irrigation "
                    "and monitor heat stress."
                )

            elif rainfall > 200:

                action = (
                    "Monitor waterlogging "
                    "and avoid excess irrigation."
                )

            else:

                action = (
                    "Continue regular crop monitoring."
                )

        # -----------------------------------------
        # Input Error
        # -----------------------------------------
        except ValueError as e:

            error_message = str(e)

        # -----------------------------------------
        # Unexpected Error
        # -----------------------------------------
        except Exception as e:

            print("Application Error:", e)

            error_message = (
                "Something went wrong while processing "
                "the farm analysis. Please check your inputs "
                "and try again."
            )

    # -----------------------------------------
    # Display Result
    # -----------------------------------------
    return render_template(

        "index.html",

        weather_advice=weather_advice,

        prediction=prediction,
        confidence=confidence,

        irrigation=irrigation,
        climate_risk=climate_risk,
        disease_risk=disease_risk,
        action=action,

        weather=weather,
        weather_error=weather_error,

        # Crop-specific information
        crop_water=crop_water,
        crop_climate=crop_climate,
        crop_tip=crop_tip,

        # Error message
        error_message=error_message
    )


# -----------------------------------------
# Run Flask Application
# -----------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load model
try:
    model = joblib.load("fare_model.pkl")
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

app = Flask(__name__)

@app.route("/")
def index():
    return jsonify({"message": "Taxi Fare Prediction API", "model_status": "ready" if model else "unavailable"})

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    required_fields = ["trip_distance", "passenger_count", "hour"]

    if not data or any(field not in data for field in required_fields):
        return jsonify({"error": f"Missing one or more fields: {required_fields}"}), 400

    try:
        trip_distance = float(data["trip_distance"])
        passenger_count = int(data["passenger_count"])
        hour = int(data["hour"])

        if trip_distance <= 0 or passenger_count <= 0 or not (0 <= hour <= 23):
            raise ValueError("Invalid input values.")

        features = np.array([[trip_distance, passenger_count, hour]])
        fare = model.predict(features)[0]
        return jsonify({"fare": round(float(fare), 2)})

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

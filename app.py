
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "ckd_model.pkl"
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "CKD Prediction API",
        "status": "ok"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Send input as JSON"}), 400

        features = pd.DataFrame([data])
        prediction = model.predict(features)[0]

        return jsonify({
            "prediction": str(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

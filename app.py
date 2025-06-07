from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import pandas as pd
import os

# Optional: pip install gdown if not already available
try:
    import gdown
except ImportError:
    os.system('pip install gdown')
    import gdown

# ✅ New Google Drive file ID for model.pkl
MODEL_FILE_ID = "1-wTGftA_jnsTxv0FcNK-EhZTt3upMvxB"
SCALER_FILE = "scaler.pkl"  # Assuming this is still local

# Download model.pkl if not already downloaded
MODEL_FILE = "model.pkl"
if not os.path.exists(MODEL_FILE):
    print("Downloading model.pkl from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_FILE, quiet=False)

# Load model and scaler
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

with open(SCALER_FILE, "rb") as f:
    scaler = pickle.load(f)

app = Flask(__name__)
CORS(app)

expected_fields = [
    'HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'Sex', 'Age',
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)

        if not all(field in data for field in expected_fields):
            missing = [field for field in expected_fields if field not in data]
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        input_data = [data[field] for field in expected_fields]
        input_df = pd.DataFrame([input_data], columns=expected_fields)
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        classes = {0: "No Diabetes", 1: "Prediabetes", 2: "Diabetes"}

        return jsonify({
            'prediction': int(prediction),
            'diagnosis': classes.get(prediction, "Unknown")
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

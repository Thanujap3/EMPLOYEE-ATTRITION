from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

# Load the pre-trained model
model = joblib.load('attrition_model.pkl')

# Frontend folder
FRONTEND_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../frontend')

@app.route('/')
def home():
    # Serve frontend index.html
    return send_from_directory(FRONTEND_FOLDER, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        age = float(data['age'])
        experience = float(data['experience'])
        salary = float(data['salary'])

        # Convert input to array for model
        input_data = np.array([[age, experience, salary]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Return result as JSON
        return jsonify({'prediction': str(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

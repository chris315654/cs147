from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the heart rate from the request
        heart_rate = request.json['heart_rate']
        
        # Convert the heart rate to a numpy array and reshape it
        heart_rate = np.array([[heart_rate]])
        
        # Standardize the heart rate
        heart_rate_scaled = scaler.transform(heart_rate)
        
        # Predict the emotion
        predicted_emotion = model.predict(heart_rate_scaled)
        
        # Return the predicted emotion as a JSON response
        return jsonify({'emotion': predicted_emotion[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

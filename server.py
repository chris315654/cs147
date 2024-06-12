from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get the heart rate from the query parameters
        heart_rate = request.args.get('heart_rate')
        
        if heart_rate is None:
            return jsonify({'error': 'heart_rate parameter is required'}), 400
        
        # Convert the heart rate to a float and then to a numpy array
        heart_rate = np.array([[float(heart_rate)]])
        
        # Standardize the heart rate
        heart_rate_scaled = scaler.transform(heart_rate)
        
        # Predict the emotion
        predicted_emotion = model.predict(heart_rate_scaled)
        
        # Return the predicted emotion as a JSON response
        return jsonify({'emotion': predicted_emotion[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

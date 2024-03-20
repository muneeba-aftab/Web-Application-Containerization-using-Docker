from flask import Flask, jsonify, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load('trained_model.pkl')  # Change 'your_model.pkl' to 'trained_model.pkl'

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.get_json(force=True)

    # Convert data to numpy array
    input_data = np.array(data['input'])

    # Make prediction using your model
    prediction = model.predict(input_data)

    # Convert prediction to list
    output = prediction.tolist()

    # Return the prediction as JSON response
    return jsonify({'prediction': output})

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app in debug mode


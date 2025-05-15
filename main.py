from flask import Flask, render_template, request
import numpy as np
import pickle

# Load your trained model
with open('crop_recommendation_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

flask_app = Flask(__name__)


@flask_app.route('/')
def index():
    return render_template('index.html')


@flask_app.route('/predict', methods=['POST'])
def predict():
    # Suppose your input features are float values from the form
    float_features = [float(x) for x in request.form.values()]
    features = np.array([float_features])

    prediction = model.predict(features)

    return render_template('index.html', prediction_text=f'The predicted crop is {prediction[0]}')


if __name__ == '__main__':
    flask_app.run(debug=True)

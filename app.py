import numpy as np
from flask import Flask, request,render_template
import pickle
from markupsafe import Markup

from flask import Flask, render_template

flask_app = Flask(__name__)
model = pickle.load(open("template/crop_recommendation_model.pkl", "rb"))

@flask_app.route("/")
def index():
    return (render_template("index.html")
@flask_app.route('/predict',methods=['POST']))
def predict():
    float_feature=[float(x) for x in request.form.values()]
    features = [np.array(float_feature)]
    prediction = model.predict(features)
    return render_template("index.html",prediction_text="The predicted crop is {}".format(prediction))




if __name__ == '__main__':
     flask_app.run(debug=True)
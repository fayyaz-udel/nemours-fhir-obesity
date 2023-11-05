from math import expm1
# import joblib
from utils import *
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin


# from tensorflow import keras
app = Flask(__name__)
cors = CORS(app)


# model = keras.models.load_model("assets/obesity_prediction_model.h5")
# transformer = joblib.load("assets/data_transformer.joblib")
@app.route("/", methods=["POST", "GET"])
@cross_origin()
def index():
    data = request.json
    prrocessed_data = process_input(data)

    anthropometric_data = extract_anthropometric(prrocessed_data)
    #########################################################################
    # prediction = model.predict(transformer.transform(df))
    predicted_bmi = 30  # expm1(prediction.flatten()[0])
    return jsonify(anthropometric_data)


if __name__ == '__main__':
    app.run(port=4000)

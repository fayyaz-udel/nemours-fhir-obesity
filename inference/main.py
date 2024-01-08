from math import expm1

import torch

from inference.util import inference
# import joblib
from util import *
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

# from tensorflow import keras
app = Flask(__name__)
cors = CORS(app)

# model = keras.models.load_model("assets/obesity_prediction_model.h5")
# transformer = joblib.load("assets/data_transformer.joblib")
map_dict = read_mapping_dicts()

models = load_models()


@app.route("/", methods=["POST", "GET"])
@cross_origin()
def index():
    data = request.json
    prrocessed_data = process_input(data, map_dict)
    prrocessed_data = map_concept_codes(prrocessed_data, map_dict)
    obser_pred_wins = determine_observ_predict_windows(prrocessed_data)  # Determine observation and prediction windows
    represantation_data = extract_representations(prrocessed_data, map_dict, obser_pred_wins)

    # represantation_data['dec'].to_csv('dec.csv')
    # represantation_data['enc'].to_csv('enc.csv')
    # represantation_data['demo'].to_csv('demo.csv')

    inference_data = inference(represantation_data, models, obser_pred_wins)

    anthropometric_data = extract_anthropometric_data(prrocessed_data)
    ehr_history = extract_ehr_history(prrocessed_data)

    response_dict = {**anthropometric_data, **inference_data, **ehr_history}
    #########################################################################
    return jsonify(response_dict)


if __name__ == '__main__':
    app.run(port=4000)

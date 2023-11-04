from math import expm1
# import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import json
from fhirclient.models.medicationrequest import MedicationRequest
from fhirclient.models.patient import Patient
from fhirclient.models.observation import Observation
from fhirclient.models.bundle import Bundle
from fhirclient.models.condition import Condition

# from tensorflow import keras
app = Flask(__name__)
cors = CORS(app)


# model = keras.models.load_model("assets/obesity_prediction_model.h5")
# transformer = joblib.load("assets/data_transformer.joblib")
@app.route("/", methods=["POST", "GET"])
@cross_origin()
def index():
    data = request.json
    medication_list = []
    observation_list = []
    condition_list = []
    patient = None
    for medication in data['medications']:
        medication_list.append(MedicationRequest(medication["resource"]))
    for observation in data['observations']:
        observation_list.append(Observation(observation["resource"]))
    for condition in data['conditions']:
        condition_list.append(Condition(condition["resource"]))
    patient = Patient(data['patient'])
    #########################################################################
    for medication in medication_list:
        print(medication.medicationCodeableConcept.coding[0].code)

    # df = pd.DataFrame(data, index=[0])
    # prediction = model.predict(transformer.transform(df))
    predicted_bmi = 30  # expm1(prediction.flatten()[0])
    return jsonify({"predicted bmi": str(predicted_bmi)})


if __name__ == '__main__':
    app.run(port=4000)

# import joblib
import os
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

from bmi_chart import plot_bmi_percentiles
from util import *

DEBUG = config["DEBUG"]

# from tensorflow import keras
app = Flask(__name__)
cors = CORS(app)

map_dict = read_mapping_dicts()

models = load_models()


@app.route("/", methods=["POST", "GET"])
@cross_origin()
def index():
    # if DEBUG:
    #     return mock()

    data = request.json
    prrocessed_data = process_input(data, map_dict)
    prrocessed_data = map_concept_codes(prrocessed_data, map_dict)
    obser_pred_wins = determine_observ_predict_windows(prrocessed_data)  # Determine observation and prediction windows
    represantation_data = extract_representations(prrocessed_data, map_dict, obser_pred_wins)

    net = models.get(obser_pred_wins["obser_max"], None)
    if net is None:
        inference_data = {'preds': "No model available to predict for patients at this age."}
    else:
        inference_data = inference(represantation_data, net, obser_pred_wins["obser_max"])

    anthropometric_data = extract_anthropometric_data(prrocessed_data, obser_pred_wins["obser_max"]) # BMI during the ages
    ehr_history = extract_ehr_history(prrocessed_data, obser_pred_wins["obser_max"] * 12)
    age_l = [x / 12 for x in anthropometric_data['bmi_x']]
    print(age_l, anthropometric_data['bmi_y'])
    buf = plot_bmi_percentiles(age_l, anthropometric_data['bmi_y'], sex=1)

    response_dict = {}
    print(data['patient']['name'][0])
    pop3 = '''<a href="./popup3.html?name='''+data['patient']['name'][0]['given'][0]+'''" target="_blank">
            <img src="assets/icon/info.png" alt="Button Image" height="25px"
            style="margin-bottom: 10px; margin-left: 10px;">
    </a>'''
    response_dict['img' ] = buf
    response_dict['pop3'] = pop3
    response_dict['name'] = data['patient']['name'][0]['given'][0] + " " + data['patient']['name'][0]['family']
    response_dict['dob'] = data['patient']['birthDate']
    response_dict['result'] = inference_data['preds']
    response_dict['risk'] = ehr_history['moc_data']
    #########################################################################
    return jsonify(response_dict)


def mock():
    result = '''    table goes here'''

    name = "<b>Name:</b> Truman Doe"
    dob = "<b>DOB:</b> 12-22-2019"
    risk = '''                <ul>
                    <li> Abilify tablet</li>
                </ul>'''

    pop3 = '''<a href="./popup3.html?name=John" target="_blank">
            <img src="assets/icon/info.png" alt="Button Image" height="25px"
            style="margin-bottom: 10px; margin-left: 10px;">
    </a>'''
    output_dict = {
        'name': name,
        'dob': dob,
        'pop3': pop3,
        'risk': risk,
        'result': result,
    }

    return output_dict


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 4000))
    if DEBUG:
        app.run(debug=True, host='0.0.0.0', port=port)
    else:
        app.run(debug=True, host='0.0.0.0', port=port, ssl_context=('./key/cert.pem', './key/key.pem'))

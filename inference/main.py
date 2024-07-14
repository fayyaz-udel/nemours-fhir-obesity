# import joblib
import os

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

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
    if DEBUG:
        return mock()

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

    anthropometric_data = extract_anthropometric_data(prrocessed_data, obser_pred_wins["obser_max"])
    ehr_history = extract_ehr_history(prrocessed_data, obser_pred_wins["obser_max"] * 12)

    response_dict = {**anthropometric_data, **inference_data, **ehr_history}
    #########################################################################
    return jsonify(response_dict)


def mock():
    result = '''                <ul>
                    <li>As part of every well child check, we like to see how a child is growing and talk about healthy
                        behaviors.
                    </li>
                    <li>Experts say that for most children maintaining a BMI (weight for height) less than the 95th
                        percentile is important to
                        avoiding
                        health risks like diabetes and heart disease.
                    </li>
                    <li><b>Truman</b> is currently at a <b>healthy weight</b>, but his <b>rate of weight gain is
                        faster</b> than we would expect.
                    </li>
                    <li>He has a <b>1 in 2</b> chance of developing an unhealthy weight by the <b> age of 7</b>.
                    </li>

                    <li>Healthy lifestyle changes can help <b>Truman</b> maintain a healthy weight and we have some great resources to share!</li>
                </ul>'''

    name = "<b>Name:</b> Truman Doe"
    dob = "<b>DOB:</b> 12-22-2019"
    risk = '''                <ul>
                    <li> Abilify tablet</li>
                </ul>'''

    curve = '''<img id="curve" src=assets/g_curve.jpeg width="450">'''

    pop3 = '''
    <a href="./popup3.html?name=John" target="_blank">
            <img src="assets/icon/info.png" alt="Button Image" height="25px"
            style="margin-bottom: 10px; margin-left: 10px;">
    </a>
    
    '''
    output_dict = {'result': result,
                   'name': name,
                   'dob': dob,
                   'risk': risk,
                   'curve': curve,
                   'pop3': pop3}

    return output_dict


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 4000))
    if DEBUG:
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        app.run(debug=False, host='0.0.0.0', port=port, ssl_context=('./key/cert.pem', './key/key.pem'))

import pandas as pd
import pickle
from fhirclient.models.condition import Condition
from fhirclient.models.medicationrequest import MedicationRequest
from fhirclient.models.observation import Observation
from fhirclient.models.patient import Patient

person_id = 820427166


def process_input(data):
    print(data)

    medication_list = []
    observation_list = []
    condition_list = []

    for medication in data['medications']:
        medication_list.append(MedicationRequest(medication["resource"]))
    for observation in data['observations']:
        observation_list.append(Observation(observation["resource"]))
    for condition in data['conditions']:
        condition_list.append(Condition(condition["resource"]))
    patient = Patient(data['patient'])
    #########################################################################
    medication_df = pd.DataFrame(
        columns=['system', 'code', 'display', 'date'])
    for medication in medication_list:
        medication_df = medication_df._append(
            {'system': medication.medicationCodeableConcept.coding[0].system,
             'code': medication.medicationCodeableConcept.coding[0].code,
             'display': medication.medicationCodeableConcept.coding[0].display,
             'date': medication.authoredOn.date}, ignore_index=True)

    observation_df = pd.DataFrame(columns=['system', 'code', 'display', 'value', 'unit', 'date'])
    for observation in observation_list:
        observation_df = observation_df._append(
            {'system': observation.code.coding[0].system,
             'code': observation.code.coding[0].code,
             'display': observation.code.coding[0].display,
             'value': observation.valueQuantity.value if observation.valueQuantity else None,
             'unit': observation.valueQuantity.unit if observation.valueQuantity else None,
             'date': observation.effectiveDateTime.date}, ignore_index=True)

    condition_df = pd.DataFrame(columns=['system', 'code', 'display', 'date'])
    for condition in condition_list:
        condition_df = condition_df._append(
            {'system': condition.code.coding[0].system,
             'code': condition.code.coding[0].code,
             'display': condition.code.coding[0].display,
             'date': condition.onsetDateTime.date}, ignore_index=True)

    ########################## Calculate Age ################################
    dob = pd.to_datetime(patient.birthDate.date, utc=True)
    for df in [medication_df, observation_df, condition_df]:
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df['age'] = (df['date'] - dob).dt.days / 365
        df.sort_values(by=['age'], inplace=True)

    return {'medications': medication_df, 'observations': observation_df, 'conditions': condition_df,
            'patient': patient}


def extract_anthropometric(data):
    observation_df = data['observations']

    observation_df['age'] = observation_df['age'].round(2)
    observation_df['value'] = observation_df['value'].round(2)

    height = observation_df[observation_df['code'] == '8302-2'][['age', 'value']]
    weight = observation_df[observation_df['code'] == '29463-7'][['age', 'value']]
    bmi = observation_df[observation_df['code'] == '39156-5'][['age', 'value']]

    return {"height_x": height["age"].to_list(), "height_y": height["value"].to_list(),
            "weight_x": weight["age"].to_list(), "weight_y": weight["value"].to_list(),
            "bmi_x": bmi["age"].to_list(), "bmi_y": bmi["value"].to_list()}


def extract_representations(processed_data):
    demo = extract_demo_data(processed_data)
    enc = extract_enc_data(processed_data)
    dec = extract_dec_data(processed_data)

    return {"demo": demo, "enc": enc, "dec": dec}


def extract_demo_data(data):
    with open('./data/vocab/demoVocab', 'rb') as f:
        demoVocab = pickle.load(f)

    eth = data["patient"].extension[1].extension[0].valueCoding.display
    race = data["patient"].extension[0].extension[0].valueCoding.display
    sex = data["patient"].gender.capitalize()
    payer = 'Medicaid/sCHIP' #TODO
    coi = 'COI_4' #TODO

    patient_dict = {}
    patient_dict["person_id"] = person_id
    patient_dict["Eth_dict"] = demoVocab.get(eth, 0)
    patient_dict["Race_dict"] = demoVocab.get(race, 0)
    patient_dict["Sex_dict"] = demoVocab.get(sex,0)
    patient_dict["Payer_dict"] = demoVocab.get(payer,0)
    patient_dict["COI_dict"] = demoVocab.get(coi,0)

    df = pd.DataFrame(patient_dict, index=[0])

    return df


def extract_enc_data(processed_data):
    # person_id	Age	value	feat_dict	age_dict

    cols_name = ["person_id", "Age", "value", "feat_dict", "age_dict"]

    df = pd.DataFrame(columns=cols_name)
    return df


def extract_dec_data(processed_data):
    with open("./data/vocab/featList", "rb") as fp:
        cols_name = pickle.load(fp)

    df = pd.DataFrame(columns=cols_name)

    return df

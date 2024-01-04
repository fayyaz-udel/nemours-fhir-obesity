import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
import math

from fhirclient.models.condition import Condition
from fhirclient.models.medicationrequest import MedicationRequest
from fhirclient.models.observation import Observation
from fhirclient.models.patient import Patient
from datetime import date
import pandas as pd
import importlib
import torch
import mimic_model_sig_obs as model

importlib.reload(model)

person_id = 820427166


def process_input(data, map_dict):
    medication_list = []
    observation_list = []
    condition_list = []
    family_history_list = []
    bmi_list = []

    if data['medications']:
        for medication in data['medications']:
            medication_list.append(MedicationRequest(medication["resource"]))
    if data['observations']:
        for observation in data['observations']:
            observation_list.append(Observation(observation["resource"]))
    if data['conditions']:
        for condition in data['conditions']:
            condition_list.append(Condition(condition["resource"]))
    patient = Patient(data['patient'])
    ####################################################################################################################
    medication_df = pd.DataFrame(columns=['system', 'code', 'display', 'date'])
    for idx, medication in enumerate(medication_list):
        curr = pd.DataFrame({'system': medication.medicationCodeableConcept.coding[0].system,
                             'code': medication.medicationCodeableConcept.coding[0].code,
                             'display': medication.medicationCodeableConcept.coding[0].display,
                             'date': medication.authoredOn.date}, index=[idx])
        medication_df = pd.concat([medication_df, curr], ignore_index=True)
    ####################################################################################################################
    observation_df = pd.DataFrame(columns=['system', 'code', 'display', 'value', 'unit', 'date'])
    for idx, observation in enumerate(observation_list):
        curr = pd.DataFrame({'system': observation.code.coding[0].system,
                             'code': observation.code.coding[0].code,
                             'display': observation.code.coding[0].display,
                             'value': observation.valueQuantity.value if observation.valueQuantity else None,
                             'unit': observation.valueQuantity.unit if observation.valueQuantity else None,
                             'date': observation.effectiveDateTime.date}, index=[idx])
        observation_df = pd.concat([observation_df, curr], ignore_index=True)
    ####################################################################################################################
    condition_df = pd.DataFrame(columns=['system', 'code', 'display', 'date'])
    for idx, condition in enumerate(condition_list):
        curr = pd.DataFrame({'system': condition.code.coding[0].system,
                             'code': condition.code.coding[0].code,
                             'display': condition.code.coding[0].display,
                             'date': condition.onsetDateTime.date}, index=[idx])
        condition_df = pd.concat([condition_df, curr], ignore_index=True)

    ################# Calculate BMI based on Height&Weight ######################
    height = observation_df[observation_df['code'] == '8302-2'][['date', 'value']]
    weight = observation_df[observation_df['code'] == '29463-7'][['date', 'value']]
    bmi = pd.merge(height, weight, on='date', how='inner')
    bmi['value'] = (bmi['value_y'] * 10000) / (bmi['value_x'] ** 2)

    bmi['code'] = '39156-5'
    bmi['display'] = 'Body mass index (BMI) [Ratio]'
    bmi['system'] = 'http://loinc.org'
    bmi['unit'] = 'kg/m2'
    bmi = bmi[['system', 'code', 'display', 'value', 'unit', 'date']]

    observation_df = pd.concat([bmi, observation_df], ignore_index=True)
    observation_df.drop_duplicates(subset=['value', 'code', 'date'], inplace=True)

    # TODO: FAMILY HISTORY
    # TODO: BMI --> maybe it need to be placed somewhere else
    ########################## Calculate Age ################################

    dob = pd.to_datetime(patient.birthDate.date, utc=True)
    medication_df = add_age(medication_df, dob)
    observation_df = add_age(observation_df, dob)
    condition_df = add_age(condition_df, dob)
    ########################## Map Concept Codes ################################
    observation_df, condition_df, medication_df = map_concept_codes(observation_df, condition_df, medication_df, map_dict)

    return {'medications': medication_df, 'observations': observation_df, 'conditions': condition_df,
            'patient': patient, 'family_history': family_history_list, 'bmi': bmi_list}


def add_age(df, dob):
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['age'] = (df['date'] - dob).dt.days / 30.4
    df['age_dict'] = df['age'].apply(lambda x: math.ceil(x) if x <= 24 else math.ceil(x / 12) + 22)
    df = df[df['age_dict'] <= 32]
    df = df.sort_values(by=['age'])
    return df


def map_concept_codes(obs_df, cond_df, med_df, map_dict):
    loinc2concept = map_dict["loinc2concept"]
    snomed2desc = map_dict["snomed2desc"]
    rxcode2concept = map_dict["rxcode2concept"]
    atc_map = map_dict["atc_map"]
    feat_vocab = map_dict["feat_vocab"]
    meas_q = map_dict["meas_quartiles"]

    ############################### Map Medication Codes ################################
    # 'system', 'code', 'display', 'date', 'age', 'age_dict'
    med_df['concept_id'] = med_df['code'].apply(lambda x: rxcode2concept.get(str(x).strip(), -111))
    med_df['atc_3_code'] = med_df['concept_id'].apply(lambda x: atc_map.get(str(x).strip(), -222))
    med_df['feat_dict'] = med_df['atc_3_code'].apply(lambda x: feat_vocab.get(x, -333))
    med_df = med_df[med_df['feat_dict'] != -333]
    ############################### Map Observation Codes ################################
    # 'system', 'code', 'display', 'value', 'unit', 'date', 'age', 'age_dict'
    obs_df['concept_id'] = obs_df['code'].apply(lambda x: loinc2concept.get(str(x).strip(), -666))
    obs_df = obs_df[obs_df['concept_id'] != -666]

    obs_df = obs_df.assign(quartile=obs_df.apply(lambda x: calc_q(x.concept_id, x.value, meas_q), axis=1))
    obs_df = obs_df.assign(
        feat_dict=obs_df.apply(lambda x: feat_vocab.get(x.concept_id + "_" + x.quartile, -777), axis=1))
    ############################### Map Conditions Codes ################################
    # 'system', 'code', 'display', 'date', 'age', 'age_dict'
    cond_df['feat'] = cond_df['code'].apply(lambda x: snomed2desc.get(str(x).strip(), -444))
    cond_df['feat_dict'] = cond_df['feat'].apply(lambda x: feat_vocab.get(str(x).strip(), -555))
    cond_df = cond_df[cond_df['feat_dict'] != -555]

    return obs_df, cond_df, med_df


def extract_representations(processed_data, map_dict, obser_pred_wins):
    demo = extract_demo_data(processed_data)
    enc = extract_enc_data(processed_data)
    dec = extract_dec_data(processed_data, map_dict, obser_pred_wins)

    return {"demo": demo, "enc": enc, "dec": dec}


def extract_demo_data(data):
    with open('./data/vocab/demoVocab', 'rb') as f:
        demoVocab = pickle.load(f)

    ### Mehak used race/eth vice versa wrongly
    # Eth_label_list = ['NI', 'White', 'Black', 'Some Other Race', 'Asian']
    # Race_label_list = ['Non-Hispanic', 'Hispanic', 'NI']
    # Sex_list = ['Female', 'Male']
    # payer_y_list = ['Medicaid/sCHIP', 'Private/Commercial', 'NI']
    # HL7_race_list =     ["American Indian or Alaska Native", "Asian", "Black or African American", "Native Hawaiian or Other Pacific Islander", "White", "Other Race"]
    eth = data["patient"].extension[1].extension[0].valueCoding.display
    race = data["patient"].extension[0].extension[0].valueCoding.display
    sex = data["patient"].gender.capitalize()
    payer = 'Medicaid/sCHIP'  # TODO
    coi = 'COI_2'  # TODO

    if eth == "Hispanic or Latino":
        eth = 'Hispanic'
    else:
        eth = 'Non-Hispanic'

    race_dict = {"American Indian or Alaska Native": 'Some Other Race',
                 "Asian": 'Asian',
                 "Black or African American": 'Black',
                 "Native Hawaiian or Other Pacific Islander": 'Some Other Race',
                 "White": 'White',
                 "Other Race": 'Some Other Race'}
    race = race_dict.get(race, 'NI')

    patient_dict = {}
    patient_dict["person_id"] = person_id
    patient_dict["Eth_dict"] = demoVocab.get(race, -1111)  ### race/eth are swapped in the Mehaks vocab
    patient_dict["Race_dict"] = demoVocab.get(eth, -1111)
    patient_dict["Sex_dict"] = demoVocab.get(sex, -1111)
    patient_dict["Payer_dict"] = demoVocab.get(payer, -1111)
    patient_dict["COI_dict"] = demoVocab.get(coi, -1111)

    df = pd.DataFrame(patient_dict, index=[0])

    return df


def extract_enc_data(processed_data):
    # person_id	Age	value	feat_dict	age_dict

    cols_name = ["person_id", "Age", "value", "feat_dict", "age_dict"]
    df = pd.DataFrame(columns=cols_name)

    cond = processed_data['conditions']
    for index, row in cond.iterrows():
        new_row = [person_id, row['age'], row['code'], row['feat_dict'], row['age_dict']]
        df = pd.concat([df, pd.DataFrame([new_row], columns=cols_name)], ignore_index=True)
    med = processed_data['medications']
    for index, row in med.iterrows():
        new_row = [person_id, row['age'], row['code'], row['feat_dict'], row['age_dict']]
        df = pd.concat([df, pd.DataFrame([new_row], columns=cols_name)], ignore_index=True)
    obs = processed_data['observations']
    for index, row in obs.iterrows():
        new_row = [person_id, row['age'], row['code'], row['feat_dict'], row['age_dict']]
        df = pd.concat([df, pd.DataFrame([new_row], columns=cols_name)], ignore_index=True)

    # TODO: FAMILY HISTORY
    # fam = processed_data['family_history']
    # for index, row in fam.iterrows():
    #     # df = df.append({"person_id": person_id, "Age": row['age'], "value": 1, "feat_dict": row['feat_dict'], "age_dict": row['age_dict']}, ignore_index=True)
    #     new_row = [person_id, row['age'], row['code'], row['feat_dict'], row['age_dict']]
    #     df = pd.concat([df, pd.DataFrame([new_row], columns=cols_name)], ignore_index=True)

    # TODO: BMI
    # bmi = processed_data['bmi']
    # for index, row in bmi.iterrows():
    #     # df = df.append({"person_id": person_id, "Age": row['age'], "value": 1, "feat_dict": row['feat_dict'], "age_dict": row['age_dict']}, ignore_index=True)
    #     new_row = [person_id, row['age'], row['code'], row['feat_dict'], row['age_dict']]
    #     df = pd.concat([df, pd.DataFrame([new_row], columns=cols_name)], ignore_index=True)

    ### filter based on age
    df = df[df['age_dict'] <= 24]
    ### filter unknowns
    df = df[df['feat_dict'] > 0]
    return df


def extract_dec_data(data, map_dict, obser_pred_wins):
    if obser_pred_wins['obser_max'] >= 3:
        obser_max_dict_value = obser_pred_wins['obser_max'] + 22
    else:
        return Exception("obser_max_dict_value is less than 3")
    d = map_dict['feat_vocab']
    dec_features = map_dict['dec_features']
    med, obs, cond = data['medications'], data['observations'], data['conditions']

    # decoder representation
    dr = pd.DataFrame(columns=dec_features['name'].values.tolist())

    # time_windows -> tw
    for tw in range(25, obser_max_dict_value + 1):
        med_t = med[med['age_dict'] == tw]
        obs_t = obs[obs['age_dict'] == tw]
        cond_t = cond[cond['age_dict'] == tw]

        dr = pd.concat([dr, pd.DataFrame({'person_id': [person_id], 'age_dict': [tw]})], ignore_index=True)
        for i, f in dec_features.iterrows():
            ############## medication ################
            if f['type'] == 'med':
                if med_t[med_t['atc_3_code'] == f['name']].shape[0] > 0:
                    dr.loc[dr.age_dict == tw, f['name']] = d[f['name']]
                else:
                    dr.loc[dr.age_dict == tw, f['name']] = d['0.0']
            ############## condition ################
            elif f['type'] == 'cond':
                if cond_t[cond_t['display'] == f['name']].shape[0] > 0:
                    dr.loc[dr.age_dict == tw, f['name']] = d[f['name']]
                else:
                    dr.loc[dr.age_dict == tw, f['name']] = d['0.0']

            ############## observation ################
            elif f['type'] == 'meas':
                curr_obs = obs_t[obs_t['concept_id'] == f['name']]
                if curr_obs.shape[0] > 0:
                    dr.loc[dr.age_dict == tw, f['name']] = curr_obs.head(1)['feat_dict'].values[0]
                else:
                    dr.loc[dr.age_dict == tw, f['name']] = d['0.0']
            #### TODO: FAMILY HISTORY
            # elif row['type'] == 'fh':
            #     dr.loc[dr.age_dict == time_window, f['name']] = d[f['name']]

    return dr


def read_mapping_dicts():
    print("Reading mapping dictionaries...")
    dec_features = pd.read_csv('./data/vocab/dec_features.csv', header=0)
    meas_quartiles = pd.read_csv('./data/meas_q_intervals.csv', header=0)
    meas_quartiles['col_name'] = meas_quartiles['col_name'].astype(str).apply(lambda x: x[:-2])
    meas_quartiles = meas_quartiles.set_index('col_name')

    with open('./data/map/loinc2concept', 'rb') as f:
        loinc2concept = pickle.load(f)
    with open('data/map/snomed2desc', 'rb') as f:
        snomed2desc = pickle.load(f)
    with open('./data/map/rxcode2conceptid', 'rb') as f:
        rxcode2concept = pickle.load(f)
    with open('./data/map/atc_map', 'rb') as f:
        atc_map = pickle.load(f)
    with open('./data/vocab/featVocab', 'rb') as f:
        feat_vocab = pickle.load(f)

    return {"meas_quartiles": meas_quartiles, "loinc2concept": loinc2concept, "snomed2desc": snomed2desc,
            "rxcode2concept": rxcode2concept, "atc_map": atc_map, "feat_vocab": feat_vocab,
            "dec_features": dec_features}


def calc_q(concept_id, value, meas_q):
    if concept_id in meas_q.index:
        qs = meas_q.loc[concept_id]
        if value < qs['q1']:
            return "1"
        elif value < qs['q2']:
            return "2"
        elif value < qs['q3']:
            return "3"
        else:
            return "4"
    else:
        return "-999"


def determine_observ_predict_windows(prrocessed_data):
    age = round((date.today() - prrocessed_data['patient'].birthDate.date).days / 365)

    if 6 >= age >= 2:
        obser_max = age
    else:
        obser_max = -10
    return {"obser_max": obser_max}


def load_models():
    models = {}
    models[2] = torch.load('./saved_models/obsNew_0.tar')
    models[3] = torch.load('./saved_models/obsNew_1.tar')
    models[4] = torch.load('./saved_models/obsNew_2.tar')
    models[5] = torch.load('./saved_models/obsNew_3.tar')
    models[6] = torch.load('./saved_models/obsNew_4.tar')

    return models


def extract_ehr_history(prrocessed_data, obser_pred_wins):
    m = prrocessed_data['medications']
    m['Type'] = 'Medication'
    o = prrocessed_data['observations']
    o['Type'] = 'Observation'
    c = prrocessed_data['conditions']
    c['Type'] = 'Condition'

    out_df = pd.concat([m, o, c], ignore_index=True)
    out_df.sort_values(by=['age'], inplace=True)
    out_df = out_df[out_df['age'] >= 0]
    out_df = out_df[out_df['feat_dict'] > 0]
    out_df['age'] = out_df['age'].astype(int)
    out_df.rename(columns={'display': 'Name', 'age': 'Age (months)', 'code': 'Code'}, inplace=True)

    out_df = out_df[['Age (months)', 'Type', 'Code', 'Name', 'value', 'unit', 'feat_dict']]

    return {"moc_data": out_df.to_html(na_rep="", index=False, justify='left')}


def extract_anthropometric_data(data):
    observation_df = data['observations']

    observation_df['age'] = observation_df['age'].round(0)
    observation_df['value'] = observation_df['value'].round(1)

    observation_df.sort_values(by=['age'], inplace=True)

    bmi = observation_df[observation_df['code'] == '39156-5'][['age', 'value']]

    return {"bmi_x": bmi["age"].to_list(), "bmi_y": bmi["value"].to_list()}


########################################################################################################################

def encXY(data):
    enc = data['enc']
    demo = data['demo']

    enc_len = pd.DataFrame(enc[enc['value'] != '0'].groupby('person_id').size().reset_index(name='counts'))

    # enc_ob.loc[enc_ob.value == 'Normal', 'label'] = 0
    # enc_ob.loc[enc_ob.value == 'Obesity', 'label'] = 2
    # enc_ob.loc[enc_ob.value == 'Overweight', 'label'] = 1

    enc_feat = enc['feat_dict'].values
    enc_eth = demo['Eth_dict'].values
    enc_race = demo['Race_dict'].values
    enc_sex = demo['Sex_dict'].values
    enc_payer = demo['Payer_dict'].values
    enc_coi = demo['COI_dict'].values
    enc_len = enc_len['counts'].values

    enc_age = enc['age_dict'].values

    # Reshape to 3-D
    # print(enc_feat.shape)
    enc_feat = torch.tensor(enc_feat.astype(int))
    enc_feat = torch.reshape(enc_feat, (1, -1))
    enc_feat = enc_feat.type(torch.LongTensor)

    enc_len = torch.tensor(enc_len)
    enc_len = enc_len.type(torch.LongTensor)

    enc_age = torch.tensor(enc_age.astype(int))
    enc_age = torch.reshape(enc_age, (1, -1))
    enc_age = enc_age.type(torch.LongTensor)
    ##########################################################################
    enc_eth = torch.tensor(enc_eth)
    enc_eth = enc_eth.unsqueeze(1)

    enc_race = torch.tensor(enc_race)
    enc_race = enc_race.unsqueeze(1)

    enc_sex = torch.tensor(enc_sex)
    enc_sex = enc_sex.unsqueeze(1)

    enc_payer = torch.tensor(enc_payer)
    enc_payer = enc_payer.unsqueeze(1)

    enc_coi = torch.tensor(enc_coi)
    enc_coi = enc_coi.unsqueeze(1)

    enc_demo = torch.cat((enc_eth, enc_race), 1)
    enc_demo = torch.cat((enc_demo, enc_sex), 1)
    enc_demo = torch.cat((enc_demo, enc_payer), 1)
    enc_demo = torch.cat((enc_demo, enc_coi), 1)
    enc_demo = enc_demo.type(torch.LongTensor)

    return enc_feat, enc_len, enc_age, enc_demo


def decXY(data):
    dec = data['dec']

    dec = dec.fillna(0)
    dec = dec.apply(pd.to_numeric)
    del dec['age_dict']

    dec_feat = dec.iloc[:, 2:].values

    dec_feat = torch.tensor(dec_feat)
    dec_feat = torch.reshape(dec_feat, (1, dec_feat.shape[0], dec_feat.shape[1]))

    return dec_feat


def inference(data, models, obser_pred_wins):
    net = models.get(obser_pred_wins["obser_max"], None)
    if net is None:
        preds = "No model available to predict for patients at this age."
    else:
        enc_feat, enc_len, enc_age, enc_demo = encXY(data)
        dec_feat = decXY(data)
        obs_idx = 0
        enc_feat, enc_len, enc_age, enc_demo, dec_feat = enc_feat[obs_idx], enc_len[obs_idx], enc_age[obs_idx],enc_demo[obs_idx], dec_feat[obs_idx]

        output, prob, disc_input, logits = net(False, False, enc_feat, enc_len, enc_age, enc_demo, dec_feat)

        output_prob_list = []
        output_time_list = []
        for i in range(0, len(prob)):
            output_prob_list.append(float(prob[i].squeeze()[1].data.cpu().numpy()))
            output_time_list.append(obser_pred_wins["obser_max"] + i + 1)

        preds = pd.DataFrame({'Age (years)': output_time_list, 'Probability': output_prob_list}).to_html()

    return {'preds': preds}

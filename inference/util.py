import pandas as pd
import torch
import importlib
import torch
import mimic_model_sig_obs as model

importlib.reload(model)

TEST = False


def encXY(data):
    if TEST:
        enc = pd.read_csv('./data/h/enc.csv', header=0)
    else:
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
    if TEST:
        dec = pd.read_csv('./data/h/dec.csv', header=0)
    else:
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
        preds =  "No model available to predict for patients at this age."
    else:
        enc_feat, enc_len, enc_age, enc_demo = encXY(data)
        dec_feat = decXY(data)
        obs_idx = 0
        enc_feat, enc_len, enc_age, enc_demo, dec_feat = enc_feat[obs_idx], enc_len[obs_idx], enc_age[obs_idx], enc_demo[
            obs_idx], dec_feat[obs_idx]

        output, prob, disc_input, logits = net(False, False, enc_feat, enc_len, enc_age, enc_demo, dec_feat)

        output_prob_list = []
        output_time_list = []
        for i in range(0, len(prob)):
            output_prob_list.append(float(prob[i].squeeze()[1].data.cpu().numpy()))
            output_time_list.append(obser_pred_wins["obser_max"] + i + 1)

        preds = pd.DataFrame({'Age (years)': output_time_list, 'Probability': output_prob_list}).to_html()

    return {'preds': preds}

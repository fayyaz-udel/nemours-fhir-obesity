import numpy as np
import torch
import importlib
import mimic_model_sig_obs as model
importlib.reload(model)

from util import encXY, decXY

feat_vocab_size = 490
age_vocab_size = 33
demo_vocab_size = 23

net = torch.load('./saved_models/obsNew_4.tar')


enc_feat, enc_len, enc_age, enc_demo = encXY()
dec_feat = decXY()
obs_idx = 0
enc_feat, enc_len, enc_age, enc_demo, dec_feat = enc_feat[obs_idx], enc_len[obs_idx], enc_age[obs_idx], enc_demo[obs_idx], dec_feat[obs_idx]

output, prob, disc_input, logits = net(False, False, enc_feat, enc_len, enc_age, enc_demo, dec_feat)

for i in range(0, len(prob)):
    print(output[i].squeeze().data.cpu().numpy())
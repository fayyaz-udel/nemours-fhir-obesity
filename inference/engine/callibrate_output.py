#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import os
import sys
import torch
import torch.nn as nn
from torch import optim
import importlib
import evaluation
importlib.reload(evaluation)

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')


def callibrate(prob,labels,logits):
    train_hids=list(range(prob.shape[0]))
    val_hids=random.sample(train_hids,int(len(train_hids)*0.05))
    test_hids=list(set(train_hids)-set(val_hids))
    prob=prob[val_hids]
    labels=labels[val_hids]
    logits_test=logits[test_hids]
    logits=logits[val_hids]
    
    
    
    if torch.cuda.is_available():
        device='cuda:0'
    device='cpu'
    temperature = nn.Parameter(torch.ones(1).to(device))
    temperature=temperature.type(torch.FloatTensor)
    #temperature=temperature.to('cuda:0')
    args = {'temperature': temperature}
    optimizer = optim.LBFGS([temperature], lr=0.0001, max_iter=1000000, line_search_fn='strong_wolfe')
    criterion = nn.CrossEntropyLoss()
    print('Final T_scaling factor: {:.2f}'.format(temperature.item()))
    def T_scaling(logits, args):
        temperature = args.get('temperature', None)
        return torch.div(logits, temperature)

    temps = []
    losses = []
    def _eval():
       # scaled=T_scaling(torch.tensor(output_dict['Prob']).cuda(), temperature)
        
        loss = criterion(T_scaling(torch.tensor(logits).type(torch.FloatTensor).to(device), args), torch.tensor(labels).type(torch.FloatTensor).to(device))
        #print(loss)
        loss.backward()
        temps.append(temperature.item())
        losses.append(loss)
        return loss

   
    

    optimizer.step(_eval)

    print('Final T_scaling factor: {:.2f}'.format(temperature.item()))
    print(args.get('temperature', None))
    pred=T_scaling(torch.tensor(logits_test).to(device), args)
    sm = nn.Sigmoid()
    prob_cal=sm(pred)
    #print(prob)
#     loss=evaluation.Loss(device,True,False,False,False,False,False,True,False,True,False,True,True)
#     print("BEFORE CALLIBRATION")
    
#     out_loss=loss(torch.tensor(prob),torch.tensor(labels))
#     output_dict['Prob']=prob.data.cpu().numpy()
    
    
#     print("AFTER CALLIBRATION")
#     out_loss=loss(prob_cal,torch.tensor(labels))   
    return test_hids,prob_cal.data.cpu().numpy()


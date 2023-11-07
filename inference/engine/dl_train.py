#!/usr/bin/env python
# coding: utf-8

import pickle
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import pandas as pd
import numpy as np
import torch as T
import torch
import math
from sklearn import metrics
import torch.nn as nn
from torch import optim
import importlib
import torch.nn.functional as F
import import_ipynb

import evaluation
import parameters
from parameters import *
#import model as model
import mimic_model as model
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from pickle import dump,load
from sklearn.model_selection import train_test_split
import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution,LayerDeepLift,DeepLift

#import torchvision.utils as utils
import argparse
from torch.autograd import Variable
from argparse import ArgumentParser
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

#save_path = "saved_models/model.tar"
if not os.path.exists("saved_models"):
    os.makedirs("saved_models")


importlib.reload(model)
import mimic_model as model
importlib.reload(parameters)
import parameters
from parameters import *
importlib.reload(evaluation)
import evaluation


class DL_models():
    def __init__(self,model_name,train,feat_vocab_size,age_vocab_size):
        
        self.model_name=model_name
        self.feat_vocab_size=feat_vocab_size
        self.age_vocab_size=age_vocab_size
        self.save_path="saved_models/"+model_name
        if torch.cuda.is_available():
            self.device='cuda:0'
        else:
            self.device='cpu'
        #self.device='cpu'
        self.loss=evaluation.Loss(self.device,False,False,False,False,False,False,True,False,True,False,False,False)
        if train:
            print("===============MODEL TRAINING===============")
            
            self.dl_train()
            
        else:
            print("===============MODEL TESTING===============")
            
            self.dl_test()
            
        
        
        
        
    def dl_test(self):
        with open('./saved_models/'+'khids', 'rb') as fp:
                   k_hids=pickle.load(fp)
        for i in range(5):
            print("==================={0:2d} FOLD=====================".format(i))
            path=self.save_path+'_'+str(i)+".tar"
            self.net=torch.load(path)
            print("[ MODEL LOADED ]")
            print(self.net)
#             self.model_test_full(test_hids[0:200],i)
#             print("TESTING BATHC 1")
            test_hids=list(k_hids[i])
            self.model_test(test_hids,i)
            self.model_test_full(test_hids[300:310],i)
            
    def create_kfolds(self):
        labels=pd.read_csv('./input_data.csv', header=0)
        
        
        hids=labels['person_id'].unique()
#         print(len(hids))
        ids=range(0,len(hids))
        batch_size=int(len(ids)/5)
        k_hids=[]
        for i in range(0,5):
            rids = random.sample(ids, batch_size)
            ids = list(set(ids)-set(rids))
            if i==0:
                k_hids.append(hids[rids])             
            else:
                k_hids.append(hids[rids])
                
        with open('./saved_models/'+'khids', 'wb') as fp:
                   pickle.dump(k_hids, fp)
        return k_hids


    def dl_train(self):
        k_hids=self.create_kfolds()
#         print(len(k_hids))
#         print(k_hids[0].shape)
              
        for i in range(5):
            self.create_model()
            print("[ MODEL CREATED ]")
            print(self.net)
            path=self.save_path+'_'+str(i)+".tar"
            print("==================={0:2d} FOLD=====================".format(i))
            #self.save_path="saved_models/"+self.model_name+"_"+str(i)+".tar"
            test_hids=list(k_hids[i])
            #test_hids=test_hids[0:200]
            train_ids=list(set([0,1,2,3,4])-set([i]))
            train_hids=[]
            for j in train_ids:
                train_hids.extend(k_hids[j])  
            #print(test_hids)
            #train_hids=train_hids[0:200]
            val_hids=random.sample(train_hids,int(len(train_hids)*0.1))
            train_hids=list(set(train_hids)-set(val_hids))
            min_loss=100
            counter=0
            for epoch in range(args.num_epochs):
                if counter==args.patience:
                    print("STOPPING THE TRAINING BECAUSE VALIDATION ERROR DID NOT IMPROVE FOR {:.1f} EPOCHS".format(args.patience))
                    break
                train_prob=[]
                train_pred=[]
                train_truth=[]
                train_loss=0.0
                self.net.train()
            
                print("======= EPOCH {:.1f} ========".format(epoch))
                for nbatch in range(int(len(train_hids)/(args.batch_size))):
                    enc_feat,enc_len,enc_age=self.encXY(train_hids[nbatch*args.batch_size:(nbatch+1)*args.batch_size])
                    dec_feat,dec_labels,mask=self.decXY(train_hids[nbatch*args.batch_size:(nbatch+1)*args.batch_size])
                    
                    output,prob,batch_loss = self.train_model(enc_feat,enc_len,enc_age,dec_feat,dec_labels,mask)
                    #print("batch_loss",batch_loss)
                    #print("dec_output",output[0].shape)
                    #print(np.asarray(output))
                    #temp=np.asarray(output)
                    #print(temp.shape)
                    #print("dec_prob",prob[0].shape)
#                     print("dec_labels",dec_labels.shape)
                    train_loss+=batch_loss
#                     train_prob.extend(prob.data.cpu().numpy())
#                     train_truth.extend(dec_labels.data.cpu().numpy())
#                     train_pred.extend(output.data.cpu().numpy())
                
                #print(train_prob)
                #print(train_truth)
#                 self.loss(torch.tensor(train_prob),torch.tensor(train_truth),torch.tensor(train_logits),False,False)
                train_loss=train_loss/(nbatch+1)        
                print("Total Train Loss: ", train_loss)
                val_loss=self.model_val(val_hids)
                print("Total Val Loss: ", val_loss)
                #print("Updating Model")
                #T.save(self.net,self.save_path)
                if(val_loss<=min_loss):
                    print("Validation results improved")
                    min_loss=val_loss
                    print("Updating Model")
                    T.save(self.net,path)
                    counter=0
                else:
                    print("No improvement in Validation results")
                    counter=counter+1
            self.model_test_full(test_hids[0:200],i)
#             print("TESTING BATHC 1")
            self.model_test(test_hids,i)
#             print("TESTING BATHC 2")
#             self.model_test(test_hids[1000:3000],i)
#             print("TESTING BATHC 3")
#             self.model_test(test_hids[3000:],i)
            #self.save_output()
    
    def train_model(self,enc_feat,enc_len,enc_age,dec_feat,dec_labels,mask):
        #print("mask",mask.shape)
        self.optimizer.zero_grad()
        # get the output sequence from the input and the initial hidden and cell states
        output,prob,disc_input = self.net(False,enc_feat,enc_len,enc_age,dec_feat,dec_labels,mask)
#         self.interpret(contri,dec_labels,disc_input)
        #print(len(prob))
        #print(len(output))
        #print(output[0].shape)
        #print(prob[0].shape)
#         print(dec_labels[0])
#         print(dec_labels[1])
        #dec_labels=dec_labels.permute(1,0)
        #dec_labels=dec_labels.reshape(-1)
        #print(dec_labels.shape)
        #print(disc_input.shape)
#         print(dec_labels[0])
#         print(dec_labels[1])
#         print(dec_labels[200])
        #d = {0:1, 1:2, 2:6, 3:13,4:25,5:33,6:174,7:39,8:42,9:44,10:173 }
        d = {0:0, 1:0, 2:1, 6:2, 13:3,25:4,33:5,174:6,39:7,42:8,44:9,173:10 }
        #dec_labels=torch.tensor([d[x.item()] for x in dec_labels])
#         print(prob)
        out_loss=0.0
        x_loss=0.0
        k_loss=0.0
        #dec_labels_dec=dec_labels_dec.type(torch.LongTensor)
        for s in range(dec_labels.shape[0]):
#             idx=torch.nonzero(mask[s,:]>0)
#             idx=idx.squeeze()
            target=dec_labels[s,:]
#             print(target)
#             target=target[idx]
            appx=disc_input[s,:]
#             print(appx)
            
            appx=(dec_labels[s,:]*mask[s,:])+(appx*(1-mask[s,:]))
#             print(mask[s,:])
#             print(appx)
            
            target_counts = [(target==1).sum(),(target==2).sum(),(target==6).sum(),(target==13).sum(),(target==25).sum(),(target==33).sum(),(target==174).sum(),(target==39).sum(),(target==42).sum(),(target==44).sum(),(target==173).sum()]
            
            appx_counts = [(appx==1).sum(),(appx==2).sum(),(appx==6).sum(),(appx==13).sum(),(appx==25).sum(),(appx==33).sum(),(appx==174).sum(),(appx==39).sum(),(appx==42).sum(),(appx==44).sum(),(appx==173).sum()]
#             print(target_counts)
#             print(appx_counts)
            appx_counts=torch.tensor(appx_counts)
            target_counts=torch.tensor(target_counts)
            appx_counts=appx_counts.type(torch.FloatTensor)
            appx_counts=appx_counts.to(self.device)
            appx_counts=F.log_softmax(torch.tensor(appx_counts, requires_grad=True))
            target_counts=target_counts.type(torch.FloatTensor)
            target_counts=target_counts.to(self.device)
            target_counts=F.softmax(torch.tensor(target_counts, requires_grad=True))
#             print(target_counts)
#             print(appx_counts)
            k_loss+=self.kl_loss(appx_counts,target_counts)
        k_loss=k_loss/dec_labels.shape[0]
        
        for i in range(len(prob)):
            idx=torch.nonzero(mask[:,i]>0)
            idx=idx.squeeze()
            m=mask[idx]
            m_prob=prob[i][idx]
            m_output=output[i][idx]
#             print(dec_labels.shape)
            dec_labels_dec=dec_labels[idx]
#             print(dec_labels_dec.shape)
            dec_labels_dec=torch.tensor([d[x.item()] for x in dec_labels_dec[:,i]])
            dec_labels_dec=dec_labels_dec.to(self.device)
            out_loss+=(self.criterion(m_prob,dec_labels_dec))/(m[:,i].sum())
#             print("ce loss",out_loss)
            inter_loss=torch.sum(torch.abs(dec_labels_dec - torch.tensor(m_output).to(self.device))*torch.tensor(m[:,i]).to(self.device))
            
            x_loss +=  inter_loss/ (torch.sum(m[:,i]) + 1e-5)
#             print("rmse loss",x_loss)
        out_loss=out_loss/len(prob)
        x_loss=x_loss/len(prob)
#         print("total CE loss",out_loss)
#         print("total RMSE loss",x_loss)
        total_loss=out_loss+x_loss+k_loss
#         print(out_loss+x_loss)
        # calculate the gradients
        total_loss.backward()
        # update the parameters of the model
        self.optimizer.step()
        
        return output,prob,total_loss
    
    def model_val(self,val_hids):
        #print("======= VALIDATION ========")
        
        val_prob=[]
        val_truth=[]
        val_logits=[]
        val_loss=0.0
        
        self.net.eval()
        

        enc_feat,enc_len,enc_age=self.encXY(val_hids)
        dec_feat,dec_labels,mask=self.decXY(val_hids)

        output,prob,disc_input = self.net(False,enc_feat,enc_len,enc_age,dec_feat,dec_labels,mask)

        d = {0:0, 1:0, 2:1, 6:2, 13:3,25:4,33:5,174:6,39:7,42:8,44:9,173:10 }

        out_loss=0.0
        x_loss=0.0
        k_loss=0.0
        #dec_labels_dec=dec_labels_dec.type(torch.LongTensor)
        for s in range(dec_labels.shape[0]):
#             idx=torch.nonzero(mask[s,:]>0)
#             idx=idx.squeeze()
            target=dec_labels[s,:]
#             target=target[idx]
            appx=disc_input[s,:]
            appx=(dec_labels[s,:]*mask[s,:])+(appx*(1-mask[s,:]))
            target_counts = [(target==1).sum(),(target==2).sum(),(target==6).sum(),(target==13).sum(),(target==25).sum(),(target==33).sum(),(target==174).sum(),(target==39).sum(),(target==42).sum(),(target==44).sum(),(target==173).sum()]
            appx_counts = [(appx==1).sum(),(appx==2).sum(),(appx==6).sum(),(appx==13).sum(),(appx==25).sum(),(appx==33).sum(),(appx==174).sum(),(appx==39).sum(),(appx==42).sum(),(appx==44).sum(),(appx==173).sum()]
#             print(target_counts)
#             print(appx_counts)
            appx_counts=torch.tensor(appx_counts)
            target_counts=torch.tensor(target_counts)
            appx_counts=appx_counts.type(torch.FloatTensor)
            appx_counts=appx_counts.to(self.device)
            appx_counts=F.log_softmax(torch.tensor(appx_counts, requires_grad=True))
            target_counts=target_counts.type(torch.FloatTensor)
            target_counts=target_counts.to(self.device)
            target_counts=F.softmax(torch.tensor(target_counts, requires_grad=True))
#             print(target_counts)
#             print(appx_counts)
            k_loss+=self.kl_loss(appx_counts,target_counts)
        k_loss=k_loss/dec_labels.shape[0]
        
        
        for i in range(len(prob)):
            idx=torch.nonzero(mask[:,i]>0)
            idx=idx.squeeze()
            m=mask[idx]
            m_prob=prob[i][idx]
            m_output=output[i][idx]
            dec_labels_dec=dec_labels[idx]
            dec_labels_dec=torch.tensor([d[x.item()] for x in dec_labels_dec[:,i]])
            dec_labels_dec=dec_labels_dec.to(self.device)
            out_loss+=(self.criterion(m_prob,dec_labels_dec))/(m[:,i].sum())
#             print("ce loss",out_loss)
            inter_loss=torch.sum(torch.abs(dec_labels_dec - torch.tensor(m_output).to(self.device))*torch.tensor(m[:,i]).to(self.device))
            
            x_loss +=  inter_loss/ (torch.sum(m[:,i]) + 1e-5)
        out_loss=out_loss/len(prob)
        x_loss=x_loss/len(prob)
        val_loss=out_loss+x_loss+k_loss
                    
        return val_loss.item()
    
    def model_test_full(self,test_hids,k):
        print("======= TESTING ========")
        
        val_prob=[]
        val_truth=[]
        val_logits=[]
        val_loss=0.0
        
        self.net.eval()


        enc_feat,enc_len,enc_age=self.encXY(test_hids)
        dec_feat,dec_labels,mask=self.decXY(test_hids)
#         print("dec_feat",dec_feat[0,0:2,:])
#         print("dec_labels",dec_labels[0])
        output,prob,disc_input,contri = self.net(True,enc_feat,enc_len,enc_age,dec_feat,dec_labels,mask)
#         print("mask",mask[0:3])
#         print("prob",prob[0][0:10])
#         print("output",output[0])
#         print(len(contri))
#         print(contri[0].shape)
        self.interpret(contri,dec_labels,disc_input)
        
        d = {0:0, 1:0, 2:1, 6:2, 13:3,25:4,33:5,174:6,39:7,42:8,44:9,173:10 }
        
        mask=np.asarray(mask)

            #print(mask.shape)

        auroc_mat=np.zeros((8,11))
        auprc_mat=np.zeros((8,11))
        auprc_base=np.zeros((8,11))


        for i in range(len(prob)):#time
            dec_labels_dec=torch.tensor([d[x.item()] for x in dec_labels[:,i]])
            m=mask[:,i]
            #print(m.shape)
            idx=list(np.where(m == 0)[0])
            #print(len(idx))

            for l in range(11):#class
                dec_labels_l=[1 if y==l else 0 for y in dec_labels_dec]
                #print("dec_labels_l",len(dec_labels_l))
                #print("=========================================================")
                prob_l=prob[i][:,l].data.cpu().numpy()

                #print(len(prob_l))
                #print(len(dec_labels_l))

                prob_l,dec_labels_l=np.asarray(prob_l),np.asarray(dec_labels_l)
                prob_l=np.delete(prob_l, idx)
                dec_labels_l=np.delete(dec_labels_l, idx)
                #print(prob_l.shape)
                #print(dec_labels_l.shape)
                auroc_mat[i,l],auprc_mat[i,l],auprc_base[i,l]=self.loss(prob_l,dec_labels_l)
#         print(auroc_mat)
#         print(auprc_mat)
#         print(auprc_base)
#         self.save_output(auroc_mat,auprc_mat,auprc_base,0,k,True)
#         self.display_output(auroc_mat,auprc_mat,auprc_base,0,k,True)
    
    def interpret(self,contri,dec_labels,disc_input):
        #print(len(contri))
        #print(contri[0].shape)
#         contri=contri[9].data.cpu().numpy()
        
#         print(dec_labels.shape)
#         print(disc_input.shape)
        #print(contri[0,:,0:2])
        
        with open("./featVocab2", 'rb') as fp:
                        featVocab=pickle.load(fp)
        with open("./ageVocab", 'rb') as fp:
                        ageVocab=pickle.load(fp)
        inv_featVocab = {v: k for k, v in featVocab.items()}
        inv_ageVocab = {v: k for k, v in ageVocab.items()}
#         print(ageVocab)
        for i in range(10):
            print("======SAMPLE======")
#             print("contri",contri.shape)
            ind_contri=pd.DataFrame(contri[i],columns=['feat','age','imp'])
            #print(ind_contri.head())
            ind_contri['feat']=[inv_featVocab[int(key)] for key in ind_contri['feat']]
            ind_contri['age']=[inv_ageVocab[int(key)] for key in ind_contri['age']]
            #print(ind_contri.head())
            ind_imp=ind_contri.groupby('feat').agg({'imp':sum}).reset_index()
            ind_imp=ind_imp.sort_values(by=['imp'])
            truth=[inv_featVocab[int(key)] for key in dec_labels[i]]
            pred=[inv_featVocab[int(key)] for key in disc_input[i]]
            self.display_contri(ind_imp,ind_contri,truth,pred)
        #print(contri[0,:,0:2])
        
        
    def model_test(self,total_test_hids,k):
        print("======= TESTING ========")
        
        n_batches=int(len(total_test_hids)/(args.batch_size))
#         print(n_batches)
#         n_batches=2
        total_auroc_mat=np.zeros((8,8,11))
        total_auprc_mat=np.zeros((8,8,11))
        total_auprc_base=np.zeros((8,8,11))
        total_samples=np.zeros((8,8,11))
                
        for nbatch in range(n_batches):
            print("==================={0:2d} BATCH=====================".format(nbatch))
            test_hids=total_test_hids[nbatch*args.batch_size:(nbatch+1)*args.batch_size]
            test_loss=0.0

            self.net.eval()

    #         print(dec_feat.shape)
    #         print(dec_labels.shape)
    #         print(mask.shape)
            
            for obs in range(8):
#                 print("======= OBS {:.1f} ========".format(obs+2))
                enc_feat,enc_len,enc_age=self.encXY(test_hids)
                dec_feat,dec_labels,mask=self.decXY(test_hids)

                pred_mask=np.zeros((mask.shape[0],mask.shape[1]))
                if obs>0:
                    pred_mask[:,0:obs]=mask[:,0:obs]#mask right
    #             print(pred_mask[2])
                pred_mask=torch.tensor(pred_mask)
                pred_mask=pred_mask.type(torch.LongTensor)
                #dec_feat
                #print(dec_labels)
                dec_labels_pred=dec_labels*pred_mask
                #print(dec_labels_pred)
                pred_mask_feat=pred_mask.unsqueeze(2)
                pred_mask_feat=pred_mask_feat.repeat(1,1,dec_feat.shape[2])
                pred_mask_feat=pred_mask_feat.type(torch.DoubleTensor)
    #             print(pred_mask_feat.shape)
    #             print(pred_mask_feat[2])
    #             print(dec_feat[7,:,:])

                dec_feat_pred=dec_feat*pred_mask_feat
    #             print(dec_feat[7,:,:])
                if obs>0:
    #                 print(pred_mask.shape)
                    obs_idx=torch.sum(pred_mask,1)
    #                 print(obs_idx.shape)
                    obs_idx=torch.add(obs_idx,-obs/2)
                    obs_idx=torch.nonzero(obs_idx>=0)
    #                 print("obs_idx",obs_idx.shape)
                    obs_idx=obs_idx.squeeze()
                    dec_feat_pred=dec_feat_pred[obs_idx]
                    dec_labels_pred=dec_labels_pred[obs_idx]
                    pred_mask=pred_mask[obs_idx]
                    mask=mask[obs_idx]
                    dec_labels=dec_labels[obs_idx]
                    enc_feat,enc_len,enc_age=enc_feat[obs_idx],enc_len[obs_idx],enc_age[obs_idx]


    #             print("dec_feat",dec_feat_pred[0,0:2,:])
    #             print("dec_labels",dec_labels_pred[0])    
                output,prob,disc_input = self.net(False,enc_feat,enc_len,enc_age,dec_feat_pred,dec_labels_pred,pred_mask)
    #             print("pred_mask",pred_mask[0:3])
    #             print("prob",prob[0][0:10])
    #             print("output",output[0])

                d = {0:0, 1:0, 2:1, 6:2, 13:3,25:4,33:5,174:6,39:7,42:8,44:9,173:10 }

                mask=np.asarray(mask)

                #print(mask.shape)

                auroc_mat=np.zeros((8,11))
                auprc_mat=np.zeros((8,11))
                auprc_base=np.zeros((8,11))
                n_samples=np.zeros((8,11))
                
                obs_mask=np.zeros((mask.shape[0],mask.shape[1]))
                obs_mask[:,obs:]=mask[:,obs:]#mask left side
                for i in range(obs,len(prob)):#time

                    dec_labels_dec=torch.tensor([d[x.item()] for x in dec_labels[:,i]])
                    m=obs_mask[:,i]
                    #print(m.shape)
                    idx=list(np.where(m == 0)[0])
                    #print(len(idx))

                    for l in range(11):#class
                        dec_labels_l=[1 if y==l else 0 for y in dec_labels_dec]
                        #print("dec_labels_l",len(dec_labels_l))
                        #print("=========================================================")
                        prob_l=prob[i][:,l].data.cpu().numpy()

                        #print(len(prob_l))
                        #print(len(dec_labels_l))

                        prob_l,dec_labels_l=np.asarray(prob_l),np.asarray(dec_labels_l)
                        prob_l=np.delete(prob_l, idx)
                        dec_labels_l=np.delete(dec_labels_l, idx)
                        #print(prob_l.shape)
                        #print(dec_labels_l.shape)
                        n_samples[i,l]=prob_l.shape[0]
                        auroc_mat[i,l],auprc_mat[i,l],auprc_base[i,l]=self.loss(prob_l,dec_labels_l)
                nan_mask = np.isnan(auroc_mat)
                nan_mask=1-nan_mask
#                 print(auroc_mat)
                auroc_mat[np.isnan(auroc_mat)] = 0
#                 print(auroc_mat)
                
#                 print(nan_mask)
                n_samples=n_samples*nan_mask
                n_samples[n_samples>0]=1
                auprc_mat=auprc_mat*nan_mask
                auprc_base=auprc_base*nan_mask
#                 print(n_samples)
                total_samples[obs]+=n_samples
                total_auroc_mat[obs]+=auroc_mat
                total_auprc_mat[obs]+=auprc_mat
                total_auprc_base[obs]+=auprc_base
#                 print(total_samples[obs])
#                 print(total_auroc_mat[obs])
        total_auroc_mat=np.divide(total_auroc_mat, total_samples)
        total_auprc_mat=np.divide(total_auprc_mat, total_samples)
        total_auprc_base=np.divide(total_auprc_base, total_samples)
        for time in range(8):
            self.save_output(total_auroc_mat[time],total_auprc_mat[time],total_auprc_base[time],time,k,False)
            self.display_output(total_auroc_mat[time],total_auprc_mat[time],total_auprc_base[time],time,k,False)
#             print(np.mean(auroc_mat,axis=1))
#             print(np.mean(auprc_mat,axis=1))
#             print(np.mean(auprc_base,axis=1))
    
    def model_interpret(self,meds,chart,out,proc,lab,stat,demo):
        meds=torch.tensor(meds).float()
        chart=torch.tensor(chart).float()
        out=torch.tensor(out).float()
        proc=torch.tensor(proc).float()
        lab=torch.tensor(lab).float()
        stat=torch.tensor(stat).float()
        demo=torch.tensor(demo).float()
        #print("lab",lab.shape)
        #print("meds",meds.shape)
        print("======= INTERPRETING ========")
        torch.backends.cudnn.enabled=False
        deep_lift=DeepLift(self.net)
        attr=deep_lift.attribute(tuple([meds,chart,out,proc,lab,stat,demo]))
        #print(attr)
        #print(attr.shape)
        torch.backends.cudnn.enabled=True
        
    def read_embeddings(self):
        print(self.net['emb_feat'])
    def encXY(self,ids):

        #print(len(ids))
        enc=pd.read_csv('./input_data.csv',header=0)
        enc_len=pd.read_csv('./enc_lengths.csv',header=0)
        #print(enc_len)
        enc=enc[enc['person_id'].isin(ids)]
        enc_len=enc_len[enc_len['index'].isin(ids)]
        
        enc_feat=enc['feat_dict'].values
        enc_len=enc_len['person_id'].values
        #print(enc_feat.shape)
        enc_age=enc['age_dict'].values
        
        #Reshape to 3-D
        #print(enc_feat.shape)
        enc_feat=torch.tensor(enc_feat)
        enc_feat=torch.reshape(enc_feat,(len(ids),-1))
        enc_feat=enc_feat.type(torch.LongTensor)
        
        enc_len=torch.tensor(enc_len)
        #enc_len=torch.reshape(enc_len,(len(ids),-1))
        enc_len=enc_len.type(torch.LongTensor)
        
        enc_age=torch.tensor(enc_age)
        enc_age=torch.reshape(enc_age,(len(ids),-1))
        enc_age=enc_age.type(torch.LongTensor)
        
#         print("enc_feat",enc_feat.shape)
#         print("enc_len",enc_len.shape)
        #print(enc_feat[0:5])
#         print("enc_age",enc_age.shape)
        #print(enc_age[0:5])
        return enc_feat,enc_len, enc_age  
            
           
    def decXY(self,ids):

        dec=pd.read_csv('./dec_input.csv',header=0)
        labels=pd.read_csv('./dec_output.csv',header=0)
        dec=dec[dec['person_id'].isin(ids)]
        labels=labels[labels['person_id'].isin(ids)]
        
        dec_feat=dec.iloc[:,2:].values
        #print(list(dec['person_id']))
        dec_labels=labels['label'].values
        mask=dec_labels.copy()
        mask[mask>0]=1
        
        #Reshape to 3-D
#         print(dec_feat.shape)
        dec_feat=torch.tensor(dec_feat)
        dec_feat=torch.reshape(dec_feat,(len(ids),8,dec_feat.shape[1]))
        
        dec_labels=torch.tensor(dec_labels)
        dec_labels=torch.reshape(dec_labels,(len(ids),-1))
        
        mask=torch.tensor(mask)
        mask=torch.reshape(mask,(len(ids),-1))
        
#         print(dec_feat.shape)
        #print(dec_feat[0:5])
#         print(dec_labels.shape)
#         print(dec_labels)
        
#         print(mask)
        return dec_feat,dec_labels,mask
    
    
    def create_model(self):
        self.net = model.EncDec2(self.device,
                           self.feat_vocab_size,
                           self.age_vocab_size,
                           embed_size=args.embedding_size,rnn_size=args.rnn_size,
                           batch_size=args.batch_size) 
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lrn_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss()
        self.net.to(self.device)
    
    def save_output(self,auroc_mat,auprc_mat,auprc_base,obs,k,full):
        if full:
            with open('./data/output/'+str(k)+'/'+'auroc_'+'full', 'wb') as fp:
                   pickle.dump(auroc_mat, fp)
            with open('./data/output/'+str(k)+'/'+'auprc_'+'full', 'wb') as fp:
                   pickle.dump(auprc_mat, fp)
            with open('./data/output/'+str(k)+'/'+'base_'+'full', 'wb') as fp:
                   pickle.dump(auprc_base, fp)
        else:
            with open('./data/output/'+str(k)+'/'+'auroc_'+str(obs), 'wb') as fp:
                   pickle.dump(auroc_mat, fp)
            with open('./data/output/'+str(k)+'/'+'auprc_'+str(obs), 'wb') as fp:
                   pickle.dump(auprc_mat, fp)
            with open('./data/output/'+str(k)+'/'+'base_'+str(obs), 'wb') as fp:
                   pickle.dump(auprc_base, fp)
            
    
    def display_output(self,auroc_mat,auprc_mat,auprc_base,obs,k,full):
        
#         print(auroc_mat)
        
        def plot(mat,base,full,title):
            #label=['3%','5%','10%','25%','50%','75%','85%','90%','95%','97%','>97%']
            
            
            auc_3 = mat[:,0]
            auc_5=mat[:,1]
            auc_10=mat[:,2]
            auc_25=mat[:,3]
            auc_50=mat[:,4]
            auc_75=mat[:,5]
            auc_85=mat[:,6]
            auc_90=mat[:,7]
            auc_95=mat[:,8]
            auc_97=mat[:,9]
            auc_ge97=mat[:,10]


            x = np.arange(8)  # the label locations
            barWidth = 0.08  # the width of the bars

            plt.rcParams["figure.figsize"] = (10,3)

            r1 = np.arange(len(auc_3))
            r2 = [x + barWidth for x in r1]
            r3 = [x + barWidth for x in r2]
            r4 = [x + barWidth for x in r3]
            r5 = [x + barWidth for x in r4]
            r6 = [x + barWidth for x in r5]
            r7 = [x + barWidth for x in r6]
            r8 = [x + barWidth for x in r7]
            r9 = [x + barWidth for x in r8]
            r10 = [x + barWidth for x in r9]
            r11 = [x + barWidth for x in r10]


            plt.bar(r1, auc_3, width=barWidth, label='3%')
            plt.bar(r2 , auc_5, width=barWidth, label='5%')
            plt.bar(r3, auc_10, width=barWidth, label='10%')
            plt.bar(r4, auc_25, width=barWidth, label='25%')
            plt.bar(r5 , auc_50, width=barWidth, label='50%')
            plt.bar(r6 , auc_75, width=barWidth, label='75%')
            plt.bar(r7, auc_85, width=barWidth, label='85%')
            plt.bar(r8, auc_90, width=barWidth, label='90%')
            plt.bar(r9, auc_95, width=barWidth, label='95%')
            plt.bar(r10, auc_97, width=barWidth, label='97%')
            plt.bar(r11, auc_ge97, color='#2d7f5e',width=barWidth, label='>97%')
            
            
            if title=='AUPRC':
                plt.bar(r1, base[:,0], color='black',width=barWidth)
                plt.bar(r2 , base[:,1], color='black',width=barWidth)
                plt.bar(r3, base[:,2], color='black',width=barWidth)
                plt.bar(r4, base[:,3], color='black',width=barWidth)
                plt.bar(r5 , base[:,4], color='black',width=barWidth)
                plt.bar(r6 , base[:,5], color='black',width=barWidth)
                plt.bar(r7, base[:,6], color='black',width=barWidth)
                plt.bar(r8, base[:,7], color='black',width=barWidth)
                plt.bar(r9, base[:,8], color='black',width=barWidth)
                plt.bar(r10, base[:,9], color='black',width=barWidth)
                plt.bar(r11, base[:,10], color='black',width=barWidth)
                
                
                
            #plt.plot(np.nanmean(mat,axis=1),marker='o')
            for x, y in zip(list(range(0,8)), np.nanmean(mat,axis=1)):
                plt.annotate(str(round(y,2)), (x,y+0.06))

            # Add some text for labels, title and custom x-axis tick labels, etc.
            plt.ylabel(title, fontsize=12)
            if full:
                plt.title('No obs and pred defined', fontsize=12)
            else:
                plt.title('Observation Window 0 to '+str(obs+2), fontsize=12)
            plt.xticks([r + 4.5*barWidth for r in range(len(auc_3))], ['3', '4', '5', '6', '7','8','9','10'], fontsize=18)
            plt.legend(loc="center right", bbox_to_anchor=(1.1,0.5), fontsize=8)
            
            if full:
                plt.savefig('./data/output/'+str(k)+'/'+title+'_'+'full'+'.png')
            else:
                plt.savefig('./data/output/'+str(k)+'/'+title+'_'+str(obs)+'.png')
            plt.show()
            
        
        plot(auroc_mat,auprc_base,full,title="AUROC")
        plot(auprc_mat,auprc_base,full,title="AUPRC")
    
    def display_contri(self,ind_imp,ind_contri,truth,pred):
       
        ind_imp=ind_imp[ind_imp['feat']!=0]
        ind_contri=ind_contri[ind_contri['feat']!=0]
        print("Ground Truth BMIp",truth)
        print("Predicted BMIp",pred)
        
#         print(ind_imp)
#         print(ind_contri)
        plt.rcParams["figure.figsize"] = (10,3)
        ind_imp["label"] = ind_imp["feat"].astype(str)
        plt.bar(ind_imp['label'], ind_imp['imp'])
        plt.xticks(rotation=20)
        plt.axhline(y=0)
        plt.title('Individual Average Contribution', fontsize=12)
        plt.show()
        
        plt.rcParams["figure.figsize"] = (10,3)
        ind_contri["label"] = ind_contri["feat"].astype(str) + '_' + ind_contri["age"].astype(str)
        plt.bar(ind_contri['label'], ind_contri['imp'])
        plt.xticks(rotation=20)
        plt.axhline(y=0)
        plt.title('Individual Contribution over Timeline', fontsize=12)
        plt.show()
            


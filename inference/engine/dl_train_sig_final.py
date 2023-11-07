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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import metrics
import torch.nn as nn
from torch import optim
import importlib
import torch.nn.functional as F
import import_ipynb
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
import evaluation
import callibrate_output
import fairness
import parameters
from parameters import *
#import model as model
import mimic_model_sig_Final as model
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
import mimic_model_sig_Final as model
importlib.reload(parameters)
import parameters
from parameters import *
importlib.reload(evaluation)
import evaluation
importlib.reload(fairness)
import fairness


class DL_models():
    def __init__(self,model_name,train,feat_vocab_size,age_vocab_size,demo_vocab_size):
        
        self.model_name=model_name
        self.feat_vocab_size=feat_vocab_size
        self.age_vocab_size=age_vocab_size
        self.demo_vocab_size=demo_vocab_size
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
        test_ids=pd.read_csv('./data/3/demo_test.csv',header=0)
        for i in range(1):
            print("==================={0:2d} FOLD=====================".format(i))
            path=self.save_path+'_'+str(i)+".tar"
            self.net=torch.load(path)
            print("[ MODEL LOADED ]")
            print(self.net)
            
#             print("TESTING BATHC 1")
            test_hids=list(test_ids['person_id'].unique())
            self.tp=0
            self.fp=0
            self.fn=0
            self.tn=0
            self.tp_non=0
            self.tp_obese=0
            self.fp_non=0
            self.fp_obese=0
    
#             self.model_test_full(test_hids,i)
#             print(self.tp)
#             print(self.fp)
#             print(self.fn)
#             print(self.tn)
#             print(self.tp_non)
#             print(self.tp_obese)
#             print(self.fp_non)
#             print(self.fp_obese)
            #self.interpret_read()
            self.model_test(test_hids,i)
            self.test_read(i)
            #self.interpret_embed()
            #self.model_test_full(test_hids[300:310],i)
    

            
#     def create_kfolds(self):
# #         labels=pd.read_csv('./input_data.csv', header=0)
        
#         train_ids=pd.read_csv('./data/3/labels_train.csv',header=0)
#         test_ids=pd.read_csv('./data/3/labels_test.csv',header=0)
#         train_ids.groupby(['person_id','label'])
#         ids2=train_ids.groupby(['person_id','label']).count().reset_index()
        
#         norm=list(ids2[ids2['label']==175]['person_id'].unique())
#         over=list(ids2[ids2['label']==176]['person_id'].unique())
#         ob=list(ids2[ids2['label']==177]['person_id'].unique())
        
        
#         norm_hids=random.sample(norm,int(26780*0.46))
#         over_hids=random.sample(over,int(26780*0.29))
#         ob_hids=random.sample(ob,int(26780*0.25))
        
# #         hids=labels['person_id'].unique()
# # #         print(len(hids))
# #         ids=range(0,len(hids))
# #         batch_size=int(len(ids)/5)
# #         k_hids=[]
# #         for i in range(0,5):
# #             rids = random.sample(ids, batch_size)
# #             ids = list(set(ids)-set(rids))
# #             if i==0:
# #                 k_hids.append(hids[rids])             
# #             else:
# #                 k_hids.append(hids[rids])
                
# #         with open('./saved_models/'+'khids_sig', 'wb') as fp:
# #                    pickle.dump(k_hids, fp)
#         train_ids=pd.read_csv('./data/3/demo_train.csv',header=0)
#         test_ids=pd.read_csv('./data/3/demo_test.csv',header=0)
#         return norm_hids,over_hids,ob_hids,list(test_ids['person_id'].unique())#list(train_ids['person_id'].unique()),list(test_ids['person_id'].unique())

    def create_kfolds(self):
#         labels=pd.read_csv('./input_data.csv', header=0)
    
        train_ids=pd.read_csv('./data/3/demo_train.csv',header=0)
        test_ids=pd.read_csv('./data/3/demo_test.csv',header=0)
        return list(train_ids['person_id'].unique()),list(test_ids['person_id'].unique())
    
    def dl_train(self):
        #norm_hids,over_hids,ob_hids,test_hids=self.create_kfolds()
        train_hids,test_hids=self.create_kfolds()
#         print(len(k_hids))
#         print(k_hids[0].shape)
              
        for i in range(1):
            self.create_model()
            print("[ MODEL CREATED ]")
            print(self.net)
            self.emb_model=torch.load("./saved_models/emb_.tar")
            print(self.emb_model)
            with torch.no_grad():
                self.emb_model.embeddings.weight.copy_(self.net.emb_feat.emb_feat.weight)

            path=self.save_path+'_'+str(i)+".tar"
#             self.net=torch.load(path)
            
#             print(self.net)
#             self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)
            print("==================={0:2d} FOLD=====================".format(i))
            #self.save_path="saved_models/"+self.model_name+"_"+str(i)+".tar"
            
            
#             val_hids1=random.sample(norm_hids,int(26780*0.016))
#             norm_hids=list(set(norm_hids)-set(val_hids1))
#             val_hids2=random.sample(over_hids,int(26780*0.016))
#             over_hids=list(set(over_hids)-set(val_hids2))
#             val_hids3=random.sample(ob_hids,int(26780*0.016))
#             ob_hids=list(set(ob_hids)-set(val_hids3))
#             val_hids1.extend(val_hids2)
#             val_hids1.extend(val_hids3)
#             val_hids=val_hids1
            
#             train_hids=norm_hids
#             train_hids.extend(over_hids)
#             train_hids.extend(ob_hids)
            val_hids=random.sample(train_hids,int(len(train_hids)*0.05))
            train_hids=list(set(train_hids)-set(val_hids))
            with open('./saved_models/'+'val_hids_sig_'+str(i), 'wb') as fp:
                  pickle.dump(val_hids, fp)
            #train_hids=list(set(train_hids)-set(val_hids))
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
                    #print(len(train_hids))
                    enc_feat,enc_len,enc_age,enc_demo,_=self.encXY(train_hids[nbatch*args.batch_size:(nbatch+1)*args.batch_size],train_data=True)
                    dec_feat,dec_labels,mask=self.decXY(train_hids[nbatch*args.batch_size:(nbatch+1)*args.batch_size],train_data=True)
                    
                    output,prob,batch_loss = self.train_model(enc_feat,enc_len,enc_age,enc_demo,dec_feat,dec_labels,mask)
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
                #print("Done training")
                val_loss1=self.model_val(val_hids[0:200])
                #print("Done val1")
                val_loss2=self.model_val(val_hids[200:400])
                val_loss3=self.model_val(val_hids[400:600])
                val_loss4=self.model_val(val_hids[600:800])
                val_loss5=self.model_val(val_hids[800:1000])
                val_loss6=self.model_val(val_hids[1000:1200])
                val_loss7=self.model_val(val_hids[1200:])
             
                val_loss=(val_loss1+val_loss2+val_loss3+val_loss4+val_loss5+val_loss6+val_loss7)/7
            
                print("Total Val Loss: ", val_loss)
                
                #print("Updating Model")
                #T.save(self.net,self.save_path)
                if(val_loss<=min_loss):
                    print("Validation results improved")
                    print("Updating Model")
                    T.save(self.net,path)
                    min_loss=val_loss
                    counter=0
                else:
                    print("No improvement in Validation results")
                    counter=counter+1
            #self.model_test_full(test_hids[0:200],i)
#             print("TESTING BATHC 1")
            #self.model_test(test_hids,i)
            #self.test_read(i)
#             print("TESTING BATHC 2")
#             self.model_test(test_hids[1000:3000],i)
#             print("TESTING BATHC 3")
#             self.model_test(test_hids[3000:],i)
            #self.save_output()
    
    def train_model(self,enc_feat,enc_len,enc_age,enc_demo,dec_feat,dec_labels,mask):
        #print("mask",mask.shape)
        self.optimizer.zero_grad()
        # get the output sequence from the input and the initial hidden and cell states
        output,prob,disc_input,_ = self.net(False,False,enc_feat,enc_len,enc_age,enc_demo,dec_feat,dec_labels,mask)
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
        
        #d = {0:0, 175:0,176:1,177:2}
        d = {0:0, 175:0,177:1}
        #dec_labels=torch.tensor([d[x.item()] for x in dec_labels])
#         print(prob)
        out_loss=0.0
        x_loss=0.0
        k_loss=0.0
        #dec_labels_dec=dec_labels_dec.type(torch.LongTensor)
        for s in range(dec_labels.shape[0]):
#             idx=torch.nonzero(mask[s,:]>0)
#             idx=idx.squeeze()
            #print(dec_labels.shape)
            #print(dec_labels[s,:])
            target=dec_labels[s,:]
#             print(target)
#             target=target[idx]
            appx=disc_input[s,:]
            #print(appx)
            
            appx=(dec_labels[s,:]*mask[s,:])+(appx*(1-mask[s,:]))
#             print(mask[s,:])
#             print(appx)
            
#             target_counts = [(target==175).sum(),(target==176).sum(),(target==177).sum()]
#             appx_counts = [(appx==175).sum(),(appx==176).sum(),(appx==177).sum()]
            
#             target_counts = [(target==175).sum(),(target==176).sum()+(target==177).sum(),(target==177).sum()]
#             appx_counts = [(appx==175).sum(),(appx==176).sum()+(appx==177).sum(),(appx==177).sum()]
            
            target_counts = [(target==175).sum(),(target==177).sum()]
            appx_counts = [(appx==175).sum(),(appx==177).sum()]
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
#             print(k_loss)
        k_loss=k_loss/dec_labels.shape[0]
#         print(k_loss)
        
        for i in range(len(prob)):#time
#             print("================================")
#             print(mask[0:10])
            idx=torch.nonzero(mask[:,i]>0)
            idx=idx.squeeze()
            #print(idx)
            m=mask[idx]
            m_prob=prob[i][idx]
            m_output=output[i][idx]
#             print(m.shape)
#             print(m_prob.shape)
#             print(m_output.shape)
#             print(dec_labels.shape)
            dec_labels_dec=dec_labels[idx]
#             print(dec_labels_dec[:,i])
            dec_labels_dec=torch.tensor([d[x.item()] for x in dec_labels_dec[:,i]])
#             print(dec_labels_dec)
#             print(m_output)
            dec_labels_sig=torch.zeros(dec_labels_dec.shape[0],args.labels)
            for x in range(dec_labels_dec.shape[0]):
                dec_labels_sig[x,dec_labels_dec[x]]=1
#             print(dec_labels_sig[0:10,:])
            dec_labels_dec=dec_labels_dec.to(self.device)
            #print(dec_labels_dec)
            #dec_labels_sig[:,1]=dec_labels_sig[:,1]+dec_labels_sig[:,2]
            dec_labels_sig=dec_labels_sig.to(self.device)
            #print(dec_labels_sig)
#             print(m_prob[0:10])
#             print(self.criterion(m_prob,dec_labels_sig))
            #out_loss+=(self.criterion(m_prob,dec_labels_sig))/(m[:,i].sum()+ 1e-5)
            out_loss+=(self.criterion(m_prob[:,1:],dec_labels_sig[:,1:]))/(m[:,i].sum()+ 1e-5)
#             print("ce loss",out_loss)
#             inter_loss=torch.sum(torch.abs(dec_labels_dec - torch.tensor(m_output).to(self.device))*torch.tensor(m[:,i]).to(self.device))
#             print("inter_loss",inter_loss)
#             x_loss +=  inter_loss/ (torch.sum(m[:,i]) + 1e-5)
#             print("rmse loss",x_loss)
        out_loss=out_loss/len(prob)
#         x_loss=x_loss/len(prob)
#         print("total CE loss",out_loss)
#         print("total RMSE loss",x_loss)
        total_loss=out_loss+k_loss#+x_loss
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
        

        enc_feat,enc_len,enc_age,enc_demo,_=self.encXY(val_hids,train_data=True)
        dec_feat,dec_labels,mask=self.decXY(val_hids,train_data=True)
        #print(enc_feat.shape,enc_demo.shape)
        output,prob,disc_input,_ = self.net(False,False,enc_feat,enc_len,enc_age,enc_demo,dec_feat,dec_labels,mask)

        #d = {0:0, 1:0, 2:1, 6:2, 13:3,25:4,33:5,174:6,39:7,42:8,44:9,173:10 }
        #d = {0:0, 175:0,176:1,177:2}
        d = {0:0, 175:0,177:1}
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
            target_counts = [(target==175).sum(),(target==177).sum()]
            appx_counts = [(appx==175).sum(),(appx==177).sum()]
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
            dec_labels_sig=torch.zeros(dec_labels_dec.shape[0],args.labels)
            for x in range(dec_labels_dec.shape[0]):
                dec_labels_sig[x,dec_labels_dec[x]]=1
#             print(dec_labels_sig[0:10,:])
            dec_labels_dec=dec_labels_dec.to(self.device)
            #dec_labels_sig[:,1]=dec_labels_sig[:,1]+dec_labels_sig[:,2]
            dec_labels_sig=dec_labels_sig.to(self.device)
            #out_loss+=(self.criterion(m_prob,dec_labels_sig))/(m[:,i].sum() + 1e-5)
            out_loss+=(self.criterion(m_prob[:,1:],dec_labels_sig[:,1:]))/(m[:,i].sum()+ 1e-5)
#             print("ce loss",out_loss)
#             inter_loss=torch.sum(torch.abs(dec_labels_dec - torch.tensor(m_output).to(self.device))*torch.tensor(m[:,i]).to(self.device))
            
#             x_loss +=  inter_loss/ (torch.sum(m[:,i]) + 1e-5)
        out_loss=out_loss/len(prob)
#         x_loss=x_loss/len(prob)
        val_loss=out_loss+k_loss#+x_loss
                    
        return val_loss.item()
    
    def model_test_full(self,total_test_hids,k):
        print("======= TESTING ========")
        
        val_prob=[]
        val_truth=[]
        val_logits=[]
        val_loss=0.0
        
        self.net.eval()
     
        n_batches=int(len(total_test_hids)/(args.batch_size))
        
        for nbatch in range(n_batches):
            print("==================={0:2d} BATCH=====================".format(nbatch))
            test_hids=total_test_hids[nbatch*args.batch_size:(nbatch+1)*args.batch_size]
            
            
            enc_feat,enc_len,enc_age,enc_demo,enc_ob=self.encXY(test_hids,train_data=False)
            dec_feat,dec_labels,mask=self.decXY(test_hids,train_data=False)
    #         print("dec_feat",dec_feat[0,0:2,:])
    #         print("dec_labels",dec_labels[0])
            output,prob,disc_input,contri = self.net(False,True,enc_feat,enc_len,enc_age,enc_demo,dec_feat,dec_labels,mask)
    #         print(len(contri))
    #         print(len(prob))
    #         print(prob[0].shape)
    #         print(prob[0][0:10])
    #         print("mask",mask[0:3])
            #print("prob",prob[0][0:10])
            #print(prob[0][9])
    #         print("output",output[0])
    #         print(len(contri))
    #         print(contri[0].shape)
            #print(contri)
            contri=contri.cpu().detach().numpy()
            #print(contri)
            self.interpret(contri,dec_labels,disc_input,prob,nbatch,enc_ob.data.cpu().numpy())
            #self.interpret(contri[3],dec_labels,disc_input)
        

    
    def interpret_embed(self):
        self.create_model()
        print("[ MODEL CREATED ]")
        print(self.net)
        #path="saved_models/emb_.tar"
        self.emb_model=torch.load("./saved_models/emb_.tar")
        print(self.emb_model)
        with torch.no_grad():
            self.emb_model.embeddings.weight.copy_(self.net.emb_feat.emb_feat.weight)
        
            
        emb=self.net(True,False,0,0,0,0,0,0,0)
        emb=emb.data.cpu().numpy()
        print(emb.shape)
        tsne = TSNE(2, verbose=1)
        tsne_proj = tsne.fit_transform(emb)
        print(tsne_proj.shape)
        # Plot those points as a scatter plot and label them based on the pred labels
        #cmap = cm.get_cmap('tab20')
        fig, ax = plt.subplots(figsize=(8,8))
        with open("./data/3/featVocab3", 'rb') as fp:
            featVocab=pickle.load(fp)
        inv_featVocab = {v: k for k, v in featVocab.items()}
        for i in range(1,174):
            ax.scatter(tsne_proj[i,0],tsne_proj[i,1], label = inv_featVocab[i] ,alpha=0.5, color = 'hotpink')
            ax.annotate(str(i), (tsne_proj[i,0],tsne_proj[i,1]))
        for i in range(174,178):
            ax.scatter(tsne_proj[i,0],tsne_proj[i,1], label = inv_featVocab[i] ,alpha=0.5, color = 'blue')
            ax.annotate(str(i), (tsne_proj[i,0],tsne_proj[i,1]))
        for i in range(178,350):
            ax.scatter(tsne_proj[i,0],tsne_proj[i,1], label = inv_featVocab[i] ,alpha=0.5, color = 'green')
            ax.annotate(str(i), (tsne_proj[i,0],tsne_proj[i,1]))
        
        plt.show()
        
    def interpret(self,contri,dec_labels,disc_input,prob,nbatch,enc_ob):
        #print(len(contri))
        #print(contri[0].shape)
#         contri=contri[9].data.cpu().numpy()
        
#         print(dec_labels.shape)
#         print(disc_input.shape)
        #print(contri[0,:,0:2])
        final_tp=pd.DataFrame()
        final_fp=pd.DataFrame()
        final_fn=pd.DataFrame()
        final_tn=pd.DataFrame()
        final_tp_non=pd.DataFrame()
        final_tp_obese=pd.DataFrame()
        final_fp_non=pd.DataFrame()
        final_fp_obese=pd.DataFrame()
        
        
        
        
        with open("./data/3/featVocab3", 'rb') as fp:
                        featVocab=pickle.load(fp)
        with open("./data/3/ageVocab", 'rb') as fp:
                        ageVocab=pickle.load(fp)
        inv_featVocab = {v: k for k, v in featVocab.items()}
        inv_ageVocab = {v: k for k, v in ageVocab.items()}
        #print(enc_ob.shape)
        #print(enc_ob)
#         print(ageVocab)
        for i in range(enc_ob.shape[0]):
            #print("======SAMPLE======")
#             print("contri",contri.shape)
            ind_contri=pd.DataFrame(contri[i],columns=['feat','age','imp'])
            #print(ind_contri.head())
            ind_contri['feat']=[inv_featVocab[int(key)] for key in ind_contri['feat']]
            ind_contri['age']=[inv_ageVocab[int(key)] for key in ind_contri['age']]
            #print(ind_contri.head())
            ind_imp=ind_contri.groupby('feat').agg({'imp':'mean'}).reset_index()
            ind_imp=ind_imp.sort_values(by=['imp'])
            #print(ind_imp.head())
            
            truth=[inv_featVocab[int(key)] for key in dec_labels[i]]
            pred=[inv_featVocab[int(key)] for key in disc_input[i]]
            #print(truth)
            #print(pred)
            
            path='data/output/imp'
            if not os.path.exists(path):
                os.makedirs(path)          
            age=0
            if truth[age]=='BMIp_obese' or truth[age]=='BMIp_overweight':
                if pred[age]=='BMIp_obese' or pred[age]=='BMIp_overweight':
                    if final_tp.empty:
                        final_tp=ind_imp
                    else:
                        final_tp=pd.concat([final_tp,ind_imp],axis=0)
                        self.tp=self.tp+1
                        #self.display_contri(ind_imp,"TP")
                    if enc_ob[i]==1 or enc_ob[i]==2:
                        if final_tp_obese.empty:
                            #print('empty')
                            final_tp_obese=ind_imp
                        else:
                            final_tp_obese=pd.concat([final_tp_obese,ind_imp],axis=0)
                            self.tp_obese=self.tp_obese+1
                            #print(self.tp_obese)
                            if ind_imp[ind_imp['feat']=='Failure to gain weight'].shape[0]>0:
                                print(ind_imp[ind_imp['feat']=='Failure to gain weight'])
                            #self.display_contri(ind_imp,"TP_obese")
                    else:
                        if final_tp_non.empty:
                            final_tp_non=ind_imp
                        else:
                            final_tp_non=pd.concat([final_tp_non,ind_imp],axis=0)
                            self.tp_non=self.tp_non+1
                else:#if pred[0]=='Not over/obese':
                    if final_fn.empty:
                        final_fn=ind_imp
                    else:
                        final_fn=pd.concat([final_fn,ind_imp],axis=0)
                        self.fn=self.fn+1
                    
            
            if truth[age]=='Not over/obese':
                if truth[age]==pred[age]:
                    if final_tn.empty:
                        final_tn=ind_imp
                    else:
                        final_tn=pd.concat([final_tn,ind_imp],axis=0)
                        self.tn=self.tn+1
                else:#if pred[0]=='BMIp_obese':
                    if final_fp.empty:
                        final_fp=ind_imp
                    else:
                        final_fp=pd.concat([final_fp,ind_imp],axis=0)
                        self.fp=self.fp+1
                    if enc_ob[i]==1 or enc_ob[i]==2:
                        if final_fp_obese.empty:
                            final_fp_obese=ind_imp
                        else:
                            final_fp_obese=pd.concat([final_fp_obese,ind_imp],axis=0)
                            self.fp_obese=self.fp_obese+1
                    else:
                        if final_fp_non.empty:
                            final_fp_non=ind_imp
                        else:
                            final_fp_non=pd.concat([final_fp_non,ind_imp],axis=0)
                            self.fp_non=self.fp_non+1
                    
                         
        with open('./data/output/imp/'+'tp_imp'+'_'+str(nbatch), 'wb') as w:
            pickle.dump(final_tp, w)
        with open('./data/output/imp/'+'tp_imp_non'+'_'+str(nbatch), 'wb') as w:
            pickle.dump(final_tp_non, w)
        with open('./data/output/imp/'+'tp_imp_obese'+'_'+str(nbatch), 'wb') as w:
            pickle.dump(final_tp_obese, w)
        with open('./data/output/imp/'+'tn_imp'+'_'+str(nbatch), 'wb') as w:
            pickle.dump(final_tn, w)
        with open('./data/output/imp/'+'fp_imp'+'_'+str(nbatch), 'wb') as w:
            pickle.dump(final_fp, w)
        with open('./data/output/imp/'+'fp_imp_obese'+'_'+str(nbatch), 'wb') as w:
            pickle.dump(final_fp_obese, w)
        with open('./data/output/imp/'+'fp_imp_non'+'_'+str(nbatch), 'wb') as w:
            pickle.dump(final_fp_non, w)
        with open('./data/output/imp/'+'fn_imp'+'_'+str(nbatch), 'wb') as w:
            pickle.dump(final_fn, w)
                                
                                
            #self.display_contri(ind_imp,ind_contri,truth,pred,prob,i)
        #print(contri[0,:,0:2])
    
    def interpret_read(self):
        final_tp=pd.DataFrame()
        final_fp=pd.DataFrame()
        final_fn=pd.DataFrame()
        final_tn=pd.DataFrame()
        final_tp_non=pd.DataFrame()
        final_tp_obese=pd.DataFrame()
        final_fp_non=pd.DataFrame()
        final_fp_obese=pd.DataFrame()
        
        for i in range(46):                        
            with open('./data/output/imp/'+'tp_imp'+'_'+str(i), 'rb') as w:
                ind_imp=pickle.load(w)
            if final_tp.empty:
                final_tp=ind_imp
            else:
                final_tp=pd.concat([final_tp,ind_imp],axis=0)
        print(final_tp['feat'].nunique())
        final_tp=final_tp.groupby('feat').agg({'imp':['sum','count']}).reset_index()    
        final_tp.columns = list(map(''.join, final_tp.columns.values))
        print(final_tp.head())
        self.display_contri(final_tp,"TP")
           
            
        for i in range(46):                        
            with open('./data/output/imp/'+'fp_imp'+'_'+str(i), 'rb') as w:
                ind_imp=pickle.load(w)
            if final_fp.empty:
                final_fp=ind_imp
            else:
                final_fp=pd.concat([final_fp,ind_imp],axis=0)
        print(final_fp['feat'].nunique())
        final_fp=final_fp.groupby('feat').agg({'imp':['sum','count']}).reset_index()
        final_fp.columns = list(map(''.join, final_fp.columns.values))
        self.display_contri(final_fp,"FP")
            
        for i in range(46):                        
            with open('./data/output/imp/'+'fn_imp'+'_'+str(i), 'rb') as w:
                ind_imp=pickle.load(w)
            if final_fn.empty:
                final_fn=ind_imp
            else:
                final_fn=pd.concat([final_fn,ind_imp],axis=0)
        print(final_fn['feat'].nunique())
        final_fn=final_fn.groupby('feat').agg({'imp':['sum','count']}).reset_index()
        final_fn.columns = list(map(''.join, final_fn.columns.values))
        self.display_contri(final_fn,"FN")
            
        for i in range(46):                        
            with open('./data/output/imp/'+'tn_imp'+'_'+str(i), 'rb') as w:
                ind_imp=pickle.load(w)
            if final_tn.empty:
                final_tn=ind_imp
            else:
                final_tn=pd.concat([final_tn,ind_imp],axis=0)
        print(final_tn['feat'].nunique())
        final_tn=final_tn.groupby('feat').agg({'imp':['sum','count']}).reset_index()  
        final_tn.columns = list(map(''.join, final_tn.columns.values))
        self.display_contri(final_tn,"TN")
            
        for i in range(46):                        
            with open('./data/output/imp/'+'tp_imp_obese'+'_'+str(i), 'rb') as w:
                ind_imp=pickle.load(w)
            if final_tp_obese.empty:
                final_tp_obese=ind_imp
            else:
                final_tp_obese=pd.concat([final_tp_obese,ind_imp],axis=0)
        print(final_tp_obese['feat'].nunique())
        final_tp_obese=final_tp_obese.groupby('feat').agg({'imp':['sum','count']}).reset_index()
        final_tp_obese.columns = list(map(''.join, final_tp_obese.columns.values))
        self.display_contri(final_tp_obese,"TP_obese")
            
        for i in range(46):                        
            with open('./data/output/imp/'+'tp_imp_non'+'_'+str(i), 'rb') as w:
                ind_imp=pickle.load(w)
            if final_tp_non.empty:
                final_tp_non=ind_imp
            else:
                final_tp_non=pd.concat([final_tp_non,ind_imp],axis=0)
        print(final_tp_non['feat'].nunique())
        final_tp_non=final_tp_non.groupby('feat').agg({'imp':['sum','count']}).reset_index()
        final_tp_non.columns = list(map(''.join, final_tp_non.columns.values))
        self.display_contri(final_tp_non,"TP_non")
            
        for i in range(46):                        
            with open('./data/output/imp/'+'fp_imp_obese'+'_'+str(i), 'rb') as w:
                ind_imp=pickle.load(w)
            if final_fp_obese.empty:
                final_fp_obese=ind_imp
            else:
                final_fp_obese=pd.concat([final_fp_obese,ind_imp],axis=0)
        print(final_fp_obese['feat'].nunique())
        final_fp_obese=final_fp_obese.groupby('feat').agg({'imp':['sum','count']}).reset_index()
        final_fp_obese.columns = list(map(''.join, final_fp_obese.columns.values))
        self.display_contri(final_fp_obese,"FP_obese")
            
        for i in range(46):                        
            with open('./data/output/imp/'+'fp_imp_non'+'_'+str(i), 'rb') as w:
                ind_imp=pickle.load(w)
            if final_fp_non.empty:
                final_fp_non=ind_imp
            else:
                final_fp_non=pd.concat([final_fp_non,ind_imp],axis=0)
        print(final_fp_non['feat'].nunique())
        final_fp_non=final_fp_non.groupby('feat').agg({'imp':['sum','count']}).reset_index()
        final_fp_non.columns = list(map(''.join, final_fp_non.columns.values))
        self.display_contri(final_fp_non,"FP_non")    
                                      
        obese=pd.concat([final_fp_obese,final_tp_obese],axis=0)
        non=pd.concat([final_fp_non,final_tp_non],axis=0)
        all_obese=pd.concat([final_fp,final_tp],axis=0)
        print(obese['feat'].nunique())
        print(non['feat'].nunique())
        print(all_obese['feat'].nunique())
        obese=obese.groupby('feat').agg({'imp':['sum','count']}).reset_index()
        obese.columns = list(map(''.join, obese.columns.values))
        non=non.groupby('feat').agg({'imp':['sum','count']}).reset_index()
        non.columns = list(map(''.join, non.columns.values))
        all_obese=all_obese.groupby('feat').agg({'imp':['sum','count']}).reset_index()   
        all_obese.columns = list(map(''.join, all_obese.columns.values))
        self.display_contri(obese,"obese")
        self.display_contri(non,"non")
        self.display_contri(all_obese,"all_obese")
        
            
        #print(contri[0,:,0:2])
    def model_test(self,total_test_hids,k):
        print("======= TESTING ========")
        
        n_batches=int(len(total_test_hids)/(args.batch_size))
        self.n_batches=n_batches
#         print(n_batches)
#         n_batches=2
        total_auroc_mat=np.zeros((8,8,args.labels))
        total_auprc_mat=np.zeros((8,8,args.labels))
        total_auprc_base=np.zeros((8,8,args.labels))
        total_samples=np.zeros((8,8,args.labels))
        total_curve=np.zeros((8,8,args.labels,9))
        total_all_curve=np.zeros((8,8,args.labels,9))
        
        
        
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
                enc_feat,enc_len,enc_age,enc_demo,enc_ob=self.encXY(test_hids,train_data=False)
                dec_feat,dec_labels,mask=self.decXY(test_hids,train_data=False)

                pred_mask=np.zeros((mask.shape[0],mask.shape[1]))
                if obs>0:
                    pred_mask[:,0:obs]=mask[:,0:obs]#mask right
    #             print(pred_mask[2])
                pred_mask=torch.tensor(pred_mask)
                pred_mask=pred_mask.type(torch.DoubleTensor)
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
                    enc_feat,enc_len,enc_age,enc_demo,enc_ob=enc_feat[obs_idx],enc_len[obs_idx],enc_age[obs_idx],enc_demo[obs_idx],enc_ob[obs_idx]
                    #print(enc_demo)


    #             print("dec_feat",dec_feat_pred[0,0:2,:])
    #             print("dec_labels",dec_labels_pred[0])    
                output,prob,disc_input,logits = self.net(False,False,enc_feat,enc_len,enc_age,enc_demo,dec_feat_pred,dec_labels_pred,pred_mask)
    #             print("pred_mask",pred_mask[0:3])
    #             print("prob",prob[0][0:10])
    #             print("output",output[0])
                #print(logits.shape)
                #print(prob[0].shape)
                #print(logits[0:5])
                #d = {0:0, 1:0, 2:1, 6:2, 13:3,25:4,33:5,174:6,39:7,42:8,44:9,173:10 }
                d = {0:0, 175:0,176:1,177:2}
                #d = {0:0, 175:0,177:1}
                mask=np.asarray(mask)

                #print(mask.shape)

                auroc_mat=np.zeros((8,args.labels))
                auprc_mat=np.zeros((8,args.labels))
                auprc_base=np.zeros((8,args.labels))
                n_samples=np.zeros((8,args.labels))
                
                
                obs_mask=np.zeros((mask.shape[0],mask.shape[1]))
                obs_mask[:,obs:]=mask[:,obs:]#mask left side
                #print(enc_ob.shape)
                eth,race,gender=enc_demo[:,0],enc_demo[:,1],enc_demo[:,2]
                eth,race,gender,ob_stat=np.asarray(eth),np.asarray(race),np.asarray(gender),np.asarray(enc_ob)
                    
                for i in range(obs,len(prob)):#time
                    dec_labels_dec=torch.tensor([d[x.item()] for x in dec_labels[:,i]])
                    m=obs_mask[:,i]
                    #print(m.shape)
                    idx=list(np.where(m == 0)[0])
                    #print(len(idx))
                    logits_dec=logits[:,i,:]
                    logits_dec=logits_dec.squeeze()
                    for l in range(args.labels):#class
                        
                        if l==1:
                            dec_labels_l=[1 if (y==1) or (y==2) else 0 for y in dec_labels_dec]
                        else:
                            dec_labels_l=[1 if y==l else 0 for y in dec_labels_dec]
                        #print("dec_labels_l",len(dec_labels_l))
                        #print("=========================================================")
                        prob_l=prob[i][:,l].data.cpu().numpy()
                        #print(logits_dec.shape)
#                         if l==1 or l==0:
#                             logits_l=logits_dec[:,0].data.cpu().numpy()
#                         elif l==2:
#                             logits_l=logits_dec[:,1].data.cpu().numpy()
                        logits_l=logits_dec.data.cpu().numpy()
                        #print(len(prob_l))
                        #print(len(dec_labels_l))
                        
                        prob_l,dec_labels_l,logits_l=np.asarray(prob_l),np.asarray(dec_labels_l),np.asarray(logits_l)
                        prob_l=np.delete(prob_l, idx)
                        dec_labels_l=np.delete(dec_labels_l, idx)
                        logits_l=np.delete(logits_l, idx)
                        race_l=np.delete(race, idx)
                        eth_l=np.delete(eth, idx)
                        gender_l=np.delete(gender, idx)
                        ob_l=np.delete(ob_stat, idx)
#                         print("prob",prob_l.shape)
#                         print("dec",dec_labels_l.shape)
#                         print("logits_l",logits_l.shape)
                        n_samples[i,l]=prob_l.shape[0]
                        
                        path='data/output/'+str(k)+'/'+str(obs)+'/'+str(i)+'/'+str(l)
                        if not os.path.exists(path):
                            os.makedirs(path)
                        with open('./'+path+'/'+'ob_list'+'_'+str(nbatch), 'wb') as fp:
                                   pickle.dump(ob_l, fp)
                        with open('./'+path+'/'+'race_list'+'_'+str(nbatch), 'wb') as fp:
                                   pickle.dump(race_l, fp)
                        with open('./'+path+'/'+'eth_list'+'_'+str(nbatch), 'wb') as fp:
                                   pickle.dump(eth_l, fp)
                        with open('./'+path+'/'+'gender_list'+'_'+str(nbatch), 'wb') as fp:
                                   pickle.dump(gender_l, fp)
                        with open('./'+path+'/'+'logits_list'+'_'+str(nbatch), 'wb') as fp:
                                   pickle.dump(logits_l, fp)
                                
                        with open('./'+path+'/'+'prob_list'+'_'+str(nbatch), 'wb') as fp:
                                   pickle.dump(prob_l, fp)
                        with open('./'+path+'/'+'dec_list'+'_'+str(nbatch), 'wb') as fp:
                                   pickle.dump(dec_labels_l, fp)
                        
#                         auroc_mat[i,l],auprc_mat[i,l],auprc_base[i,l],n_curve[i,l],n_all_curve[i,l]=self.loss(prob_l,dec_labels_l)
            
#                 #print(n_curve)
#                 nan_mask = np.isnan(auroc_mat)
#                 nan_mask=1-nan_mask
# #                 print(auroc_mat)
#                 auroc_mat[np.isnan(auroc_mat)] = 0
# #                 print(auroc_mat)
#                 n_curve[np.isnan(n_curve)] = 0
#                 n_all_curve[np.isnan(n_curve)] = 0
# #                 print(nan_mask)
#                 n_samples=n_samples*nan_mask
#                 n_samples[n_samples>0]=1
#                 auprc_mat=auprc_mat*nan_mask
#                 auprc_base=auprc_base*nan_mask
# #                 print(n_samples)
#                 total_samples[obs]+=n_samples
#                 total_auroc_mat[obs]+=auroc_mat
#                 total_auprc_mat[obs]+=auprc_mat
#                 total_auprc_base[obs]+=auprc_base
#                 total_curve[obs]+=n_curve
#                 total_all_curve[obs]+=n_all_curve
# #                 print(total_samples[obs])
# #                 print(total_auroc_mat[obs])
#                 #print(total_curve[obs])

#         total_auroc_mat=np.divide(total_auroc_mat, total_samples)
#         total_auprc_mat=np.divide(total_auprc_mat, total_samples)
#         total_auprc_base=np.divide(total_auprc_base, total_samples)
        
#         #print(total_samples)
#         c_total_samples=np.expand_dims(total_samples, axis=3)
#         c_total_samples=np.repeat(c_total_samples,9,axis=3)
#         #print(c_total_samples)
#         total_curve=np.divide(total_curve, c_total_samples)
#         total_all_curve=np.divide(total_all_curve, c_total_samples)
        
        

        #for time in range(8):
            #self.save_output(total_auroc_mat[time],total_auprc_mat[time],total_auprc_base[time],time,k,False)
            #self.display_output(total_auroc_mat[time],total_auprc_mat[time],total_auprc_base[time],total_curve[time],time,k,False)

        
    
    def test_read(self,k):
        print("======= TESTING ========")
        for obs in range(8):
#                 print("======= OBS {:.1f} ========".format(obs+2))
            
            auroc_mat=np.zeros((8,args.labels))
            auprc_mat=np.zeros((8,args.labels))
            auprc_base=np.zeros((8,args.labels))
            micro=np.zeros((8,args.labels-1))
            
            for i in range(obs,8):#time

                for l in range(args.labels):#class
                    prob_l=[]
                    race_l=[]
                    ob_l=[]
                    eth_l=[]
                    gender_l=[]
                    logits_l=[]
                    dec_labels_l=[]
                    for nbatch in range(45):
                        path='data/output/'+str(k)+'/'+str(obs)+'/'+str(i)+'/'+str(l)
                        
                        with open('./'+path+'/'+'ob_list'+'_'+str(nbatch), 'rb') as fp:
                                   ob_l.extend(pickle.load(fp))
                        with open('./'+path+'/'+'race_list'+'_'+str(nbatch), 'rb') as fp:
                                   race_l.extend(pickle.load(fp))
                        with open('./'+path+'/'+'eth_list'+'_'+str(nbatch), 'rb') as fp:
                                   eth_l.extend(pickle.load(fp))
                        with open('./'+path+'/'+'gender_list'+'_'+str(nbatch), 'rb') as fp:
                                   gender_l.extend(pickle.load(fp))
                                
                        with open('./'+path+'/'+'logits_list'+'_'+str(nbatch), 'rb') as fp:
                                   logits_l.extend(pickle.load(fp))
                                
                        with open('./'+path+'/'+'prob_list'+'_'+str(nbatch), 'rb') as fp:
                                   prob_l.extend(pickle.load(fp))
                        with open('./'+path+'/'+'dec_list'+'_'+str(nbatch), 'rb') as fp:
                                   dec_labels_l.extend(pickle.load(fp))
                    #print(prob_l)
                    
                    
                    prob_l=np.asarray(prob_l)
                    logits_l=np.asarray(logits_l)
                    dec_labels_l=np.asarray(dec_labels_l)
                    ob_l=np.asarray(ob_l)
                    if l>0:
                        micro[i,l-1]=dec_labels_l.sum()
                    
                    if len(prob_l)>0 and l>0:
#                         print(ob_l)                        
#                         ob_l[ob_l==0]=20
#                         ob_l[ob_l==1]=50
#                         ob_l[ob_l==2]=80
#                         ob_l=ob_l/100
#                         ob_l[ob_l==1]=0
#                         ob_l[ob_l==2]=1
#                         print(ob_l)
                        auroc_mat[i,l],auprc_mat[i,l],auprc_base[i,l]=self.loss(prob_l,dec_labels_l,logits_l,ob_l)   
                        #fairness.fairness_evaluation(race_l,eth_l,gender_l,ob_l,prob_l,dec_labels_l)
                    else:
                        auroc_mat[i,l],auprc_mat[i,l],auprc_base[i,l]=np.nan,np.nan,np.nan
                        
                    #print(curve)
#                     plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],curve,color='blue',marker='o')
#                     plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],all_curve,color='red',marker='o')
#                     plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],[0,0,0,0,0,0,0,0,0],color='red',marker='o')
#                     plt.show()
            self.display_output(auroc_mat,auprc_mat,auprc_base,obs,k,micro,False)
            

       
            
#             if time==0:
#                 print(total_curve[time,2,9])

#                 plt.show()
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
        
        
    def encXY(self,ids,train_data):

        if train_data:
            enc1=pd.read_csv('./data/3/enc_train.csv',header=0)
            enc_len1=pd.read_csv('./data/3/lengths_train.csv',header=0)
            demo1=pd.read_csv('./data/3/demo_train.csv',header=0)
            
            demo1=demo1.groupby('person_id').last().reset_index()
            enc=pd.DataFrame()
            enc_len=pd.DataFrame()
            demo=pd.DataFrame()
            for i in ids:
                enc=pd.concat([enc,enc1[enc1['person_id']==i]],axis=0)
                demo=pd.concat([demo,demo1[demo1['person_id']==i]],axis=0)
                enc_len=pd.concat([enc_len,enc_len1[enc_len1['index']==i]],axis=0)
            
        else:
            enc=pd.read_csv('./data/3/enc_test.csv',header=0)
            enc_len=pd.read_csv('./data/3/lengths_test.csv',header=0)
            demo=pd.read_csv('./data/3/demo_test.csv',header=0)
            demo=demo.groupby('person_id').last().reset_index()
            enc=enc[enc['person_id'].isin(ids)]
            demo=demo[demo['person_id'].isin(ids)]
            enc_len=enc_len[enc_len['index'].isin(ids)]
        
        #print(enc.shape)
        #enc=pd.read_csv('./input_data_med.csv',header=0)
        #enc_len=pd.read_csv('./enc_lengths_med.csv',header=0)
        #print(enc_len)
        
        enc_ob=pd.read_csv('./data/3/enc_labels.csv',header=0)
        enc_ob=enc_ob.sort_values(by=['person_id','Age'])
        enc_ob=enc_ob.dropna()
        enc_ob=enc_ob.groupby('person_id').last().reset_index()
        enc_ob=enc_ob[enc_ob['person_id'].isin(ids)]
        enc_ob.loc[enc_ob.label<85,'label']=0
        enc_ob.loc[enc_ob.label>=95,'label']=2
        enc_ob.loc[(enc_ob.label>=85) & (enc_ob.label<95),'label']=1
        
        #demo=demo.groupby('person_id').last().reset_index()
        #print(demo.shape)
    
        #print(enc_len.shape)
    
        
        enc_feat=enc['feat_dict'].values
        enc_eth=demo['Eth_dict'].values
        enc_race=demo['Race_dict'].values
        enc_sex=demo['Sex_dict'].values
        enc_len=enc_len['person_id'].values
        enc_ob=enc_ob['label'].values
        
        enc_age=enc['age_dict'].values
        
        dic={350:0,351:1,352:2,353:3,354:4,355:5,356:6,357:7,358:8}
        enc_eth=torch.tensor([dic[x.item()] for x in enc_eth])
        enc_race=torch.tensor([dic[x.item()] for x in enc_race])
        enc_sex=torch.tensor([dic[x.item()] for x in enc_sex])
        
        #Reshape to 3-D
        #print(enc_feat.shape)
        enc_feat=torch.tensor(enc_feat)
        enc_feat=torch.reshape(enc_feat,(len(ids),-1))
        enc_feat=enc_feat.type(torch.LongTensor)
        
        enc_len=torch.tensor(enc_len)
        #enc_len=torch.reshape(enc_len,(len(ids),-1))
        enc_len=enc_len.type(torch.LongTensor)
        
        enc_ob=torch.tensor(enc_ob)
        enc_ob=enc_ob.type(torch.LongTensor)
        
        enc_age=torch.tensor(enc_age)
        enc_age=torch.reshape(enc_age,(len(ids),-1))
        enc_age=enc_age.type(torch.LongTensor)
        
        enc_eth=torch.tensor(enc_eth)
        enc_eth=enc_eth.unsqueeze(1)
        enc_race=torch.tensor(enc_race)
        enc_race=enc_race.unsqueeze(1)
        enc_sex=torch.tensor(enc_sex)
        enc_sex=enc_sex.unsqueeze(1)
        #print(enc_eth.shape)
        #print(enc_sex)
        enc_demo=torch.cat((enc_eth,enc_race),1)
        enc_demo=torch.cat((enc_demo,enc_sex),1)
        enc_demo=enc_demo.type(torch.LongTensor)
        #print(enc_demo.shape)
        #print(enc_demo)

        return enc_feat,enc_len, enc_age, enc_demo,enc_ob
            
           
    def decXY(self,ids,train_data):
        #print("decoder")
        if train_data:
            dec1=pd.read_csv('./data/3/dec_train.csv',header=0)
            labels1=pd.read_csv('./data/3/labels_train.csv',header=0)
            
            del dec1['age_dict']
            dec=pd.DataFrame()
            labels=pd.DataFrame()
            
            for i in ids:
                dec=pd.concat([dec,dec1[dec1['person_id']==i]],axis=0)
                labels=pd.concat([labels,labels1[labels1['person_id']==i]],axis=0)
                
                
        else:
            dec=pd.read_csv('./data/3/dec_test.csv',header=0)
            labels=pd.read_csv('./data/3/labels_test.csv',header=0)
            del dec['age_dict']
            dec=dec[dec['person_id'].isin(ids)]
            labels=labels[labels['person_id'].isin(ids)]
        
        dec_feat=dec.iloc[:,2:].values
        #print(list(dec['person_id']))
        
        dec_labels=labels['label'].values
        #print(dec_labels)
        dec_labels[dec_labels==176]=175
        #print(dec_labels)
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
        
        #print(dec_feat.shape)
        #print(dec_feat[0:5])
#         print(dec_labels.shape)
#         print(dec_labels)
        
#         print(mask)
        return dec_feat,dec_labels,mask
    
    
    
    def create_model(self):
        self.net = model.EncDec2(self.device,
                           self.feat_vocab_size,
                           self.age_vocab_size,
                           self.demo_vocab_size,
                           embed_size=args.embedding_size,rnn_size=args.rnn_size,
                           batch_size=args.batch_size) 
        
            
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lrn_rate)#,weight_decay=1e-5
        #self.criterion = model.BCELossWeight(self.device)
        self.criterion = nn.BCELoss(reduction='sum')
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
            
    
    def display_output(self,auroc_mat,auprc_mat,auprc_base,obs,k,micro,full):
        
#         print(auroc_mat)
        
        def plot(mat,base,full,title):
            #label=['3%','5%','10%','25%','50%','75%','85%','90%','95%','97%','>97%']
            #print(mat)
            #print(micro)
            temp_mat=mat[:,1:]
            normed_matrix = normalize(micro, axis=1, norm='l1')
            #print(normed_matrix)
            normed_matrix=np.multiply(temp_mat,normed_matrix)
            #print(normed_matrix)
            print(np.sum(normed_matrix,axis=1))
            
            x = np.arange(8)  # the label locations
            barWidth = 0.2  # the width of the bars

            plt.rcParams["figure.figsize"] = (10,3)
           
        
            r1 = np.arange(len(mat[:,0]))
            n_class_labels=['Not Overweight/Obese','Overweight','Obese']
            plt.bar(r1, mat[:,0], width=barWidth, label=n_class_labels[0])
            
            for n_class in range(1,args.labels):
                r1=[x + barWidth for x in r1]
                plt.bar(r1, mat[:,n_class], width=barWidth, label=n_class_labels[n_class])
            
                
                
                
            
            
            if title=='AUPRC':
                r1 = np.arange(len(base[:,0]))                
                plt.bar(r1, base[:,0], color='black', width=barWidth)
                for n_class in range(1,args.labels):
                    r1=[x + barWidth for x in r1]
                    plt.bar(r1, base[:,n_class], color='black', width=barWidth)
           
            #for x, y in zip(list(np.arange(0,24,0.2)), mat.reshape(-1)):
            #    plt.annotate(str(round(y,2)), (x,y+0.06))
            print(mat)   
            #print(zip(list(range(0,24)), mat.reshape(-1)))
            #plt.plot(np.nanmean(mat,axis=1),marker='o')
            for x, y in zip(list(range(0,8)), np.nanmean(temp_mat,axis=1)):
                plt.annotate(str(round(y,2)), (x,y+0.09))

            # Add some text for labels, title and custom x-axis tick labels, etc.
            plt.ylabel(title, fontsize=12)
            if full:
                plt.title('No obs and pred defined', fontsize=12)
            else:
                plt.title('Observation Window 0 to '+str(obs+2), fontsize=12)
            plt.xticks([r + barWidth for r in range(len(mat[:,0]))], ['3', '4', '5', '6', '7','8','9','10'], fontsize=18)
            plt.legend(loc="center right", bbox_to_anchor=(1.1,0.5), fontsize=8)
            
            if full:
                plt.savefig('./data/output/'+str(k)+'/'+title+'_'+'full'+'.png')
            else:
                plt.savefig('./data/output/'+str(k)+'/'+title+'_'+str(obs)+'.png')
            plt.show()
            
        
        plot(auroc_mat,auprc_base,full,title="AUROC")
        plot(auprc_mat,auprc_base,full,title="AUPRC")
    def display_output_old(self,auroc_mat,auprc_mat,auprc_base,obs,k,full):
        
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
    
    def display_contri(self,ind_imp,title):
        n=int(ind_imp.shape[0]/2)
        ind_imp=ind_imp.sort_values(by='impcount',ascending=False)
        ind_imp_high=ind_imp[:30]
        ind_imp_high=ind_imp_high.sort_values(by='impsum',ascending=False)
        ind_imp_high=ind_imp_high[ind_imp_high['feat']!=0]
        
        
        plt.rcParams["figure.figsize"] = (20,5)
        ind_imp_high["label"] = ind_imp_high["feat"].astype(str)
        plt.bar(ind_imp_high['label'], ind_imp_high['impsum'])
        plt.xticks(rotation=90)
        plt.axhline(y=0)
        plt.title(title, fontsize=20)
        plt.show()
        
        
        ind_imp_mid=ind_imp[n:n+30]
        ind_imp_mid=ind_imp_mid.sort_values(by='impsum',ascending=False)
        ind_imp_mid=ind_imp_mid[ind_imp_mid['feat']!=0]
        
        
        plt.rcParams["figure.figsize"] = (20,5)
        ind_imp_mid["label"] = ind_imp_mid["feat"].astype(str)
        plt.bar(ind_imp_mid['label'], ind_imp_mid['impsum'])
        plt.xticks(rotation=90)
        plt.axhline(y=0)
        plt.title(title, fontsize=20)
        plt.show()
        
        ind_imp_low=ind_imp[(n*2)-30:]
        ind_imp_low=ind_imp_low.sort_values(by='impsum',ascending=False)
        ind_imp_low=ind_imp_low[ind_imp_low['feat']!=0]
        
        
        plt.rcParams["figure.figsize"] = (20,5)
        ind_imp_low["label"] = ind_imp_low["feat"].astype(str)
        plt.bar(ind_imp_low['label'], ind_imp_low['impsum'])
        plt.xticks(rotation=90)
        plt.axhline(y=0)
        plt.title(title, fontsize=20)
        plt.show()
              
        
#     def display_contri(self,ind_imp,ind_contri,truth,pred,prob,ids):
#         #print(ind_imp)
#         ind_imp=ind_imp[ind_imp['feat']!=0]
#         ind_contri=ind_contri[ind_contri['feat']!=0]
#         print("Ground Truth BMIp",truth)
#         print("Predicted BMIp",pred)
#         #print("Predicted Prob")
#         prob_l=[]
#         for i in range(len(prob)):
#             prob_l.append(prob[i][ids].data.cpu().numpy())
#         #print(len(prob_l))
#         #print(prob_l[0])
# #         print(ind_imp)
# #         print(ind_contri)

#         plt.rcParams["figure.figsize"] = (10,3)
#         barWidth=0.2
#         n_class_labels=['Not Overweight/Obese','Overweight','Obese']
        

#         for time in range(0,len(prob_l)):
#             barlist=plt.bar([time-barWidth,time,time+barWidth], prob_l[time], width=barWidth)
#             barlist[0].set_color('g')
#             barlist[1].set_color('b')
#             barlist[2].set_color('r')
#         plt.title('Predicetd Probability', fontsize=12)
#         plt.ylabel('Predicetd Probability', fontsize=12)
#         plt.xticks([r + barWidth for r in range(len(prob_l))], ['3', '4', '5', '6', '7','8','9','10'], fontsize=18)
#         plt.xlabel('Age (in years)', fontsize=12)
#         plt.show()       
        
        
#         plt.rcParams["figure.figsize"] = (10,3)
#         ind_imp["label"] = ind_imp["feat"].astype(str)
#         plt.bar(ind_imp['label'], ind_imp['imp'])
#         plt.xticks(rotation=20)
#         plt.axhline(y=0)
#         plt.title('Individual Average Contribution', fontsize=12)
#         plt.show()
        
#         plt.rcParams["figure.figsize"] = (10,3)
#         ind_contri["label"] = ind_contri["feat"].astype(str) + '_' + ind_contri["age"].astype(str)
#         plt.bar(ind_contri['label'], ind_contri['imp'])
#         plt.xticks(rotation=20)
#         plt.axhline(y=0)
#         plt.title('Individual Contribution over Timeline', fontsize=12)
#         plt.show()
          


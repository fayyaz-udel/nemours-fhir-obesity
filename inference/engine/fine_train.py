#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import pandas as pd
import numpy as np
import torch as T
import torch
import torch.nn as nn
from torch import optim
import importlib

import evaluation
import parameters
from parameters import *
#import model as model
import mimic_model as model
import random

#from imblearn.over_sampling import RandomOverSampler

#import torchvision.utils as utils
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
from parameters import *
importlib.reload(evaluation)


class Fine_models():
    def __init__(self,model_name,feat_vocab_size,age_vocab_size,demo_vocab_size):
        
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

     
        print("===============MODEL FINE TUNING===============")
        
        self.dl_train()
            
        
    def create_kfolds(self):
        train_ids=pd.read_csv('./data/3/demo_train.csv',header=0)
        test_ids=pd.read_csv('./data/3/demo_test.csv',header=0)         
        return list(train_ids['person_id'].unique()),list(test_ids['person_id'].unique())


    def dl_train(self):
        train_hids,test_hids=self.create_kfolds()
#         print(len(k_hids))
#         print(k_hids[0].shape)
              
        for i in range(1):
            path=self.save_path+'_'+str(i)+".tar"
            self.net=torch.load(path)
            print("[ MODEL LOADED ]")
            
            
            self.optimizer = optim.Adam(self.net.parameters(), lr=args.lrn_rate)
            self.criterion = nn.BCELoss(reduction='sum')
            self.kl_loss = nn.KLDivLoss()
            self.net.to(self.device)
        
            print(self.net)
            
            path="saved_models/"+self.model_name+'_'+'fine'+'_'+str(i)+".tar"
            print("==================={0:2d} FOLD=====================".format(i))
            #self.save_path="saved_models/"+self.model_name+"_"+str(i)+".tar"
            
            
            val_hids=random.sample(train_hids,int(len(train_hids)*0.05))
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
                    
                    
                    batch_loss = self.train_model(nbatch,train_hids)
#                     batch_loss=0
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
#                 print(len(val_hids))
#                 val_loss1=self.model_val(val_hids[0:200])
#                 val_loss2=self.model_val(val_hids[200:400])
#                 val_loss3=self.model_val(val_hids[400:600])
#                 val_loss4=self.model_val(val_hids[600:800])
#                 val_loss5=self.model_val(val_hids[800:1000])
#                 val_loss6=self.model_val(val_hids[1000:1200])
#                 val_loss7=self.model_val(val_hids[1200:1400])
#                 #val_loss8=self.model_val(val_hids[1400:])
#                 val_loss=(val_loss1+val_loss2+val_loss3+val_loss4+val_loss5+val_loss6+val_loss7)/7
#                 print("Total Val Loss: ", val_loss)
#                 #print("Updating Model")
#                 #T.save(self.net,self.save_path)
                if (epoch<4):
                    
                    print("Updating Model")
                    T.save(self.net,path)
#                 if(val_loss<=min_loss) or (epoch<4):
#                     print("Validation results improved")
#                     min_loss=val_loss
#                     print("Updating Model")
#                     T.save(self.net,path)
#                     counter=0
#                 else:
#                     print("No improvement in Validation results")
#                     counter=counter+1
            
    
    def train_model(self,nbatch,train_hids):
        #print("mask",mask.shape)
        self.optimizer.zero_grad()
        batch_loss=0.0
        print("======= Bathch {:.1f} ========".format(nbatch))
        for obs in range(2,3):
            
            #print("======= OBS {:.1f} ========".format(obs+2))
            enc_feat,enc_len,enc_age,enc_demo,enc_ob=self.encXY(train_hids[nbatch*args.batch_size:(nbatch+1)*args.batch_size],True)
            dec_feat,dec_labels,mask=self.decXY(train_hids[nbatch*args.batch_size:(nbatch+1)*args.batch_size],True)
            
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
                #print("obs_idx",obs_idx.shape)
                obs_idx=torch.nonzero(obs_idx>=0)
                #print("obs_idx",obs_idx.shape)
                obs_idx=obs_idx.squeeze()
                dec_feat_pred=dec_feat_pred[obs_idx]
                dec_labels_pred=dec_labels_pred[obs_idx]
                pred_mask=pred_mask[obs_idx]
                mask=mask[obs_idx]
                dec_labels=dec_labels[obs_idx]
                print(obs_idx)
                print(enc_demo.shape)
                print(enc_feat.shape)
                enc_feat,enc_len,enc_age,enc_demo,enc_ob=enc_feat[obs_idx],enc_len[obs_idx],enc_age[obs_idx],enc_demo[obs_idx],enc_ob[obs_idx]
            
            output,prob,disc_input,_ = self.net(False,False,enc_feat,enc_len,enc_age,enc_demo,dec_feat_pred,dec_labels_pred,pred_mask) 
                
                
            #d = {0:1, 1:2, 2:6, 3:13,4:25,5:33,6:174,7:39,8:42,9:44,10:173 }
            #d = {0:0, 1:0, 2:1, 6:2, 13:3,25:4,33:5,174:6,39:7,42:8,44:9,173:10 }
            d = {0:0, 175:0,176:1,177:2}
            #dec_labels=torch.tensor([d[x.item()] for x in dec_labels])
    #         print(prob)
            out_loss=0.0
            x_loss=0.0

        
            for i in range(obs,len(prob)):
                idx=torch.nonzero(mask[:,i]>0)
                idx=idx.squeeze()
                m=mask[idx]
                m_prob=prob[i][idx]
                m_output=output[i][idx]
    #             print(dec_labels.shape)
                dec_labels_dec=dec_labels[idx]
    #             print(dec_labels_dec.shape)
                dec_labels_dec=torch.tensor([d[x.item()] for x in dec_labels_dec[:,i]])
                dec_labels_sig=torch.zeros(dec_labels_dec.shape[0],args.labels)
                for x in range(dec_labels_dec.shape[0]):
                    dec_labels_sig[x,dec_labels_dec[x]]=1
            
                dec_labels_dec=dec_labels_dec.to(self.device)
                
                dec_labels_sig[:,1]=dec_labels_sig[:,1]+dec_labels_sig[:,2]
                dec_labels_sig=dec_labels_sig.to(self.device)
                
                out_loss+=(self.criterion(m_prob[:,1:],dec_labels_sig[:,1:]))/(m[:,i].sum()+ 1e-5)
    #             print("ce loss",out_loss)
                #inter_loss=torch.sum(torch.abs(dec_labels_dec - torch.tensor(m_output).to(self.device))*torch.tensor(m[:,i]).to(self.device))

                #x_loss +=  inter_loss/ (torch.sum(m[:,i]) + 1e-5)
    #             print("rmse loss",x_loss)
            out_loss=out_loss/len(prob)
            #x_loss=x_loss/len(prob)
    #         print("total CE loss",out_loss)
    #         print("total RMSE loss",x_loss)
            total_loss=out_loss#+x_loss#+k_loss
    #         print(out_loss+x_loss)
            # calculate the gradients
            total_loss.backward()
            # update the parameters of the model
            self.optimizer.step()
        
            batch_loss+=total_loss.item()
        return batch_loss
    
    def model_val(self,val_hids):
        #print("======= VALIDATION ========")
        #val_hids=val_hids[0:500]
        self.net.eval()
        
        val_loss=0.0
        
        for obs in range(2,3):
            #print("======= OBS {:.1f} ========".format(obs+2))
            enc_feat,enc_len,enc_age,enc_demo,enc_ob=self.encXY(val_hids,True)
            dec_feat,dec_labels,mask=self.decXY(val_hids,True)
#             print(enc_feat.shape)
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
                enc_feat,enc_len,enc_age,enc_demo,enc_ob=enc_feat[obs_idx],enc_len[obs_idx],enc_age[obs_idx],enc_demo[obs_idx],enc_ob[obs_idx]
                
#             print(enc_feat.shape,enc_len.shape,enc_age.shape,dec_feat_pred.shape,dec_labels_pred.shape,pred_mask.shape)
            output,prob,disc_input,_ = self.net(False,False,enc_feat,enc_len,enc_age,enc_demo,dec_feat_pred,dec_labels_pred,pred_mask) 

            #d = {0:0, 1:0, 2:1, 6:2, 13:3,25:4,33:5,174:6,39:7,42:8,44:9,173:10 }
            d = {0:0, 175:0,176:1,177:2}
            out_loss=0.0
            x_loss=0.0


            for i in range(obs,len(prob)):
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
            
                dec_labels_dec=dec_labels_dec.to(self.device)
                dec_labels_sig[:,1]=dec_labels_sig[:,1]+dec_labels_sig[:,2]
                dec_labels_sig=dec_labels_sig.to(self.device)
                
                out_loss+=(self.criterion(m_prob[:,1:],dec_labels_sig[:,1:]))/(m[:,i].sum()+ 1e-5)
            out_loss=out_loss/len(prob)
            #x_loss=x_loss/len(prob)
            val_loss+=out_loss.item()#+x_loss.item()#+k_loss       
                    
        return val_loss
    
    
        
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
        #dec_labels[dec_labels==176]=175
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
    



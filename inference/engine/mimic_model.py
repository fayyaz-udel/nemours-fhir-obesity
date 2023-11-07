#import jsondim
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import *
import importlib
import parameters

importlib.reload(parameters)
from parameters import *

class EncDec(nn.Module):
    def __init__(self,device,feat_vocab_size,age_vocab_size,embed_size,rnn_size,batch_size):
        super(EncDec, self).__init__()
        self.embed_size=embed_size
        self.latent_size=args.latent_size
        self.rnn_size=rnn_size
        self.feat_vocab_size=feat_vocab_size
 
        self.age_vocab_size=age_vocab_size
        
        self.batch_size=batch_size
        self.padding_idx = 0
        self.device=device
        self.build()
        
    def build(self):
        
        self.emb_feat=FeatEmbed(self.device,self.feat_vocab_size,self.embed_size,self.batch_size) 
        
        self.emb_age=AgeEmbed(self.device,self.age_vocab_size,self.embed_size,self.batch_size) 
#         self.disc=Discriminator(self.device)
        self.enc=Encoder(self.device,self.feat_vocab_size,self.age_vocab_size,self.embed_size,self.rnn_size,self.batch_size)
        self.dec=Decoder(self.device,self.feat_vocab_size,self.age_vocab_size,self.embed_size,self.rnn_size,self.emb_feat,self.batch_size)
        

        
        
    def forward(self,find_contri,enc_feat,enc_len,enc_age,dec_feat,labels,mask):   
        
        contri=torch.cat((enc_feat.unsqueeze(2),enc_age.unsqueeze(2)),2)
#         print("Enc featEmbed",contri.shape)
#         print("Enc featEmbed",contri[0])
        enc_feat=self.emb_feat(enc_feat)
        enc_age=self.emb_age(enc_age)
#         print("Enc featEmbed",enc_feat.shape)
#         print("Enc ageEmbed",enc_age.shape)
   
        code_pool,code_indices,code_h_n,code_c_n=self.enc(enc_feat,enc_len,enc_age)
#         print("pool",code_pool.shape)
#         print("code_indices",code_indices.shape)
        #print("========================")
        
        #===========DECODER======================
        dec_feat=self.emb_feat(dec_feat)
        dec_feat=torch.sum(dec_feat, 2)
#         print("Dec featEmbed",dec_feat.shape)
        dec_labels=self.emb_feat(labels)
#         print("Dec LabelEmbed",dec_labels.shape)
        
        if find_contri: 
            dec_output,dec_prob,disc_input,kl_input,all_contri=self.dec(find_contri,contri,dec_feat,dec_labels,code_pool,code_indices,code_h_n,code_c_n,mask,labels)
        else:
            dec_output,dec_prob,disc_input,kl_input=self.dec(find_contri,contri,dec_feat,dec_labels,code_pool,code_indices,code_h_n,code_c_n,mask,labels)
         
            
        kl_input=torch.tensor(kl_input)
#         print(len(disc_input))
#         print(disc_input[0:5])
#         print("------------------")
#         print(disc_input[195:202])
        disc_input=torch.stack(disc_input)
#         print(disc_input.shape)
#         print(disc_input[0:5])
#         print(disc_input[198:205])
        disc_input=torch.reshape(disc_input,(8,-1,disc_input.shape[1]))
#         print(disc_input.shape)
#         print(disc_input[:,0,:])
        disc_input=disc_input.permute(1,0,2)
#         self.disc(disc_input,mask,labels)
    
#         print(disc_input.shape)
#         print(disc_input[0,:,:])
#         print(disc_input[0:10])
#         print(disc_input[195:205])
        kl_input=torch.reshape(kl_input,(8,-1))
#         print(disc_input)
        kl_input=kl_input.permute(1,0)
#         print(disc_input.shape)
#         print(disc_input[0])
#         print(disc_input[1])
        #print(len(dec_prob))
        #print(dec_prob)
        #dec_prob=torch.stack(dec_prob)
        #print(dec_prob)
        
        #print("===================================================")
        
#         print(dec_prob[0].shape)
        #dec_output=torch.tensor(dec_output)
#         print("dec_output",dec_output.shape)
#         dec_output=dec_output.permute(1,0)
        #print("dec_output",dec_output)
#         print("dec_output",dec_output[0])
        #dec_prob=torch.tensor(dec_prob)
#         print("dec_prob",dec_prob.shape)
#         print(dec_prob[:,0,:])

#         dec_prob=dec_prob.permute(1,0,2)
#         print("dec_prob",dec_prob.shape)
#         print(dec_prob[0])

#         print("dec_output",dec_output.shape)
        if find_contri: 
            return dec_output,dec_prob,kl_input,all_contri
        else:
            return dec_output,dec_prob,kl_input
    
    
 
    
class Encoder(nn.Module):
    def __init__(self,device,feat_vocab_size,age_vocab_size,embed_size,rnn_size,batch_size):
        super(Encoder, self).__init__()
        self.embed_size=embed_size
#         self.latent_size=args.latent_size
        self.rnn_size=rnn_size
        self.feat_vocab_size=feat_vocab_size
 
        self.age_vocab_size=age_vocab_size
        
        
        self.padding_idx = 0
        self.device=device
        self.build()
        
    def build(self):

        
        self.rnn=nn.LSTM(input_size=self.embed_size*2,hidden_size=self.rnn_size,num_layers = args.rnnLayers,batch_first=True)
        self.code_max = nn.AdaptiveMaxPool1d(1, True)
        
 
        
    def forward(self,featEmbed,lengths,ageEmbed):   

        out1=torch.cat((featEmbed,ageEmbed),2)
        
#         print("out",out1.shape)
        
        out1=out1.type(torch.FloatTensor)
        out1=out1.to(self.device)
        
        
        h_0, c_0 = self.init_hidden(featEmbed.shape[0])
        h_0, c_0 = h_0.to(self.device), c_0.to(self.device)
        
        lengths=lengths.type(torch.LongTensor)
        lengths=lengths.to(self.device)
#         print("lengths",lengths.shape)
        
        code_pack = torch.nn.utils.rnn.pack_padded_sequence(out1, lengths, batch_first=True,enforce_sorted=False)
        
        #Run through LSTM
        code_output, (code_h_n, code_c_n)=self.rnn(code_pack, (h_0, c_0))
        code_h_n=code_h_n.squeeze()
        code_c_n=code_c_n.squeeze()
        #unpack sequence
        code_output, _ = torch.nn.utils.rnn.pad_packed_sequence(code_output, batch_first=True)
#         print("code_output",code_output.shape)
#         print("code_h_n",code_h_n.shape)
#         print("code_c_n",code_h_n.shape)
        code_output=code_output.view(code_output.shape[0],code_output.shape[2],code_output.shape[1])
#         print("code_output",code_output.shape)
        
        code_pool=[]
        code_indices=[]
        
        for i in range(code_output.shape[0]):
            pool, indices = self.code_max(code_output[i:i+1,:,0:lengths[i]])
            #print("pool",pool.shape)
            #print("indices",indices.shape)
            if i==0:
                code_pool=pool
                code_indices=indices
            else:
                code_pool=torch.cat((code_pool,pool),0)
                code_indices=torch.cat((code_indices,indices),0)
        #print("======================")
        #print("code_pool",code_pool.shape)
        #print("code_indices",code_indices.shape)
        #print("======================")
        
 
        code_pool = torch.squeeze(code_pool)
        code_indices = torch.squeeze(code_indices)
#         print("code_indices",code_indices.shape)
#         print("code_indices_ob",code_indices_ob.shape)
        #print("code_pool",code_pool.shape)
        #print(code_pool_ob[2,:])
        #print(code_pool[2,:])
        
        
        
        return code_pool,code_indices,code_h_n,code_c_n
    
    def init_hidden(self,batch_size):
        # initialize the hidden state and the cell state to zeros
        h=torch.zeros(args.rnnLayers,batch_size, self.rnn_size)
        c=torch.zeros(args.rnnLayers,batch_size, self.rnn_size)

#         if self.hparams.on_gpu:
#             hidden_a = hidden_a.cuda()
#             hidden_b = hidden_b.cuda()

        h = Variable(h)
        c = Variable(c)

        return (h, c)    

    
class Decoder(nn.Module):
    def __init__(self,device,feat_vocab_size,age_vocab_size,embed_size,rnn_size,emb_feat,batch_size):
        super(Decoder, self).__init__()
        self.embed_size=embed_size
        #self.latent_size=args.latent_size
        self.rnn_size=rnn_size
        self.feat_vocab_size=feat_vocab_size
 
        self.age_vocab_size=age_vocab_size
        
        
        self.padding_idx = 0
        self.device=device
        self.emb_feat=emb_feat
        self.build()
        
    def build(self):
        
        self.fc_pool=nn.Linear(self.rnn_size, 11, False)
        self.rnn_cell = nn.GRUCell(52*2, self.rnn_size)
        self.regression = nn.Linear(self.rnn_size, 11)

        
    def forward(self,find_contri,contri,featEmbed,labelsEmbed,pool,indices,h_n,c_n,mask,labels):   
#         print("pool",pool.shape)
#         print("indices",indices.shape)
        
        attn=self.fc_pool(pool)
#         print("attn",attn.shape)

#         print("fc_w",fc_w.shape)
        if find_contri:
            fc_w=self.fc_pool.weight.data
            fc_w=torch.squeeze(fc_w)
            all_contri=self.attention(contri,pool,indices,attn,fc_w)
                    
        dec_output=[]
        kl_input=[]
        disc_input=[]
        dec_prob=[]
        for t in range(featEmbed.shape[1]):
                #print("===============",t,"======================")
                #predict bmi
                bmi_h = self.regression(h_n)
#                 print("bmi_h",bmi_h.shape)
                
                #combine attn
                bmi_h=torch.add(bmi_h,attn)
#                 print("bmi_h",bmi_h.shape)

                bmi_prob=F.softmax(bmi_h)
#                 print("bmi_prob",bmi_prob[0])
#                 bmi_prob=torch.max(bmi_label,dim=1).values
#                 print(bmi_prob)
                bmi_label=torch.argmax(bmi_prob,dim=1)
                dec_output.append(bmi_label)
                #print("bmi_label",bmi_label.shape)
                
                d = {0:1, 1:2, 2:6, 3:13,4:25,5:33,6:174,7:39,8:42,9:44,10:173 }
                bmi_label_dict=torch.tensor([d[x.item()] for x in bmi_label])
#                 print(bmi_label[0])
#                 print("bmi_label",bmi_label.shape)
                bmi_label_enc=self.emb_feat(bmi_label_dict)
#                 print("bmi_label",bmi_label_enc.shape)
                
                x = featEmbed[:,t,:]
                bmi = labelsEmbed[:, t, :]
                #y = labels[:, t]
                m = mask[:, t]
#                 print("mask",mask.shape)
#                 print(m)
                m=m.type(torch.LongTensor)
                m=m.to(self.device)
                bmi_label=bmi_label*m
#                 bmi_prob_mask=mask[:, t]
#                 bmi_prob_mask=bmi_prob_mask.unsqueeze(1)
#                 bmi_prob_mask=bmi_prob_mask.repeat(1,11)
#                 bmi_prob_mask=bmi_prob_mask.type(torch.FloatTensor)
# #                 print(bmi_prob_mask)
#                 bmi_prob_mask=bmi_prob_mask.type(torch.FloatTensor)
#                 bmi_prob_mask=bmi_prob_mask.to(self.device)
#                 bmi_prob=bmi_prob*bmi_prob_mask
                
                #print(bmi_prob)
#                 print(m.shape)
                m=m.unsqueeze(1)
                m=m.repeat(1,self.embed_size)
                m=m.type(torch.FloatTensor)
                m=m.to(self.device)
        
#                 print("x",x.shape)
#                 print("bmi",bmi.shape)
#                 print("mask",mask.shape)
#                 print(mask)
#                 print(1-mask)
#                 print(bmi)
#                 print(bmi_label)
#                 print(bmi_label*(1-mask))


                bmi =  bmi*m + (1-m)*bmi_label_enc
#                 print(bmi)
#                 print("bmi",bmi.shape)

                
                #x_loss += torch.sum(torch.abs(bmi - bmi_h) * m) / (torch.sum(m) + 1e-5)
                
                kl_input.extend(bmi_label_dict)
                disc_input.extend(bmi_h)
                dec_prob.append(bmi_prob)
#                 print("dec_prob",dec_prob)
#                 print("dec_output",dec_output)
#                 print("dec_output",len(dec_output))
                inputs=torch.cat([x, bmi], dim = 1)

#                 print("Next input",inputs.size())
#                 print("Next input",h_n.size())
                
        
                h_n = self.rnn_cell(inputs, (h_n))
        if find_contri:
            return dec_output,dec_prob,disc_input,kl_input,all_contri
        else:
            return dec_output,dec_prob,disc_input,kl_input
    
    def attention(self,contri,pool,indices,attn,fc_w):
        #pool = pool.data.cpu().numpy()
        all_contri=[]
        assert pool.shape == indices.shape
        #fc_w = fc_w.cpu().numpy()
#         print(contri.shape)
        dummy=torch.zeros(contri.shape[0],contri.shape[1],1)
        contri=contri.type(torch.FloatTensor)
        contri=torch.cat((contri,dummy),2)
        contri=contri.type(torch.FloatTensor)
#         print(contri.shape)
                #m=m.to(self.device)
        for lab in range(fc_w.shape[0]):#for each label
            for i in range(10):#for each sample
                con = pool[i,:] * fc_w[lab,:]
                con[con < 0] = 0
                con=con+0.0001
                #print("con",con)
                con=(con/torch.sum(con))*100
                #print("con",con)
                #print("con",con.shape)
                for j in range(pool.shape[1]):
                    idx = indices[i, j]
                    contri[i,idx,2] += con[j]
#             print(contri.shape)
#             print(contri[0])
            all_contri.append(contri)
        return all_contri

    
class EncDec2(nn.Module):
    def __init__(self,device,feat_vocab_size,age_vocab_size,embed_size,rnn_size,batch_size):
        super(EncDec2, self).__init__()
        self.embed_size=embed_size
        self.latent_size=args.latent_size
        self.rnn_size=rnn_size
        self.feat_vocab_size=feat_vocab_size
 
        self.age_vocab_size=age_vocab_size
        
        self.batch_size=batch_size
        self.padding_idx = 0
        self.device=device
        self.build()
        
    def build(self):
        
        self.emb_feat=FeatEmbed(self.device,self.feat_vocab_size,self.embed_size,self.batch_size) 
        
        self.emb_age=AgeEmbed(self.device,self.age_vocab_size,self.embed_size,self.batch_size) 
#         self.disc=Discriminator(self.device)
        self.enc=Encoder2(self.device,self.feat_vocab_size,self.age_vocab_size,self.embed_size,self.rnn_size,self.batch_size)
        self.dec=Decoder2(self.device,self.feat_vocab_size,self.age_vocab_size,self.embed_size,self.rnn_size,self.emb_feat,self.batch_size)
        

        
        
    def forward(self,find_contri,enc_feat,enc_len,enc_age,dec_feat,labels,mask):   
        
        contri=torch.cat((enc_feat.unsqueeze(2),enc_age.unsqueeze(2)),2)
#         print("contri",contri.shape)
#         print("Enc featEmbed",contri.shape)
#         print("Enc featEmbed",contri[0])
        enc_feat=self.emb_feat(enc_feat)
        enc_age=self.emb_age(enc_age)
#         print("Enc featEmbed",enc_feat.shape)
#         print("Enc ageEmbed",enc_age.shape)
   
        code_output,code_h_n,code_c_n=self.enc(enc_feat,enc_len,enc_age)
        #print("code_output",code_output.shape)
        #print("code_output_n",code_output_n.shape)
        #print("========================")
        
        #===========DECODER======================
        dec_feat=self.emb_feat(dec_feat)
        dec_feat=torch.sum(dec_feat, 2)
#         print("Dec featEmbed",dec_feat.shape)
        dec_labels=self.emb_feat(labels)
#         print("Dec LabelEmbed",dec_labels.shape)
        
        if find_contri: 
            dec_output,dec_prob,disc_input,kl_input,all_contri=self.dec(find_contri,contri,dec_feat,dec_labels,code_output,code_h_n,code_c_n,mask,labels)
        else:
            dec_output,dec_prob,disc_input,kl_input=self.dec(find_contri,contri,dec_feat,dec_labels,code_output,code_h_n,code_c_n,mask,labels)
         
            
        kl_input=torch.tensor(kl_input)
#         print(len(disc_input))
#         print(disc_input[0:5])
#         print("------------------")
#         print(disc_input[195:202])
        disc_input=torch.stack(disc_input)
#         print(disc_input.shape)
#         print(disc_input[0:5])
#         print(disc_input[198:205])
        disc_input=torch.reshape(disc_input,(8,-1,disc_input.shape[1]))
#         print(disc_input.shape)
#         print(disc_input[:,0,:])
        disc_input=disc_input.permute(1,0,2)
#         self.disc(disc_input,mask,labels)
    
#         print(disc_input.shape)
#         print(disc_input[0,:,:])
#         print(disc_input[0:10])
#         print(disc_input[195:205])
        kl_input=torch.reshape(kl_input,(8,-1))
#         print(disc_input)
        kl_input=kl_input.permute(1,0)
#         print(disc_input.shape)
#         print(disc_input[0])
#         print(disc_input[1])
        #print(len(dec_prob))
        #print(dec_prob)
        #dec_prob=torch.stack(dec_prob)
        #print(dec_prob)
        
        #print("===================================================")
        
#         print(dec_prob[0].shape)
        #dec_output=torch.tensor(dec_output)
#         print("dec_output",dec_output.shape)
#         dec_output=dec_output.permute(1,0)
        #print("dec_output",dec_output)
#         print("dec_output",dec_output[0])
        #dec_prob=torch.tensor(dec_prob)
#         print("dec_prob",dec_prob.shape)
#         print(dec_prob[:,0,:])

#         dec_prob=dec_prob.permute(1,0,2)
#         print("dec_prob",dec_prob.shape)
#         print(dec_prob[0])

#         print("dec_output",dec_output.shape)
        if find_contri: 
            return dec_output,dec_prob,kl_input,all_contri
        else:
            return dec_output,dec_prob,kl_input
    
    
class Encoder2(nn.Module):
    def __init__(self,device,feat_vocab_size,age_vocab_size,embed_size,rnn_size,batch_size):
        super(Encoder2, self).__init__()
        self.embed_size=embed_size
#         self.latent_size=args.latent_size
        self.rnn_size=rnn_size
        self.feat_vocab_size=feat_vocab_size
 
        self.age_vocab_size=age_vocab_size
        
        
        self.padding_idx = 0
        self.device=device
        self.build()
        
    def build(self):

        
        self.rnn=nn.LSTM(input_size=self.embed_size*2,hidden_size=self.rnn_size,num_layers = args.rnnLayers,batch_first=True)
        self.code_max = nn.AdaptiveMaxPool1d(1, True)
        
 
        
    def forward(self,featEmbed,lengths,ageEmbed):   

        out1=torch.cat((featEmbed,ageEmbed),2)
        
#         print("out",out1.shape)
        
        out1=out1.type(torch.FloatTensor)
        out1=out1.to(self.device)
        
        
        h_0, c_0 = self.init_hidden(featEmbed.shape[0])
        h_0, c_0 = h_0.to(self.device), c_0.to(self.device)
        
        lengths=lengths.type(torch.LongTensor)
        lengths=lengths.to(self.device)
#         print("lengths",lengths.shape)
        
        code_pack = torch.nn.utils.rnn.pack_padded_sequence(out1, lengths, batch_first=True,enforce_sorted=False)
        
        #Run through LSTM
        code_output, (code_h_n, code_c_n)=self.rnn(code_pack, (h_0, c_0))
        code_h_n=code_h_n.squeeze()
        code_c_n=code_c_n.squeeze()
        #unpack sequence
        code_output,_  = torch.nn.utils.rnn.pad_packed_sequence(code_output, batch_first=True)
#         print("code_output",code_output.shape)
#         print("code_h_n",code_h_n.shape)
#         print("code_c_n",code_h_n.shape)
       
        
        
        
        
        
        return code_output,code_h_n,code_c_n

    
    def init_hidden(self,batch_size):
        # initialize the hidden state and the cell state to zeros
        h=torch.zeros(args.rnnLayers,batch_size, self.rnn_size)
        c=torch.zeros(args.rnnLayers,batch_size, self.rnn_size)

#         if self.hparams.on_gpu:
#             hidden_a = hidden_a.cuda()
#             hidden_b = hidden_b.cuda()

        h = Variable(h)
        c = Variable(c)

        return (h, c)    
    
    
class Decoder2(nn.Module):
    def __init__(self,device,feat_vocab_size,age_vocab_size,embed_size,rnn_size,emb_feat,batch_size):
        super(Decoder2, self).__init__()
        self.embed_size=embed_size
        
        self.rnn_size=rnn_size
        self.feat_vocab_size=feat_vocab_size
 
        self.age_vocab_size=age_vocab_size
        
        
        self.padding_idx = 0
        self.device=device
        self.emb_feat=emb_feat
        self.build()
        
    def build(self):
        
        self.fc_pool=nn.Linear(self.rnn_size, 11, False)
        self.rnn_cell = nn.GRUCell(self.embed_size*2+self.rnn_size, self.rnn_size)
        self.regression = nn.Linear(self.rnn_size, 11)
        self.attn=LSTMAttention(self.rnn_size)

        
    def forward(self,find_contri,contri,featEmbed,labelsEmbed,encoder_outputs,h_n,c_n,mask,labels):   

        dec_output=[]
        kl_input=[]
        disc_input=[]
        dec_prob=[]
        for t in range(featEmbed.shape[1]):
                #print("===============",t,"======================")
                #predict bmi
                bmi_h = self.regression(h_n)
#                 print("bmi_h",bmi_h.shape)
                a = self.attn(h_n, encoder_outputs)
                if (find_contri) and (t==2):
                    all_contri=self.attention(contri,a)
                a = a.unsqueeze(1)
#                 print("a",a.shape)
#                 print("attn",a[0,0,0:5])
#                 print("encoder_outputs",encoder_outputs[0,0:5,:])

        #         print("fc_w",fc_w.shape)
                weighted = torch.bmm(a, encoder_outputs)
#                 print("weighted",weighted[0,0,:])
#                 print("weighted",weighted.shape)
                weighted = weighted.permute(1, 0, 2)
                weighted = weighted.squeeze()
#                 print("weighted",weighted.shape)

#                 print("bmi_h",bmi_h.shape)

                bmi_prob=F.softmax(bmi_h)
#                 print("bmi_prob",bmi_prob[0])
#                 bmi_prob=torch.max(bmi_label,dim=1).values
#                 print(bmi_prob)
                bmi_label=torch.argmax(bmi_prob,dim=1)
                dec_output.append(bmi_label)
                #print("bmi_label",bmi_label.shape)
                
                d = {0:1, 1:2, 2:6, 3:13,4:25,5:33,6:174,7:39,8:42,9:44,10:173 }
                bmi_label_dict=torch.tensor([d[x.item()] for x in bmi_label])
#                 print(bmi_label[0])
#                 print("bmi_label",bmi_label.shape)
                bmi_label_enc=self.emb_feat(bmi_label_dict)
#                 print("bmi_label",bmi_label_enc.shape)
                
                x = featEmbed[:,t,:]
                bmi = labelsEmbed[:, t, :]
                #y = labels[:, t]
                m = mask[:, t]
#                 print("mask",mask.shape)
#                 print(m)
                m=m.type(torch.LongTensor)
                m=m.to(self.device)
                bmi_label=bmi_label*m
#                 bmi_prob_mask=mask[:, t]
#                 bmi_prob_mask=bmi_prob_mask.unsqueeze(1)
#                 bmi_prob_mask=bmi_prob_mask.repeat(1,11)
#                 bmi_prob_mask=bmi_prob_mask.type(torch.FloatTensor)
# #                 print(bmi_prob_mask)
#                 bmi_prob_mask=bmi_prob_mask.type(torch.FloatTensor)
#                 bmi_prob_mask=bmi_prob_mask.to(self.device)
#                 bmi_prob=bmi_prob*bmi_prob_mask
                
                #print(bmi_prob)
#                 print(m.shape)
                m=m.unsqueeze(1)
                m=m.repeat(1,self.embed_size)
                m=m.type(torch.FloatTensor)
                m=m.to(self.device)
        
#                 print("x",x.shape)
#                 print("bmi",bmi.shape)
#                 print("mask",mask.shape)
#                 print(mask)
#                 print(1-mask)
#                 print(bmi)
#                 print(bmi_label)
#                 print(bmi_label*(1-mask))


                bmi =  bmi*m + (1-m)*bmi_label_enc
#                 print(bmi)
#                 print("bmi",bmi.shape)

                
                #x_loss += torch.sum(torch.abs(bmi - bmi_h) * m) / (torch.sum(m) + 1e-5)
                
                kl_input.extend(bmi_label_dict)
                disc_input.extend(bmi_h)
                dec_prob.append(bmi_prob)
#                 print("dec_prob",dec_prob)
#                 print("dec_output",dec_output)
#                 print("dec_output",len(dec_output))
                
                #combine attn
#                 print("x",x.shape)
                rnn_input = torch.cat((x, weighted), dim =1)
#                 print("rnn_input",rnn_input.shape)
                inputs=torch.cat([rnn_input, bmi], dim = 1)

#                 print("Next input",inputs.size())
#                 print("Next input",h_n.size())
                
        
                h_n = self.rnn_cell(inputs, (h_n))
        if find_contri:
            return dec_output,dec_prob,disc_input,kl_input,all_contri
        else:
            return dec_output,dec_prob,disc_input,kl_input
    
    def attention(self,contri,attn):
        #pool = pool.data.cpu().numpy()
        all_contri=[]
        contri=contri[:attn.shape[0],:attn.shape[1],:]
        attn=attn.unsqueeze(2)
        attn=attn.to('cpu')
        contri=contri.type(torch.FloatTensor)
        contri=torch.cat((contri,attn),2)
        contri=contri.type(torch.FloatTensor)
#         print("contri in dec",contri.shape)
#         all_contri.append(contri)
        return contri
    
class LSTMAttention(nn.Module):
    def __init__(self, rnn_size):
        super().__init__()
        
        self.attn = nn.Linear((rnn_size * 2) , rnn_size)
        #self.attn = nn.Linear((enc_hid_dim * 2), dec_hid_dim)
        self.v = nn.Linear(rnn_size, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        #print("=====================inside attention======================")
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        #repeat decoder hidden state src_len times
        #print("hidden",hidden.shape)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        #print("hidden",hidden.shape)
#         print("encoder_outputs",encoder_outputs.shape)
        #hidden = [batch size, src len, dec hid dim]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        #energy = torch.tanh(self.attn(encoder_outputs)) 
        #print("energy",energy.shape)
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        #print("attention",attention.shape)
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)

    
class EncDec3(nn.Module):
    def __init__(self,device,feat_vocab_size,age_vocab_size,embed_size,rnn_size,batch_size):
        super(EncDec, self).__init__()
        self.embed_size=embed_size
        self.latent_size=args.latent_size
        self.rnn_size=rnn_size
        self.feat_vocab_size=feat_vocab_size
 
        self.age_vocab_size=age_vocab_size
        
        self.batch_size=batch_size
        self.padding_idx = 0
        self.device=device
        self.build()
        
    def build(self):
        
        self.emb_feat=FeatEmbed(self.device,self.feat_vocab_size,self.embed_size,self.batch_size) 
        
        self.emb_age=AgeEmbed(self.device,self.age_vocab_size,self.embed_size,self.batch_size) 
#         self.disc=Discriminator(self.device)
        self.enc=Encoder3(self.device,self.feat_vocab_size,self.age_vocab_size,self.embed_size,self.rnn_size,self.batch_size)
        self.dec=Decoder3(self.device,self.feat_vocab_size,self.age_vocab_size,self.embed_size,self.rnn_size,self.emb_feat,self.batch_size)
        

        
        
    def forward(self,find_contri,enc_feat,enc_len,enc_age,dec_feat,labels,mask):   
        
        contri=torch.cat((enc_feat.unsqueeze(2),enc_age.unsqueeze(2)),2)
#         print("Enc featEmbed",contri.shape)
#         print("Enc featEmbed",contri[0])
        enc_feat=self.emb_feat(enc_feat)
        enc_age=self.emb_age(enc_age)
#         print("Enc featEmbed",enc_feat.shape)
#         print("Enc ageEmbed",enc_age.shape)
   
        code_output,code_h_n,code_c_n=self.enc(enc_feat,enc_len,enc_age)
#         print("pool",code_pool.shape)
#         print("code_indices",code_indices.shape)
        #print("========================")
        
        #===========DECODER======================
        dec_feat=self.emb_feat(dec_feat)
        dec_feat=torch.sum(dec_feat, 2)
#         print("Dec featEmbed",dec_feat.shape)
        dec_labels=self.emb_feat(labels)
#         print("Dec LabelEmbed",dec_labels.shape)
        
        if find_contri: 
            dec_output,dec_prob,disc_input,kl_input,all_contri=self.dec(find_contri,contri,dec_feat,dec_labels,enc_len,code_output,code_h_n,code_c_n,mask,labels)
        else:
            dec_output,dec_prob,disc_input,kl_input=self.dec(find_contri,contri,dec_feat,dec_labels,enc_len,code_output,code_h_n,code_c_n,mask,labels)
         
            
        kl_input=torch.tensor(kl_input)
#         print(len(disc_input))
#         print(disc_input[0:5])
#         print("------------------")
#         print(disc_input[195:202])
        disc_input=torch.stack(disc_input)
#         print(disc_input.shape)
#         print(disc_input[0:5])
#         print(disc_input[198:205])
        disc_input=torch.reshape(disc_input,(8,-1,disc_input.shape[1]))
#         print(disc_input.shape)
#         print(disc_input[:,0,:])
        disc_input=disc_input.permute(1,0,2)
#         self.disc(disc_input,mask,labels)
    
#         print(disc_input.shape)
#         print(disc_input[0,:,:])
#         print(disc_input[0:10])
#         print(disc_input[195:205])
        kl_input=torch.reshape(kl_input,(8,-1))
#         print(disc_input)
        kl_input=kl_input.permute(1,0)
#         print(disc_input.shape)
#         print(disc_input[0])
#         print(disc_input[1])
        #print(len(dec_prob))
        #print(dec_prob)
        #dec_prob=torch.stack(dec_prob)
        #print(dec_prob)
        
        #print("===================================================")
        
#         print(dec_prob[0].shape)
        #dec_output=torch.tensor(dec_output)
#         print("dec_output",dec_output.shape)
#         dec_output=dec_output.permute(1,0)
        #print("dec_output",dec_output)
#         print("dec_output",dec_output[0])
        #dec_prob=torch.tensor(dec_prob)
#         print("dec_prob",dec_prob.shape)
#         print(dec_prob[:,0,:])

#         dec_prob=dec_prob.permute(1,0,2)
#         print("dec_prob",dec_prob.shape)
#         print(dec_prob[0])

#         print("dec_output",dec_output.shape)
        if find_contri: 
            return dec_output,dec_prob,kl_input,all_contri
        else:
            return dec_output,dec_prob,kl_input
    
    
 
    
class Encoder3(nn.Module):
    def __init__(self,device,feat_vocab_size,age_vocab_size,embed_size,rnn_size,batch_size):
        super(Encoder3, self).__init__()
        self.embed_size=embed_size
#         self.latent_size=args.latent_size
        self.rnn_size=rnn_size
        self.feat_vocab_size=feat_vocab_size
 
        self.age_vocab_size=age_vocab_size
        
        
        self.padding_idx = 0
        self.device=device
        self.build()
        
    def build(self):

        
        self.rnn=nn.LSTM(input_size=self.embed_size*2,hidden_size=self.rnn_size,num_layers = args.rnnLayers,batch_first=True)
        
    def forward(self,featEmbed,lengths,ageEmbed):   

        out1=torch.cat((featEmbed,ageEmbed),2)
        
#         print("out",out1.shape)
        
        out1=out1.type(torch.FloatTensor)
        out1=out1.to(self.device)
        
        
        h_0, c_0 = self.init_hidden(featEmbed.shape[0])
        h_0, c_0 = h_0.to(self.device), c_0.to(self.device)
        
        lengths=lengths.type(torch.LongTensor)
        lengths=lengths.to(self.device)
#         print("lengths",lengths.shape)
        
        code_pack = torch.nn.utils.rnn.pack_padded_sequence(out1, lengths, batch_first=True,enforce_sorted=False)
        
        #Run through LSTM
        code_output, (code_h_n, code_c_n)=self.rnn(code_pack, (h_0, c_0))
        code_h_n=code_h_n.squeeze()
        code_c_n=code_c_n.squeeze()
        #unpack sequence
        code_output, _ = torch.nn.utils.rnn.pad_packed_sequence(code_output, batch_first=True)
#         print("code_output",code_output.shape)
#         print("code_h_n",code_h_n.shape)
#         print("code_c_n",code_h_n.shape)
  
        
        
        
        return code_pool,code_indices,code_h_n,code_c_n
    
    def init_hidden(self,batch_size):
        # initialize the hidden state and the cell state to zeros
        h=torch.zeros(args.rnnLayers,batch_size, self.rnn_size)
        c=torch.zeros(args.rnnLayers,batch_size, self.rnn_size)

#         if self.hparams.on_gpu:
#             hidden_a = hidden_a.cuda()
#             hidden_b = hidden_b.cuda()

        h = Variable(h)
        c = Variable(c)

        return (h, c)    

    
class Decoder3(nn.Module):
    def __init__(self,device,feat_vocab_size,age_vocab_size,embed_size,rnn_size,emb_feat,batch_size):
        super(Decoder3, self).__init__()
        self.embed_size=embed_size
        #self.latent_size=args.latent_size
        self.rnn_size=rnn_size
        self.feat_vocab_size=feat_vocab_size
 
        self.age_vocab_size=age_vocab_size
        
        
        self.padding_idx = 0
        self.device=device
        self.emb_feat=emb_feat
        self.build()
        
    def build(self):
        
        
        self.rnn_cell = nn.GRUCell(52*2, self.rnn_size)
        self.regression = nn.Linear(self.rnn_size, 11)
        self.attn=MaxAttention()
   
    def forward(self,find_contri,contri,featEmbed,labelsEmbed,enc_len,encoder_output,h_n,c_n,mask,labels):   
#         print("pool",pool.shape)
#         print("indices",indices.shape)
        
            
        dec_output=[]
        kl_input=[]
        disc_input=[]
        dec_prob=[]
        for t in range(featEmbed.shape[1]):
                #print("===============",t,"======================")
                #predict bmi
                bmi_h = self.regression(h_n)
#                 print("bmi_h",bmi_h.shape)
                if (find_contri) and (t==2):
                    a, all_contri = self.attn(find_contri,contri,encoder_outputs,enc_len)
                else:
                    a = self.attn(find_contri,contri,encoder_outputs,enc_len)
                
                
                #combine attn
                bmi_h=torch.add(bmi_h,a)
#                 print("bmi_h",bmi_h.shape)

                bmi_prob=F.softmax(bmi_h)
#                 print("bmi_prob",bmi_prob[0])
#                 bmi_prob=torch.max(bmi_label,dim=1).values
#                 print(bmi_prob)
                bmi_label=torch.argmax(bmi_prob,dim=1)
                dec_output.append(bmi_label)
                #print("bmi_label",bmi_label.shape)
                
                d = {0:1, 1:2, 2:6, 3:13,4:25,5:33,6:174,7:39,8:42,9:44,10:173 }
                bmi_label_dict=torch.tensor([d[x.item()] for x in bmi_label])
#                 print(bmi_label[0])
#                 print("bmi_label",bmi_label.shape)
                bmi_label_enc=self.emb_feat(bmi_label_dict)
#                 print("bmi_label",bmi_label_enc.shape)
                
                x = featEmbed[:,t,:]
                bmi = labelsEmbed[:, t, :]
                #y = labels[:, t]
                m = mask[:, t]
#                 print("mask",mask.shape)
#                 print(m)
                m=m.type(torch.LongTensor)
                m=m.to(self.device)
                bmi_label=bmi_label*m
#                 bmi_prob_mask=mask[:, t]
#                 bmi_prob_mask=bmi_prob_mask.unsqueeze(1)
#                 bmi_prob_mask=bmi_prob_mask.repeat(1,11)
#                 bmi_prob_mask=bmi_prob_mask.type(torch.FloatTensor)
# #                 print(bmi_prob_mask)
#                 bmi_prob_mask=bmi_prob_mask.type(torch.FloatTensor)
#                 bmi_prob_mask=bmi_prob_mask.to(self.device)
#                 bmi_prob=bmi_prob*bmi_prob_mask
                
                #print(bmi_prob)
#                 print(m.shape)
                m=m.unsqueeze(1)
                m=m.repeat(1,self.embed_size)
                m=m.type(torch.FloatTensor)
                m=m.to(self.device)
        
#                 print("x",x.shape)
#                 print("bmi",bmi.shape)
#                 print("mask",mask.shape)
#                 print(mask)
#                 print(1-mask)
#                 print(bmi)
#                 print(bmi_label)
#                 print(bmi_label*(1-mask))


                bmi =  bmi*m + (1-m)*bmi_label_enc
#                 print(bmi)
#                 print("bmi",bmi.shape)

                
                #x_loss += torch.sum(torch.abs(bmi - bmi_h) * m) / (torch.sum(m) + 1e-5)
                
                kl_input.extend(bmi_label_dict)
                disc_input.extend(bmi_h)
                dec_prob.append(bmi_prob)
#                 print("dec_prob",dec_prob)
#                 print("dec_output",dec_output)
#                 print("dec_output",len(dec_output))
                inputs=torch.cat([x, bmi], dim = 1)

#                 print("Next input",inputs.size())
#                 print("Next input",h_n.size())
                
        
                h_n = self.rnn_cell(inputs, (h_n))
        if find_contri:
            return dec_output,dec_prob,disc_input,kl_input,all_contri
        else:
            return dec_output,dec_prob,disc_input,kl_input
    


class MaxAttention(nn.Module):
    def __init__(self, rnn_size):
        super().__init__()
        
        self.code_max = nn.AdaptiveMaxPool1d(1, True)
        self.fc_pool=nn.Linear(self.rnn_size, 11, False)
        
    def forward(self, find_contri,contri,code_output,lengths):
        
        code_output=code_output.view(code_output.shape[0],code_output.shape[2],code_output.shape[1])
#         print("code_output",code_output.shape)
        
        code_pool=[]
        code_indices=[]

        for i in range(code_output.shape[0]):
            pool, indices = self.code_max(code_output[i:i+1,:,0:lengths[i]])
            #print("pool",pool.shape)
            #print("indices",indices.shape)
            if i==0:
                code_pool=pool
                code_indices=indices
            else:
                code_pool=torch.cat((code_pool,pool),0)
                code_indices=torch.cat((code_indices,indices),0)
        #print("======================")
        #print("code_pool",code_pool.shape)
        #print("code_indices",code_indices.shape)
        #print("======================")


        code_pool = torch.squeeze(code_pool)
        code_indices = torch.squeeze(code_indices)
        #         print("code_indices",code_indices.shape)
        #         print("code_indices_ob",code_indices_ob.shape)
        #print("code_pool",code_pool.shape)
        #print(code_pool_ob[2,:])
        #print(code_pool[2,:])
        attn=self.fc_pool(code_pool)
#         print("attn",attn.shape)

#         print("fc_w",fc_w.shape)
        if find_contri:
            fc_w=self.fc_pool.weight.data
            fc_w=torch.squeeze(fc_w)
            all_contri=self.attention(contri,code_pool,code_indices,attn,fc_w)
            
            return attn,all_contri
        return attn
    
    def attention(self,contri,pool,indices,attn,fc_w):
        #pool = pool.data.cpu().numpy()
        all_contri=[]
        assert pool.shape == indices.shape
        #fc_w = fc_w.cpu().numpy()
#         print(contri.shape)
        dummy=torch.zeros(contri.shape[0],contri.shape[1],1)
        contri=contri.type(torch.FloatTensor)
        contri=torch.cat((contri,dummy),2)
        contri=contri.type(torch.FloatTensor)
#         print(contri.shape)
                #m=m.to(self.device)
        for lab in range(fc_w.shape[0]):#for each label
            for i in range(10):#for each sample
                con = pool[i,:] * fc_w[lab,:]
                con[con < 0] = 0
                con=con+0.0001
                #print("con",con)
                con=(con/torch.sum(con))*100
                #print("con",con)
                #print("con",con.shape)
                for j in range(pool.shape[1]):
                    idx = indices[i, j]
                    contri[i,idx,2] += con[j]
#             print(contri.shape)
#             print(contri[0])
            all_contri.append(contri)
        return all_contri
    
class Discriminator(nn.Module):
    def __init__(self,device):
        super(Discriminator, self).__init__()       
        self.device=device
        self.build()
        
    def build(self):
        self.fc = nn.Linear(1, 11)
        self.js_loss = JSD()
        
    def forward(self,pred,mask,labels):   
        print(pred.shape)
        print(mask.shape)
        print(labels.shape)
        labels=labels.unsqueeze(2)
        labels=labels.type(torch.FloatTensor)
        labels=labels.to(self.device)
        print(labels.shape)
        labels=self.fc(labels)
        print(labels.shape)
        labels=F.softmax(labels)
        print(labels.shape)
        pred=F.softmax(pred)
        print(pred.shape)
        js=self.js_loss(pred, labels)
        print(js)
        
        


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, p: torch.tensor, q: torch.tensor):
        #p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        print(p)
        print(q)
        m = (0.5 * (p + q))
        print(m)
        return 0.5 * (self.kl(p.log(), m) + self.kl(q.log(), m))
    
    
class FeatEmbed(nn.Module):
    def __init__(self,device,feat_vocab_size,embed_size,batch_size):
        super(FeatEmbed, self).__init__()
        self.embed_size=embed_size

        self.feat_vocab_size=feat_vocab_size

        
        self.padding_idx = 0
        self.device=device
        self.build()
        
    def build(self):
        
        self.emb_feat=nn.Embedding(self.feat_vocab_size,self.embed_size,self.padding_idx) 

    def forward(self,feat):  
      
        feat=feat.type(torch.LongTensor)
        feat=feat.to(self.device)
        #print("enc feat",feat[0])
        #print(self.emb_feat(torch.LongTensor([0,167])))
        featEmbed=self.emb_feat(feat)
        #print("enc feat",featEmbed[0])
        featEmbed=featEmbed.type(torch.FloatTensor)
        featEmbed=featEmbed.to(self.device)
            
        

        return featEmbed   

class AgeEmbed(nn.Module):
    def __init__(self,device,age_vocab_size,embed_size,batch_size):
        super(AgeEmbed, self).__init__()
        self.embed_size=embed_size

        self.age_vocab_size=age_vocab_size

        
        self.padding_idx = 0
        self.device=device
        self.build()
        
    def build(self):
        
        self.emb_age=nn.Embedding(self.age_vocab_size,self.embed_size,self.padding_idx) 

    def forward(self,age):  
        
        age=age.type(torch.LongTensor)
        age=age.to(self.device)
        ageEmbed=self.emb_age(age)

        ageEmbed=ageEmbed.type(torch.FloatTensor)
        ageEmbed=ageEmbed.to(self.device)


        return ageEmbed  
        


class LSTMAttn(nn.Module):
    def __init__(self,device,cond_vocab_size,cond_seq_len,proc_vocab_size,proc_seq_len,med_vocab_size,med_seq_len,out_vocab_size,out_seq_len,chart_vocab_size,chart_seq_len,lab_vocab_size,lab_seq_len,eth_vocab_size,gender_vocab_size,age_vocab_size,med_signal,lab_signal,embed_size,rnn_size,batch_size):
        super(LSTMAttn, self).__init__()
        self.embed_size=embed_size
        self.rnn_size=rnn_size
        self.eth_vocab_size=eth_vocab_size
        self.gender_vocab_size=gender_vocab_size
        self.age_vocab_size=age_vocab_size
        self.cond_vocab_size=cond_vocab_size
        self.cond_seq_len=cond_seq_len
        self.proc_vocab_size=proc_vocab_size
        self.proc_seq_len=proc_seq_len
        self.med_vocab_size=med_vocab_size
        self.med_seq_len=med_seq_len
        self.out_vocab_size=out_vocab_size
        self.out_seq_len=out_seq_len
        self.chart_vocab_size=chart_vocab_size
        self.chart_seq_len=chart_seq_len
        self.lab_vocab_size=lab_vocab_size
        self.lab_seq_len=lab_seq_len
        if self.chart_seq_len>500:
            self.chart_seq_len=500
        self.batch_size=batch_size
        self.padding_idx = 0
        self.modalities=0
        self.device=device
        self.med_signal,self.lab_signal=med_signal,lab_signal
        self.build()
        
    def build(self):
        
        if self.med_vocab_size:
            self.med=CodeAttn(self.device,self.embed_size,self.rnn_size,self.med_vocab_size,self.med_seq_len,self.batch_size,self.med_signal,False)
            self.modalities=self.modalities+1
                
        if self.proc_vocab_size:
            self.proc=CodeAttn(self.device,self.embed_size,self.rnn_size,self.proc_vocab_size,self.proc_seq_len,self.batch_size,True,False)
            self.modalities=self.modalities+1
        if self.out_vocab_size:
            self.out=CodeAttn(self.device,self.embed_size,self.rnn_size,self.out_vocab_size,self.out_seq_len,self.batch_size,True,False)
            self.modalities=self.modalities+1
        if self.chart_vocab_size:
            self.chart=CodeAttn(self.device,self.embed_size,self.rnn_size,self.chart_vocab_size,self.chart_seq_len,self.batch_size,self.lab_signal,True)
            self.modalities=self.modalities+1
        if self.lab_vocab_size:
            self.lab=CodeAttn(self.device,self.embed_size,self.rnn_size,self.lab_vocab_size,self.lab_seq_len,self.batch_size,self.lab_signal,False)
            self.modalities=self.modalities+1

        
        self.condEmbed=nn.Embedding(self.cond_vocab_size,self.embed_size,self.padding_idx) 
        self.condfc=nn.Linear((self.embed_size*self.cond_seq_len),self.rnn_size, False)
        
        self.ethEmbed=nn.Embedding(self.eth_vocab_size,self.embed_size,self.padding_idx) 
        self.genderEmbed=nn.Embedding(self.gender_vocab_size,self.embed_size,self.padding_idx) 
        self.ageEmbed=nn.Embedding(self.age_vocab_size,self.embed_size,self.padding_idx) 
        self.demo_fc=nn.Linear(self.embed_size*3, self.rnn_size, False)
        
        #self.fc=nn.Linear((self.embed_size*self.cond_seq_len)+3*self.rnn_size, 1, False)
        self.fc1=nn.Linear(int(self.rnn_size*(self.modalities+2)), int((self.rnn_size*(self.modalities+2))/2), False)
        self.fc2=nn.Linear(int((self.rnn_size*(self.modalities+2))/2), int((self.rnn_size*(self.modalities+2))/4), False)
        self.fc3=nn.Linear(int((self.rnn_size*(self.modalities+2))/4), 1, False)
        
        #self.sig = nn.Sigmoid()
        
    def forward(self,X):        
        meds,chart,out,proc,lab,conds,demo=X[0],X[1],X[2],X[3],X[4],X[5],X[6]    
        
        out1 = torch.zeros(size=(1,0))
        
        if len(meds[0]):
            med_h_n = self.med(meds)  
            med_h_n=med_h_n.view(med_h_n.shape[0],-1)
            #print("med_h_n",med_h_n.shape)
            out1=med_h_n
            #print(out1.shape)
            #print(out1.nelement())
        if len(procs):
            proc_h_n = self.proc(procs)  
            proc_h_n=proc_h_n.view(proc_h_n.shape[0],-1)
            #print("proc_h_n",proc_h_n.shape)
            if out1.nelement():
                out1=torch.cat((out1,proc_h_n),1)
            else:
                out1=proc_h_n
        if len(labs[0]):
            lab_h_n = self.lab(labs)  
            lab_h_n=lab_h_n.view(lab_h_n.shape[0],-1)
            #print("lab_h_n",lab_h_n.shape)
            if out1.nelement():
                out1=torch.cat((out1,lab_h_n),1)
            else:
                out1=lab_h_n
        if len(outs):
            out_h_n = self.out(outs)  
            out_h_n=out_h_n.view(out_h_n.shape[0],-1)
            if out1.nelement():
                out1=torch.cat((out1,out_h_n),1)
            else:
                out1=out_h_n
        if len(charts[0]):
            chart_h_n = self.chart(charts)  
            chart_h_n=out_h_n.view(chart_h_n.shape[0],-1)
            if out1.nelement:
                out1=torch.cat((out1,chart_h_n),1)
            else:
                out1=chart_h_n
        
        conds=conds.to(self.device)
        conds=self.condEmbed(conds)
        #print(conds.shape)
        conds=conds.view(conds.shape[0],-1)
        conds=self.condfc(conds)
        #print(conds.shape)
        #print("cond_pool_ob",cond_pool_ob.shape)
        #out1=torch.cat((cond_pool,cond_pool_ob),1)
        #out1=cond_pool
        eth=demo[0].to(self.device)
        eth=self.ethEmbed(eth)
        
        gender=demo[1].to(self.device)
        gender=self.genderEmbed(gender)
        
        age=demo[2].to(self.device)
        age=self.ageEmbed(age)
        
        demog=torch.cat((eth,gender),1)
        demog=torch.cat((demog,age),1)
        #print("demog",demog.shape)
        demog=self.demo_fc(demog)
        
        out1=torch.cat((out1,conds),1)
        out1=torch.cat((out1,demog),1)
        #print("out1",out1.shape)
        out1 = self.fc1(out1)
        out1 = self.fc2(out1)
        out1 = self.fc3(out1)
        #print("out1",out1.shape)
        
        sig = nn.Sigmoid()
        sigout1=sig(out1)
        #print("sig out",sigout1[16])
        #print("sig out",sigout1)
        #print(out1[0])
        #print("hi")
        
        return sigout1,out1
        
            


# In[ ]:


class CodeAttn(nn.Module):
    def __init__(self,device,embed_size,rnn_size,code_vocab_size,code_seq_len,batch_size,signal,lab):           
        super(CodeAttn, self).__init__()
        self.embed_size=embed_size
        self.rnn_size=rnn_size
        self.code_vocab_size=code_vocab_size
        self.code_seq_len=code_seq_len
        self.batch_size=batch_size
        self.padding_idx = 0
        self.device=device
        self.signal=signal
        self.build()
        self.lab_sig=lab
    
    def build(self):
        
        self.codeEmbed=nn.Embedding(self.code_vocab_size,self.embed_size,self.padding_idx)
        if self.signal: 
            self.codeRnn = nn.LSTM(input_size=int(self.embed_size*self.code_seq_len),hidden_size=self.rnn_size,num_layers = 2,dropout=0.2,batch_first=True)
            #self.codeRnn = nn.LSTM(input_size=self.embed_size,hidden_size=self.rnn_size,num_layers = 2,dropout=0.2,batch_first=True)
        else:
            self.codeRnn = nn.LSTM(input_size=int((self.embed_size+1)*self.code_seq_len),hidden_size=self.rnn_size,num_layers = 2,dropout=0.2,batch_first=True)
            #self.codeRnn = nn.LSTM(input_size=self.embed_size+1,hidden_size=self.rnn_size,num_layers = 2,dropout=0.2,batch_first=True)

        self.code_fc=nn.Linear(self.rnn_size, 1, False)
        #self.dropout1 = nn.Dropout(0.2)
        
    def forward(self, code):
        #print(conds.shape)

        h_0, c_0 = self.init_hidden()
        h_0, c_0, code = h_0.to(self.device), c_0.to(self.device),code.to(self.device)

        #Embedd all sequences
        #print(code.shape)
        #print(code[0,:,:])

        if code.shape[0]==2:
            dat=code[1]
            code=code[0]
            if self.lab_sig:
                if code.shape[1]>500:
                    code=code[:,0:500,:]
                    dat=dat[:,0:500,:]
            codeEmbedded=self.codeEmbed(code)
            #code=torch.transpose(code,1,2)
            #code=torch.reshape(code,(code.shape[0],code.shape[1],-1))
            #code=torch.sum(code,1)
            #print(code.shape)
            #print(self.signal)
            if not self.signal:
                if self.lab_sig:
                    test=torch.max(code,2)
                    test=test.values
                    test=test.unsqueeze(2)
                    code=torch.zeros(code.shape[0],code.shape[1],code.shape[2])
                    code=code.type(torch.FloatTensor)
                    code=code.to(self.device)
                    test=test.type(torch.FloatTensor)
                    test=test.to(self.device)
                    code=torch.add(code,test)
                    code=code.type(torch.LongTensor)
                    code=code.to(self.device)
                    codeEmbedded=self.codeEmbed(code)
                dat=dat.unsqueeze(3)
                #print(dat.shape)
                dat=dat.type(torch.FloatTensor)
                dat=dat.to(self.device)
                codeEmbedded=torch.cat((codeEmbedded,dat),3)
            code=torch.transpose(codeEmbedded,1,2)
            code=torch.reshape(code,(code.shape[0],code.shape[1],-1))
            #code=torch.sum(codeEmbedded,1)
            
                #print(code.shape)
        else:
            code=self.codeEmbed(code)
            code=torch.transpose(code,1,2)
            code=torch.reshape(code,(code.shape[0],code.shape[1],-1))
            #code=torch.sum(code,1)
        #print(code.shape)
        #code=torch.transpose(code,1,2)
        #print(code[0])
        #print(dat[0])
        #print(code[0,:,:])

        h_0, c_0, code = h_0.to(self.device), c_0.to(self.device),code.to(self.device)
        #print(code.shape)
        #code=code.type(torch.FloatTensor)
#        code_time=code_time.type(torch.FloatTensor)
        #h_0, c_0, code = h_0.to(self.device), c_0.to(self.device),code.to(self.device)

#        code=torch.cat((code,code_time),dim=2)
            
        #Run through LSTM
        code_output, (code_h_n, code_c_n)=self.codeRnn(code, (h_0, c_0))
        #print("code_output",code_output.shape)
        
        code_softmax=self.code_fc(code_output)
        #code_output=self.dropout1(code_output) 
        #print("softmax",code_softmax.shape)
        code_softmax=F.softmax(code_softmax)
        #print("softmax",code_softmax.shape)
        code_softmax=torch.sum(torch.mul(code_output,code_softmax),dim=1)
        #print("softmax",code_softmax.shape)
        #print("========================")
        
        return code_softmax
    
    
    def init_hidden(self):
        # initialize the hidden state and the cell state to zeros
        h=torch.zeros(2,self.batch_size, self.rnn_size)
        c=torch.zeros(2,self.batch_size, self.rnn_size)

#         if self.hparams.on_gpu:
#             hidden_a = hidden_a.cuda()
#             hidden_b = hidden_b.cuda()

        h = Variable(h)
        c = Variable(c)

        return (h, c)    
    

            




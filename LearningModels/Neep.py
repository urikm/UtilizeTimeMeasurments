# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 23:25:01 2020

@title: Take.1 - try to resamble entropy prod rate as features of net

@author: Uri Kapustin
"""

# %% imports
import numpy as np

from scipy import stats


# PyTorch libraries and modules
import torch

from torch.nn import Linear, Conv2d, MaxPool2d, Module, LSTM, Embedding, GRU


# %% Define Models

# %% RNEEP - no time data as input
class RNEEP(Module):
    def __init__(self):
        super(RNEEP, self).__init__()
        nSymbols = 3 # state 1 ,2 and 3(hidden)
        nHiddenSize = 128
        nLayersGRU = 1
        self.encoder = Embedding(nSymbols, nHiddenSize)
        self.rnn = GRU(nHiddenSize, nHiddenSize, nLayersGRU)
        self.fc = Linear(nHiddenSize, 1)

        self.nhid = nHiddenSize
        self.nlayers = nLayersGRU

    def forward(self, x):
        # 1dim - size of seq to train ; 0dim - num of sequences
        bsz = x.size(1)
        h_f = self.init_hidden(bsz)
        emb_forward = self.encoder(x)
        output_f, _ = self.rnn(emb_forward, h_f)

        h_r = self.init_hidden(bsz)
        x_r = torch.flip(x, [0])
        emb_reverse = self.encoder(x_r)
        output_r, _ = self.rnn(emb_reverse, h_r)

        S = self.fc(output_f.mean(dim=0)) - self.fc(output_r.mean(dim=0))
        # S = self.fc(output_f[-1:,:,:]) - self.fc(output_r[-1:,:,:])
        return S

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, bsz, self.nhid).detach()

# %% RNEEP - with time data as input
class RNEEPT(Module):
    def __init__(self):
        super(RNEEPT, self).__init__()
        nSymbols = 3 # state 1 ,2 and 3(hidden)
        nHiddenSize = 128
        nHiddenSizeT = 128
        nOutFc1 = 16
        nOutFc1t = nOutFc1
        nLayersGRU = 1
        nFilters = 2
        nFc = 1
        self.encoder = Embedding(nSymbols, nHiddenSize)
        self.rnn = GRU(nHiddenSize, nHiddenSize, nLayersGRU)
        self.rnnT = GRU(1, nHiddenSizeT, nLayersGRU)
        # self.do = Dropout(p=0.2)
        # self.fc1 = Linear(nHiddenSize, nOutFc1)
        # self.fc1t = Linear(nHiddenSizeT, nOutFc1t)  
        # self.fc2 = Linear(nFilters*nOutFc1t, nFc)
        self.fc1 = Linear(nHiddenSize, nOutFc1)
        self.fc2 = Linear(nOutFc1, 1)        
        self.conv2d = Conv2d(2,1,(1,1))
        
        self.nhid = nHiddenSize
        self.nhidT = nHiddenSizeT
        self.nlayers = nLayersGRU
        
    def forward(self, x):
        # 1dim - size of seq to train ; 0dim - num of sequences
        bsz = x.size(1)
        ## Forward Trajectory 
        # states
        h_f = self.init_hidden(bsz)
        emb_forward = self.encoder(x[:,:,0].long().squeeze())
        rnn_f, _ = self.rnn(emb_forward, h_f)
        rnn_f = rnn_f.mean(dim=0).unsqueeze(1).unsqueeze(3)  # take only the output from last rnn node
        # output_f = self.fc1(rnn_f) # dimension reduction? no needed..
        
        # Waiting times
        h_f_t = self.init_hidden(bsz,states=False)
        rnn_f_t, _ = self.rnnT(x[:,:,1].unsqueeze(2), h_f_t)  
        rnn_f_t = rnn_f_t.mean(dim=0).unsqueeze(1).unsqueeze(3) # take only the output from last rnn node
        # output_f_t = self.fc1t(rnn_f_t) # dimension reduction? no needed..
        out_conv = self.conv2d(torch.cat((rnn_f,rnn_f_t),1))
        out_f = self.fc1(out_conv.squeeze())
        out_f = self.fc2(out_f)
        ## Backward Trajectory
        # states
        h_r = self.init_hidden(bsz)
        x_r = torch.flip(x, [0])# Maybe need to be changed
        emb_reverse = self.encoder(x_r[:,:,0].long().squeeze())
        rnn_r, _ = self.rnn(emb_reverse, h_r)
        rnn_r = rnn_r.mean(dim=0).unsqueeze(1).unsqueeze(3) 
        # output_r = self.fc1(rnn_r)
        # Waiting times
        h_r_t = self.init_hidden(bsz,states=False)
        rnn_r_t, _ = self.rnnT(x_r[:,:,1].unsqueeze(2), h_r_t)     
        rnn_r_t = rnn_r_t.mean(dim=0).unsqueeze(1).unsqueeze(3) 
        # output_r_t = self.fc1t(rnn_r_t)
        out_conv_r = self.conv2d(torch.cat((rnn_r,rnn_r_t),1))
        out_r = self.fc1(out_conv_r.squeeze())  
        out_r = self.fc2(out_r)
        # S = self.fc(output_f.mean(dim=0)) - self.fc(output_r.mean(dim=0))
        S = out_f - out_r
        # S = (output_f-output_r) + (output_f_t-output_r_t)
        return S

    def init_hidden(self, bsz,states=True):
        weight = next(self.parameters())
        if states:
            we = weight.new_zeros(self.nlayers, bsz, self.nhid).detach()
        else: # for later use
            we = weight.new_zeros(self.nlayers, bsz, self.nhidT).detach()   
        return we
    

# %% train RNN neep models
def make_trainRnn(model,optimizer,seqSize,device):
    def trainRnn(trainLoader,validationLoader,iEpoch):
        
        # avgTrainLosses = []
        bestValidLoss = 1e3 # Init
        bestEpRate = 0
        bestEpErr = 1e6
        k=0
        
        for x_batch, y_batch in trainLoader:
            model.train()
            
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # prediction for training
            entropy_train = model(torch.transpose(x_batch,0,1))
            # y_batch = y_batch.unsqueeze(1)
            # clearing the Gradients of the model parameters
            optimizer.zero_grad()       
            # computing the training
            loss_train = (-entropy_train + torch.exp(-entropy_train)).mean()
            # computing the updated weights of all the model parameters
            loss_train.backward()
            optimizer.step()           
            # avgTrainLosses.append(-output_train.mean().item())
            
            avgValLosses = []
            avgValScores = []
            k+=1
            if k % (384/seqSize) == 0 or k == 1:
                with torch.no_grad():
                    for x_val, y_val in validationLoader:
                        x_val = x_val.to(device)
                        y_val = y_val.to(device)                    
                        model.eval()
                        
                        entropy_val = model(torch.transpose(x_val,0,1))
                        # y_val = y_val.unsqueeze(1)
                        val_loss = (-entropy_val + torch.exp(-entropy_val)).mean()
                        avgValLosses.append(val_loss.item())
                        avgValScores.append(entropy_val.cpu().numpy())
                    avgValScores = np.cumsum(np.concatenate(avgValScores))
                    predEntRate, _, _, _, _ = stats.linregress(
                        np.arange(len(avgValScores)) * (seqSize - 1), avgValScores
                    )
                    avgValLoss = np.average(np.array(avgValLosses)) 
                    # avgValScore = np.average(np.array(avgValScores))
                    
                    if avgValLoss < bestValidLoss:
                        bestEpRate = predEntRate
                        bestEpErr = np.abs(bestEpRate-y_val[0].cpu().numpy())/y_val[0].cpu().numpy()
                        bestValidLoss = avgValLoss
        if iEpoch % 1 == 0:                
            print('Epoch : ',iEpoch+1,'\t' 'Best Loss :',bestValidLoss, '\t' 'Best EP rate err Train :', bestEpErr)
        
        return bestValidLoss,bestEpRate,bestEpErr
    return trainRnn  

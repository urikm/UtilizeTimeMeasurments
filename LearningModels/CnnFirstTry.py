# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 23:25:01 2020

@title: Take.1 - try to resamble entropy prod rate as features of net

@author: Uri Kapustin
"""

# %% imports
import numpy as np
import random as rd
import os
import pickle
import matplotlib.pyplot as plt
# %matplotlib inline
import DataSetsCreation as dbCreate

from scipy import stats

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
# from tqdm import tqdm

# PyTorch libraries and modules
import torch

from torch.nn import Linear, ReLU, BatchNorm2d, BCEWithLogitsLoss ,Sequential, Conv2d, MaxPool2d, Module, LSTM, BatchNorm2d, Embedding, GRU
from torch.optim import Adam, SGD


# %% Inits
lenOfTraj = 1000 # the length of each data sample for the network
# %% Define Model 
class SimpleNet(Module):   
    def __init__(self, lenOfTraj):
        super(SimpleNet, self).__init__()
        
        firstLayersSize = 16
        secondLayerSize = 24
        self.cnn_layers = Sequential(
            # BatchNorm2d(1),
            # Defining a 2D convolution layer
            Conv2d(1, firstLayersSize, kernel_size=(2,1)),
            ReLU(inplace=True),
            BatchNorm2d(firstLayersSize),
            # MaxPool2d(kernel_size=(2,1), stride=1),
            # Defining another 2D convolution layer
            Conv2d(firstLayersSize, secondLayerSize, kernel_size=(2,2)),
            ReLU(inplace=True),
            BatchNorm2d(secondLayerSize),
            # MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            # Linear(secondLayerSize * int(lenOfTraj-9) * 1,2),
            Linear(23952,8),
            ReLU(inplace=True),
            Linear(8,2),
            ReLU(inplace=True),
            Linear(2, 1)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
    
# %% Define train and validation process
def make_train(model,optimizer,criterion):
    def trainSNet(trainLoader,validationLoader,iEpoch):
        
        tr_loss = 0
        avgTrainLosses = []
        avgValLosses = []
        avgValScore = []
        
        for x_batch, y_batch in trainLoader:
            model.train()
            
            # prediction for training
            output_train = model(x_batch)
        
            # computing the training
            loss_train = criterion(output_train, y_batch)
            avgTrainLosses.append(loss_train.item())
            # clearing the Gradients of the model parameters
            optimizer.zero_grad()
            # computing the updated weights of all the model parameters
            loss_train.backward()
            optimizer.step()            
            
        with torch.no_grad():
            for x_val, y_val in validationLoader:
                model.eval()
    
                yhat = model(x_val)
                val_loss = criterion(y_val, yhat)
                avgValLosses.append(val_loss.item())
                softmax = torch.sigmoid(yhat)
                prob = list(softmax.numpy())
                predictions = np.round(prob)
                avgValScore.append(accuracy_score(y_val, predictions))
                
        avgTrainLoss = np.average(np.array(avgTrainLosses)) 
        avgValLoss = np.average(np.array(avgValLosses))      
        avgValScore = np.average(np.array(avgValScore)) 
        print('Epoch : ',iEpoch+1, '\t' 'Avg. loss :', avgValLoss, '\t' 'Avg. score :',avgValScore)
        return avgTrainLoss,avgValLoss
    return trainSNet  

########################################################################
# %% RNEEP
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

# %% RNEEP
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
    
# %% train RNN
def make_trainRnn(model,optimizer,seqSize,device):
    def trainRnn(trainLoader,validationLoader,iEpoch):
        
        tr_loss = 0
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
##############################################################
    
if __name__ == '__main__':
    # %% Load/Create Db
    # Check if variable exist(dont do nothing)->exist file(load db)->create db
    if not 'dTrain' in locals():
        if os.path.isfile('TrainSet.pickle'):
            with open('TrainSetRnn.pickle', 'rb') as handle:
                dTrain = pickle.load(handle)
            with open('ValidSetRnn.pickle', 'rb') as handle:
                dValid = pickle.load(handle)
            with open('TestSetRnn.pickle', 'rb') as handle:
                dTest = pickle.load(handle)
        else:
            dTrain,dValid,dTest = dbCreate.CreateBasicDataSet(lenOfTraj,1e3,500,100)   
    
    seqSize = 100 # for now it should be divider of 5
    
    # %% Data for CNN
    # # Standardize each trajectory jump times and states
    # mTrain = torch.from_numpy(dTrain['mData']).unsqueeze(1).type(torch.FloatTensor)
    # # Standardize
    # mTrain = torch.div(torch.sub(mTrain,mTrain.mean(dim=2).unsqueeze(2).repeat(1,1,lenOfTraj,1)),mTrain.std(dim=2).unsqueeze(2).repeat(1,1,lenOfTraj,1))
    
    # mValid = torch.from_numpy(dValid['mData']).unsqueeze(1).type(torch.FloatTensor)
    # # Standardize
    # mValid = torch.div(torch.sub(mValid,mValid.mean(dim=2).unsqueeze(2).repeat(1,1,lenOfTraj,1)),mValid.std(dim=2).unsqueeze(2).repeat(1,1,lenOfTraj,1))
    
    # # Standardize each trajectory jump times and states
    # mTest = torch.from_numpy(dTest['mData']).unsqueeze(1).type(torch.FloatTensor)
    # # Standardize
    # # mTest = torch.div(torch.sub(mTest,mTest.mean(dim=2).unsqueeze(2).repeat(1,1,lenOfTraj,1)),mTest.std(dim=2).unsqueeze(2).repeat(1,1,lenOfTraj,1))
    
    # %% Data for RNN
    # Create data for the rnn - (seq_size,batch_size)
    mTrain = dTrain['mData'][:,:,0].reshape(seqSize,-1,order='F').transpose()
    mTrain = torch.from_numpy(mTrain).long()
    vTrainL = np.kron(dTrain['vLabels'],np.ones(int(np.size(dTrain['mData'],1)/seqSize)))
    vTrainL = torch.from_numpy(vTrainL).type(torch.FloatTensor)
    
    # mValid = dValid['mData'][:,:,0].reshape(seqSize,-1,order='F').transpose()
    # mValid = torch.from_numpy(mValid).long()
    # vValidL = np.kron(dValid['vLabels'],np.ones(int(np.size(dValid['mData'],1)/seqSize)))
    # vValidL = torch.from_numpy(vValidL).type(torch.FloatTensor)
    
    # Standardize each trajectory jump times and states
    # mTest = dTest['mData'][:,:,0].reshape(seqSize,-1,order='F').transpose()
    # mTest = torch.from_numpy(mTest).long()
    # vTestL = np.kron(dTest['vLabels'],np.ones(int(np.size(dTest['mData'],1)/seqSize)))
    # vTestL = torch.from_numpy(vTestL).type(torch.FloatTensor)
    
    trainDataSet = torch.utils.data.TensorDataset(mTrain,vTrainL)
    # validDataSet = torch.utils.data.TensorDataset(mValid,vValidL) 
    # testDataSet = torch.utils.data.TensorDataset(torch.from_numpy(dTest['mData']),torch.from_numpy(dTest['vLabels']))   
    trainLoader =  torch.utils.data.DataLoader(trainDataSet, batch_size=int(50), shuffle=True)
    # validLoader =  torch.utils.data.DataLoader(validDataSet, batch_size=int(100), shuffle=True)
    
    # %% Train model
    # defining the model
    # model = SimpleNet(lenOfTraj)
    model = RNEEP()
    # defining the optimizer
    optimizer = Adam(model.parameters(), lr=0.00035)
    # defining the loss function
    criterion = BCEWithLogitsLoss()
    
    # defining the number of epochs
    n_epochs = 2
    # empty list to store training losses
    train_losses = []
    # empty list to store validation losses
    val_losses = []
    # training the model
    trainRnn = make_trainRnn(model,optimizer,seqSize)
    for epoch in range(n_epochs):
        trainLoss = trainRnn(trainLoader,epoch)
        train_losses.append(trainLoss)
        # val_losses.append(avgValLos)
    # %% Analyze results
    # with torch.no_grad():
    #     model.eval()
    #     vTestResults=model(mTest)
    #     lossTest = criterion(vTestResults,vTestL)
    #     softmax = torch.sigmoid(vTestResults)
    #     prob = list(softmax.numpy())
    #     predictions = np.round(prob)
    #     print(accuracy_score(dTest['vLabels'], predictions))

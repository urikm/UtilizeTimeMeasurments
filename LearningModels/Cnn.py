# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 23:25:01 2020

@title: Take.1 - try to resamble entropy prod rate as features of net

@author: Uri Kapustin
"""

# %% imports
import numpy as np

# for evaluating the model
from sklearn.metrics import accuracy_score
# from tqdm import tqdm

# PyTorch libraries and modules
import torch

from torch.nn import Linear, ReLU ,Sequential, Conv2d, Module, BatchNorm2d


# %% Inits
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


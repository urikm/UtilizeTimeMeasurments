# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 00:04:45 2020

@title: KLD entropy production rate estimator(Reproduction of inferred broken detailed balance paper from 2019)
    
@author: Uri Kapustin

@description: This is the main script to run 
"""
import numpy as np

from PhysicalModels.MasterEqSim import MasterEqSolver as MESolver
import PhysicalModels.PartialTrajectories as pt 
import LearningModels.Neep as neep

import torch
from torch.optim import Adam, SGD, Adagrad, RMSprop , Rprop

import pickle
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

# %% Comparing KLD estimator to previous
if __name__ == '__main__':
    ## UI
    # 
    rneeptFlag = False # True - use time data ; False - only states data
    vSeqSize = np.array([3,16,32,64,128])
    # vSeqSize = np.array([128])
    # vLrate = np.array([1e-3,1e-3,5e-3,1e-2,5e-2])
    maxSeqSize = np.max(vSeqSize)
    batchSize = 64#4096
    vEpochs = 5*np.array([1,5,10,20,40]) # raising in order to keep same # of iterations for each seq size
    
    flagPlot = True
    nDim = 4 # dimension of the problem
    nTimeStamps = int(maxSeqSize*batchSize*5e0) # how much time stamps will be saved
    vHiddenStates = np.array([2,3]) # states 3 and 4 for 4-D state sytem
    
    ## Define base dynamics
    if 0:
        mW = pt.GenRateMat(nDim) # transition matrix
        timeRes = 1
    else:
        mW = np.array([[-11.,2.,0.,1.],[3.,-52.2,2.,35.],[0.,50.,-77.,0.7],[8.,0.2,75.,-36.7]])
        timeRes = 0.001

    # Calculate Stalling data
    vPiSt,xSt,r01,r10  = pt.CalcStallingData(mW)    
    # Init vectors for plotting
    vGrid = np.concatenate((np.arange(-1.,xSt,1),np.arange(xSt,3.,1)))
    # This used for running with different grid, dont change the upper version - its the defualt
    # vGrid = np.concatenate((np.arange(xSt-0.05,xSt-0.005,0.01),np.arange(xSt,xSt+0.05,0.01)))

    # vGrid = np.arange(-2.,0.,0.5)
    vInformed = np.zeros(np.size(vGrid))
    vPassive = np.zeros(np.size(vGrid))
    vKld =  np.zeros(np.size(vGrid))
    vKldValid =  np.zeros(np.size(vGrid))
    vFull = np.zeros(np.size(vGrid))
    mNeep = np.zeros([np.size(vSeqSize),np.size(vGrid)])
    # define RNN model
    # model = SimpleNet(lenOfTraj)
    # model = neep.RNEEP()
    # # defining the optimizer
    # optimizer = Adam(model.parameters(), lr=0.00045,weight_decay=0.00005)

    
    # trainRnn = neep.make_trainRnn(model,optimizer,seqSize)
    print("Used device is:"+device)
    i=0
    for x in vGrid: 
        mWx = pt.CalcW4DrivingForce(mW,x) # Calculate W matrix after applying force
        # Passive partial entropy production rate
        vP0 = np.random.uniform(size=(nDim))
        vP0 = vP0/sum(vP0)
        n,vPiX,mWx,vWPn = MESolver(nDim,vP0,mWx,timeRes)
        vPassive[i] = pt.CalcPassivePartialEntropyProdRate(mWx,vPiX)
        # Informed partial entropy production rate
        vInformed[i] = pt.CalcInformedPartialEntropyProdRate(mWx,vPiX,vPiSt)
        # The full entropy rate
        vFull[i] = pt.EntropyRateCalculation(nDim,mWx,vPiX)
        # KLD bound
        mCgTrajectory,nCgDim = pt.CreateCoarseGrainedTraj(nDim,nTimeStamps,mWx,vHiddenStates,timeRes)
        sigmaDotKld,T,sigmaDotAff,sigmaWtd,dd1H2,dd2H1 = pt.CalcKLDPartialEntropyProdRate(mCgTrajectory,nCgDim)
        vKld[i] = sigmaDotKld
        mCgTrajValid,_ = pt.CreateCoarseGrainedTraj(nDim,nTimeStamps,mWx,vHiddenStates,timeRes)
        sigmaDotKld,T,sigmaDotAff,sigmaWtd,dd1H2,dd2H1 = pt.CalcKLDPartialEntropyProdRate(mCgTrajectory,nCgDim)
        vKldValid[i] = sigmaDotKld        
        k = 0       
        for iSeqSize in vSeqSize:
            if rneeptFlag == False:
            # ==============================================
            # # NEEP entropy rate
                mTrain = mCgTrajectory[:int(np.floor(mCgTrajectory.shape[0]/iSeqSize)*iSeqSize),0].reshape(iSeqSize,-1,order='F').transpose()
                mTrain = torch.from_numpy(mTrain).long()
                vTrainL = np.kron(vKld[i]*T,np.ones(int(np.floor(mCgTrajectory.shape[0]/iSeqSize))))
                vTrainL = torch.from_numpy(vTrainL).type(torch.FloatTensor)
                trainDataSet = torch.utils.data.TensorDataset(mTrain,vTrainL)
                # trainLoader =  torch.utils.data.DataLoader(trainDataSet, batch_size=batchSize, shuffle=True)
    
                mValid = mCgTrajValid[:int(np.floor(mCgTrajValid.shape[0]/iSeqSize)*iSeqSize),0].reshape(iSeqSize,-1,order='F').transpose()
                mValid = torch.from_numpy(mValid).long()
                vValidL = np.kron(vKldValid[i]*T,np.ones(int(np.floor(mCgTrajValid.shape[0]/iSeqSize))))
                vValidL = torch.from_numpy(vValidL).type(torch.FloatTensor)
                validDataSet = torch.utils.data.TensorDataset(mValid,vValidL)
                validLoader =  torch.utils.data.DataLoader(validDataSet, batch_size=batchSize, shuffle=False)
            else:   
            # ==============================================
            # NEEP entropy rate using time
            # Obsolete Naive sampler - TODO : implement more "standard" sampler   
                tmpStates = mCgTrajectory[:int(np.floor(mCgTrajectory.shape[0]/iSeqSize)*iSeqSize),0].reshape(iSeqSize,-1,order='F').transpose()
                tmpWtd = mCgTrajectory[:int(np.floor(mCgTrajectory.shape[0]/iSeqSize)*iSeqSize),1].reshape(iSeqSize,-1,order='F').transpose()
       
                mTrain = np.concatenate((np.expand_dims(tmpStates,2),np.expand_dims(tmpWtd,2)),axis=2)
                mTrain = torch.from_numpy(mTrain).float()
                # vTrainL = np.kron(vKld[i]*T,np.ones(int(.7*nTimeStamps/seqSize)))
                vTrainL = np.kron(vKld[i]*T,np.ones(int(np.floor(mCgTrajectory.shape[0]/iSeqSize))))
                vTrainL = torch.from_numpy(vTrainL).float()
                trainDataSet = torch.utils.data.TensorDataset(mTrain,vTrainL)
                # trainLoader =  torch.utils.data.DataLoader(trainDataSet, batch_size=batchSize, shuffle=True)
                
                tmpSValid = mCgTrajValid[:int(np.floor(mCgTrajValid.shape[0]/iSeqSize)*iSeqSize),0].reshape(iSeqSize,-1,order='F').transpose()
                tmpWValid = mCgTrajValid[:int(np.floor(mCgTrajValid.shape[0]/iSeqSize)*iSeqSize),1].reshape(iSeqSize,-1,order='F').transpose()
       
                mValid = np.concatenate((np.expand_dims(tmpSValid,2),np.expand_dims(tmpWValid,2)),axis=2)
                mValid = torch.from_numpy(mValid).float()
                # vTrainL = np.kron(vKld[i]*T,np.ones(int(.7*nTimeStamps/seqSize)))
                vValidL = np.kron(vKldValid[i]*T,np.ones(int(np.floor(mCgTrajValid.shape[0]/iSeqSize))))
                vValidL = torch.from_numpy(vValidL).float()
                validDataSet = torch.utils.data.TensorDataset(mValid,vValidL)
                validLoader =  torch.utils.data.DataLoader(validDataSet, batch_size=batchSize, shuffle=False)
            # ==============================================            
            
            print('Calculating estimator for x =',x,'Sequence size:',iSeqSize)
            # define RNN model
            
            if rneeptFlag == False:
                model = neep.RNEEP().to(device)
                outFileadd =''
            else:
                model = neep.RNEEPT().to(device)
                outFileadd ='T_'
            if device == 'cuda':
                model = torch.nn.parallel.DistributedDataParallel(model,device_ids=list(range(torch.cuda.device_count())))
            # defining the optimizer
            # optimizer = SGD(model.parameters(),lr=vLrate[k])
            optimizer = Adam(model.parameters(),lr=1e-4,weight_decay=0.5e-4)
            trainRnn = neep.make_trainRnn(model,optimizer,iSeqSize,device)
            bestLoss = 1e3
            for epoch in range(int(vEpochs[k])):
                trainLoader =  torch.utils.data.DataLoader(trainDataSet, batch_size=batchSize, shuffle=False)
                bestLossEpoch,bestEpRate,bestEpErr = trainRnn(trainLoader,validLoader,epoch)/T
                if bestLossEpoch < bestLoss:
                    mNeep[k,i] = bestEpRate
                    bestLoss = bestLossEpoch
            k += 1   
         

        i += 1
        
# %% Save results
        with open('vInformed_x_'+outFileadd+str(i-1)+'.pickle', 'wb') as handle:
            pickle.dump(vInformed, handle)
        with open('vPassive_x_'+outFileadd+str(i-1)+'.pickle', 'wb') as handle:
            pickle.dump(vPassive, handle)
        with open('vKld_x_'+outFileadd+str(i-1)+'.pickle', 'wb') as handle:
            pickle.dump(vKld, handle)    
        with open('mNeep_x_'+outFileadd+str(i-1)+'.pickle', 'wb') as handle:
            pickle.dump(mNeep, handle)            
        

# %% model size
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

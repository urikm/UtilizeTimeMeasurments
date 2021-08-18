# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 00:04:45 2020

@title: KLD entropy production rate estimator(Reproduction of inferred broken detailed balance paper from 2019)
    
@author: Uri Kapustin

@description: This is the main script to run 

"""
import argparse

import numpy as np
import time
import os

from PhysicalModels.MasterEqSim import MasterEqSolver as MESolver
import PhysicalModels.PartialTrajectories as pt 
import LearningModels.Neep as neep

import torch
import torch.utils.data as dat
from torch.optim import Adam, SGD, Adagrad, RMSprop , Rprop

import pickle
 
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

# %% Comparing KLD estimator to previous
if __name__ == '__main__':
	## Handle parsing arguments
    parser = argparse.ArgumentParser(description="Hidden Markov EPR estimation using NEEP")
    parser.add_argument("--save-path",default="",type=str,metavar="PATH",help="path to save result (default: none)")	
    opt = parser.parse_args()	
    ## UI
    # 
    loadDbFlag = True # True - read dataset from file; False - create new(very slow)
    rneeptFlag = False # True - use time data ; False - only states data
    plotDir = opt.save_path#'Results'+os.sep+'Analysis_0'
    try:
        os.mkdir(plotDir)
    except:
        pass
    dbName = 'RneepDbCoarse'
    dbPath = 'StoredDataSets'+os.sep+dbName
    dbFileName = 'InitRateMatAsGilis'
    logFile = 'log.txt'

    vSeqSize = np.array([3,16,32,64,128])
    # vSeqSize = np.array([128])
    maxSeqSize = np.max(vSeqSize)
    batchSize = 4096
    nEpochs = 1
    
    flagPlot = True
    nDim = 4 # dimension of the problem
    nTimeStamps = int(maxSeqSize*batchSize*1e2) # how much time stamps will be saved
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
    

    print("Used device is:"+device)
    
    # For each driving force in vGrid, estimate the ERP 
    i=0
    for idx,x in enumerate(vGrid): 
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
        if loadDbFlag:
            # TODO : use different data for validation
            # TODO : support for using Time Data??
            # Choose the wanted trajectory according to x
            with open(dbPath+os.sep+'MappingVector'+'.pickle', 'rb') as handle:
                vX = pickle.load(handle)
                wantedIdx = (np.abs(vX - x)).argmin()
            with open(dbPath+os.sep+dbFileName+'_'+str(wantedIdx)+'.pickle', 'rb') as handle:
                dDataTraj = pickle.load(handle)

            mCgTrajectory = dDataTraj.pop('vStates')
            mCgTrajectory = np.array([mCgTrajectory,dDataTraj.pop('vTimeStamps')]).T
            vKld[i] = dDataTraj['kldBound']
            mCgTrajValid = mCgTrajectory
            vKldValid[i] = vKld[i]
            T = dDataTraj['timeFactor']
        else:   
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
                mTrain = torch.from_numpy(mCgTrajectory[:,0]).long()
                vTrainL = np.kron(vKld[i]*T,np.ones(len(mTrain)))
                vTrainL = torch.from_numpy(vTrainL).type(torch.FloatTensor)
                trainDataSet = torch.utils.data.TensorDataset(mTrain,vTrainL)
                # trainLoader =  torch.utils.data.DataLoader(trainDataSet, batch_size=batchSize, shuffle=True)
    
                # mValid = mCgTrajValid[:int(np.floor(mCgTrajValid.shape[0]/iSeqSize)*iSeqSize),0].reshape(iSeqSize,-1,order='F').transpose()
                mValid = torch.from_numpy(mCgTrajValid[:,0]).long()
                vValidL = np.kron(vKldValid[i]*T,np.ones(len(mValid)))
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
                vTrainL = np.kron(vKld[i]*T,np.ones(int(np.floor(mCgTrajectory.shape[0]/iSeqSize))))
                vTrainL = torch.from_numpy(vTrainL).float()
                trainDataSet = torch.utils.data.TensorDataset(mTrain,vTrainL)
                # trainLoader =  torch.utils.data.DataLoader(trainDataSet, batch_size=batchSize, shuffle=True)
                
                tmpSValid = mCgTrajValid[:int(np.floor(mCgTrajValid.shape[0]/iSeqSize)*iSeqSize),0].reshape(iSeqSize,-1,order='F').transpose()
                tmpWValid = mCgTrajValid[:int(np.floor(mCgTrajValid.shape[0]/iSeqSize)*iSeqSize),1].reshape(iSeqSize,-1,order='F').transpose()
       
                mValid = np.concatenate((np.expand_dims(tmpSValid,2),np.expand_dims(tmpWValid,2)),axis=2)
                mValid = torch.from_numpy(mValid).float()
                vValidL = np.kron(vKldValid[i]*T,np.ones(int(np.floor(mCgTrajValid.shape[0]/iSeqSize))))
                vValidL = torch.from_numpy(vValidL).float()
                validDataSet = torch.utils.data.TensorDataset(mValid,vValidL)
                validLoader =  torch.utils.data.DataLoader(validDataSet, batch_size=batchSize, shuffle=False)
            # ==============================================            
            
            print('Calculating estimator for x = '+str(x)+' ; Sequence size: '+str(iSeqSize))
            # define RNN model
            
            if rneeptFlag == False:
                model = neep.RNEEP()
                outFileadd =''
            else:
                model = neep.RNEEPT()
                outFileadd ='T_'
            if device == 'cuda:0':
                model = torch.nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count())))
            model.to(device)
            # defining the optimizer
            # optimizer = SGD(model.parameters(),lr=vLrate[k])
            optimizer = Adam(model.parameters(),lr=1e-4,weight_decay=0.5e-4)
            trainRnn = neep.make_trainRnn(model,optimizer,iSeqSize,device)
            bestLoss = 1e3
            
            # Define sampler
            baseInd=np.repeat(np.array([range(len(mTrain)),]),iSeqSize)
            addInd=np.repeat(np.array([range(iSeqSize),]),len(mTrain),axis=0).flatten()
            seqInd=baseInd+addInd
            batchInd=seqInd.reshape(-1,iSeqSize)
            for epoch in range(int(nEpochs)):
                validLoader =  torch.utils.data.DataLoader(validDataSet, batch_size=batchSize,sampler=batchInd)
                trainLoader =  torch.utils.data.DataLoader(trainDataSet, batch_size=batchSize,sampler=np.random.permutation(batchInd))
                tic = time.time()
                bestLossEpoch,bestEpRate,bestEpErr = trainRnn(trainLoader,validLoader,epoch)/T
                toc = time.time()
                print('Elapsed time of Epoch '+str(epoch)+' is: '+str(toc-tic))
                if bestLossEpoch < bestLoss:
                    mNeep[k,i] = bestEpRate
                    bestLoss = bestLossEpoch
            k += 1   
         

        i += 1
        
# %% Save results
        print("DB mNeep:"+str(mNeep))
        with open(plotDir+os.sep+'vInformed_x_'+outFileadd+str(i-1)+'.pickle', 'wb') as handle:
            pickle.dump(vInformed, handle)
        with open(plotDir+os.sep+'vPassive_x_'+outFileadd+str(i-1)+'.pickle', 'wb') as handle:
            pickle.dump(vPassive, handle)
        with open(plotDir+os.sep+'vKld_x_'+outFileadd+str(i-1)+'.pickle', 'wb') as handle:
            pickle.dump(vKld, handle)    
        with open(plotDir+os.sep+'mNeep_x_'+outFileadd+str(i-1)+'.pickle', 'wb') as handle:
            pickle.dump(mNeep, handle)            
        


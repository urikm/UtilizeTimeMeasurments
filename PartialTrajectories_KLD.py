# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 00:04:45 2020

@title: KLD entropy production rate estimator(Reproduction of inferred broken detailed balance paper from 2019)
    
@author: Uri Kapustin
"""
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity as KD
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from scipy.interpolate import interp1d
# import scipy.stats as stats
from MasterEqSim import MasterEqSolver as MESolver
from TrajectoryCreation import *
from PartialTrajectories import  * 
import CnnFirstTry as neep

import torch
from torch.optim import Adam, SGD, Adagrad, RMSprop , Rprop

import pickle
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
# %% Coarse Grain trajectory
def CoarseGrainTrajectory(mTrajectory,nFullDim,vHiddenStates):
    mCgTrajectory = np.copy(mTrajectory)
    nJumps = np.size(mCgTrajectory,0)
    iHidden = nFullDim-np.size(vHiddenStates,0)
    hState = -2 #iHidden #  define as intermediate state every hidden state that is not last in it's sequence
    
    for iJump in range(nJumps):
        
        if (mCgTrajectory[iJump,0] in vHiddenStates):
            if (iJump < nJumps-1):
                    if (mCgTrajectory[iJump+1,0] in vHiddenStates):
                        mCgTrajectory[iJump,0] = hState
                        mCgTrajectory[iJump+1,1] += mCgTrajectory[iJump,1] # cumsum waiting times for each sequence
                    else:
                         mCgTrajectory[iJump,0] = iHidden                                              
            else:
                mCgTrajectory[iJump,0] = iHidden
            
    # Now remove '-2' states
    mCgTrajectory = mCgTrajectory[(mCgTrajectory[:,0]!=hState),:]  # TODO: uncomment for right CG   
    nCgDim = iHidden+1
    return mCgTrajectory,nCgDim

# Estimate 2nd order statistics of the trajectory (i->j->k transitions)
# TODO: make competible for every combination , now competible only for 2019's paper example
def EstimateTrajParams2ndOrder(nDim,mTrajectory):
    mP2ndOrderTransitions = np.zeros((nDim,) * nDim) # all the Pijk - 3D tenzor
    mWtd = []
    vInitStates = np.arange(nDim)
    for iState in vInitStates:
        vIndInitTrans = np.array(np.where(mTrajectory[:-2,0] == iState)[0])
        nTrans4State = np.size(vIndInitTrans,0)
        vFirstTrans = np.roll(vInitStates,-iState)[1:]
        for jState in vFirstTrans:
            vIndFirstTrans = vIndInitTrans[np.array(np.where(mTrajectory[vIndInitTrans + 1,0] == jState)[0])]+1
            # find which is the k state ( assuming only 3 states!!!)
            kState = np.argwhere((vInitStates!=iState)&(vInitStates!=jState))[0][0]
            vIndSecondTrans = vIndFirstTrans[np.array(np.where(mTrajectory[vIndFirstTrans + 1,0] == kState)[0])]+1
            mP2ndOrderTransitions[iState,jState,kState] = np.size(vIndSecondTrans,0)/(nTrans4State+2)
            if (jState == 2) & (iState != 2) & (kState != 2) & (iState != kState):
                mWtd.append(mTrajectory[vIndSecondTrans-1,1]) 
            
    return mP2ndOrderTransitions, mWtd
# Calculate KLD entropy production rate as explained in 2019  paper
def CalcKLDPartialEntropyProdRate(mCgTrajectory,nDim):
    # First estimate all statistics from hidden trajectory
    mIndStates,mWaitTimes,vEstLambdas,mWest,vSS = EstimateTrajParams(nDim,mCgTrajectory)
    mP2ndOrdTrans, mWtd = EstimateTrajParams2ndOrder(nDim,mCgTrajectory)
    
    # Calculate common paramerters
    vTau = np.zeros(3)
    vTau[0] = np.sum(mWaitTimes[0])/np.size(mWaitTimes[0],0)
    vTau[1] = np.sum(mWaitTimes[1])/np.size(mWaitTimes[1],0)
    vTau[2] = np.sum(mWaitTimes[2])/np.size(mWaitTimes[2],0)
    
    
    vR = np.zeros(3)
    nTot = np.size(mIndStates[0],0)+np.size(mIndStates[1],0)+np.size(mIndStates[2],0)
    vR[0] = np.size(mIndStates[0],0)/nTot
    vR[1] = np.size(mIndStates[1],0)/nTot
    vR[2] = np.size(mIndStates[2],0)/nTot
    
    T = np.dot(vTau,vR)
    
    ## Find affinity part 
    # Math: R12 = p21*R[1] = (tau[1]*w21)*(Pi[1]*T/tau[1])=w21*Pi[1]*T
    R12 = mWest[1,0]*vSS[0]*T
    R13 = mWest[2,0]*vSS[0]*T
    R21 = mWest[0,1]*vSS[1]*T
    R23 = mWest[2,1]*vSS[1]*T
    R31 = mWest[0,2]*vSS[2]*T
    R32 = mWest[1,2]*vSS[2]*T
    # Pijk = Pr{to observe i>j>k} => Pijk=R[ijk]*R[i](this probaibility related to markoc chain, not time related)
    p12_23 = mP2ndOrdTrans[0,1,2]*vR[0]/R12
    p23_31 = mP2ndOrdTrans[1,2,0]*vR[1]/R23
    p31_12 = mP2ndOrdTrans[2,0,1]*vR[2]/R31
    p13_32 = mP2ndOrdTrans[0,2,1]*vR[0]/R13
    p32_21 = mP2ndOrdTrans[2,1,0]*vR[2]/R32
    p21_23 = mP2ndOrdTrans[1,0,2]*vR[1]/R21
    
    sigmaDotAff = ((R12-R21)/T*np.log(p12_23*p23_31*p31_12/p13_32/p32_21/p21_23))
    
    ## Find Wtd part
    p1H2 = mP2ndOrdTrans[0,2,1]*vR[0]
    p2H1 = mP2ndOrdTrans[1,2,0]*vR[1]
    ## Use KDE to build Psi functions
    # First estimate bandwidths
    # bandwidths = np.linspace(-0.1, 0.1, 20)
    # grid = GridSearchCV(KD(kernel='gaussian'),{'bandwidth': bandwidths},cv=LeaveOneOut())
    # grid.fit(np.int64(mWtd[0][:, None]))
    # b1H2 = grid.best_params_
    # grid.fit(np.int64(mWtd[1][:, None]))
    # b2H1 = grid.best_params_
    b1H2 = 0.0043 # manually fixed after running some optimization, see the lines commented before
    b2H1 = b1H2
    # Define density destribution grid
    vGridDest = np.linspace(0,0.2,100)
    # kde1H2 = KD(bandwidth=b1H2['bandwidth'])
    # kde2H1 = KD(bandwidth=b2H1['bandwidth'])
    kde1H2 = KD(bandwidth=b1H2)
    kde2H1 = KD(bandwidth=b2H1)
    kde1H2.fit(mWtd[0][:,None])
    kde2H1.fit(mWtd[1][:,None])
    dd1H2 = np.exp(kde1H2.score_samples(vGridDest[:,None])) # density distribution 1->H->2
    dd2H1 = np.exp(kde2H1.score_samples(vGridDest[:,None])) # density distribution 2->H->1
    pDd1H2 = dd1H2/np.sum(dd1H2) # Probability density distribution
    pDd2H1 = dd2H1/np.sum(dd2H1) # Probability density distribution
    kld1H2 = np.sum(np.multiply(pDd1H2,np.log(np.divide(pDd1H2,pDd2H1))))
    kld2H1 = np.sum(np.multiply(pDd2H1,np.log(np.divide(pDd2H1,pDd1H2))))
    
    sigmaDotWtd = (p1H2*kld1H2+p2H1*kld2H1)/T
    
    sigmaDotKld = sigmaDotAff + sigmaDotWtd
    return sigmaDotKld,T,sigmaDotAff,sigmaDotWtd,dd1H2,dd2H1

def CreateCoarseGrainedTraj(nDim,nTimeStamps,mW,vHiddenStates,timeRes):
    # randomize init state from the steady-state distribution
    vP0 = np.array([0.25,0.25,0.25,0.25])
    n,vPi,mW,vWPn = MESolver(nDim,vP0,mW,timeRes)
    initState = np.random.choice(nDim,1,p=vPi)
    # Create trajectory
    mTrajectory, mW = CreateTrajectory(nDim,nTimeStamps,initState,mW) # Run Create Trajectory
    mCgTrajectory,nCgDim = CoarseGrainTrajectory(mTrajectory,nDim,vHiddenStates)   
    return mCgTrajectory,nCgDim
# %% Comparing KLD estimator to previous
if __name__ == '__main__':
    ## UI
    # 
    rneeptFlag = False # True - use time data ; False - only states data
    vSeqSize = np.array([3,16,32,64,128])
    # vSeqSize = np.array([128])
    # vLrate = np.array([1e-3,1e-3,5e-3,1e-2,5e-2])
    maxSeqSize = np.max(vSeqSize)
    batchSize = 4096
    vEpochs = 5*np.array([1,5,10,20,40]) # raising in order to keep same # of iterations for each seq size
    
    flagPlot = True
    nDim = 4 # dimension of the problem
    nTimeStamps = int(maxSeqSize*batchSize*5e0) # how much time stamps will be saved
    vHiddenStates = np.array([2,3]) # states 3 and 4 for 4-D state sytem
    
    ## Define base dynamics
    if 0:
        mW = GenRateMat(nDim) # transition matrix
        timeRes = 1
    else:
        mW = np.array([[-11.,2.,0.,1.],[3.,-52.2,2.,35.],[0.,50.,-77.,0.7],[8.,0.2,75.,-36.7]])
        timeRes = 0.001

    # Calculate Stalling data
    vPiSt,xSt,r01,r10  = CalcStallingData(mW)    
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
        mWx = CalcW4DrivingForce(mW,x) # Calculate W matrix after applying force
        # Passive partial entropy production rate
        vP0 = np.random.uniform(size=(nDim))
        vP0 = vP0/sum(vP0)
        n,vPiX,mWx,vWPn = MESolver(nDim,vP0,mWx,timeRes)
        vPassive[i] = CalcPassivePartialEntropyProdRate(mWx,vPiX)
        # Informed partial entropy production rate
        vInformed[i] = CalcInformedPartialEntropyProdRate(mWx,vPiX,vPiSt)
        # The full entropy rate
        vFull[i] = EntropyRateCalculation(nDim,mWx,vPiX)
        # KLD bound
        mCgTrajectory,nCgDim = CreateCoarseGrainedTraj(nDim,nTimeStamps,mWx,vHiddenStates,timeRes)
        sigmaDotKld,T,sigmaDotAff,sigmaWtd,dd1H2,dd2H1 = CalcKLDPartialEntropyProdRate(mCgTrajectory,nCgDim)
        vKld[i] = sigmaDotKld
        mCgTrajValid,_ = CreateCoarseGrainedTraj(nDim,nTimeStamps,mWx,vHiddenStates,timeRes)
        sigmaDotKld,T,sigmaDotAff,sigmaWtd,dd1H2,dd2H1 = CalcKLDPartialEntropyProdRate(mCgTrajectory,nCgDim)
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
        
# %% Plot for Single sequence - cannot be used in terminal  
    # plt.plot(np.flip(-vGrid),np.flip(vKld),'b-.')    
    # plt.plot(np.flip(-vGrid),np.flip(vInformed),'k.')
    # plt.plot(np.flip(-vGrid),np.flip(vPassive),'g.')
    # plt.plot(np.flip(-vGrid),np.flip(vNeep),'m:') 
    # plt.plot(np.flip(-vGrid),np.flip(vFull),'r')   
    
    # plt.yscale('log')
    # plt.xlabel('x - Driving Force')
    # plt.ylabel('Entropy Production rate')
    # plt.legend(['KLD','Informed','Passive','NEEP(no time data)','Total - Full Trajectory'])
    # plt.show()
# %% model size
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# %% plot partial data
    # nLast = 50
    # mTrajec2Plot = mTrajectory[:nLast,:]
    # fStatesFTraj = interp1d(np.cumsum(mTrajec2Plot[:,1])-mTrajec2Plot[0,1],mTrajec2Plot[:,0],kind='zero')
    # vXinterp = np.arange(0,np.max(np.cumsum(mTrajec2Plot[:,1])-mTrajec2Plot[0,1]),0.001)
    # plt.plot(vXinterp,fStatesFTraj(vXinterp))
    
    # mTrajec2PlotCg = mCgTrajectory[:nLast,:]
    # fStatesFTrajCg = interp1d(np.cumsum(mTrajec2PlotCg[:,1])-mTrajec2PlotCg[0,1],mTrajec2PlotCg[:,0],kind='zero')
    # vXinterpCg = np.arange(0,np.max(np.cumsum(mTrajec2PlotCg[:,1])-mTrajec2PlotCg[0,1]),0.001)
    # plt.plot(vXinterpCg,fStatesFTrajCg(vXinterpCg))
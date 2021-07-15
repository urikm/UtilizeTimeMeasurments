# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 07:05:53 2021

@title: DataSets Module

@author: Uri Kapustin

@description: This module creates synthetic databases of trajectories according to wanted physical models
"""
# %% TODO : This module need some "dust removal", it probably wont run 

# %% imports
import numpy as np
import pickle
import os

# import scipy.stats as stats
from PhysicalModels.UtilityTraj import GenRateMat
from PhysicalModels.MasterEqSim import MasterEqSolver as MESolver
from PhysicalModels.TrajectoryCreation import EstimateTrajParams
import PhysicalModels.PartialTrajectories as pt

# %% Calculate dual rate matrix
def CalcDualRateMat(mW,mPartTraj,nCgDim):
    mWdual = np.copy(mW)
    # Find steady state
    _ , _ , _ , _ ,vPss = EstimateTrajParams(nCgDim,mPartTraj)
    # Create Dual rate matrix
    mWdual = mWdual.transpose()
    mWdual[0,1] *= vPss[0]/vPss[1]
    mWdual[0,2] *= vPss[0]/vPss[2]
    mWdual[1,0] *= vPss[1]/vPss[0]
    mWdual[1,2] *= vPss[1]/vPss[2]
    mWdual[2,0] *= vPss[2]/vPss[0]
    mWdual[2,1] *= vPss[2]/vPss[1]
    
    return mWdual

# %% Create Data set for simple CNN to try resamble EPR for hidden network system
def CreateForwardSet(nTraj,lenTraj):
    # nTraj is the number of trajectories for backwrad or forward process - in total will be created 2*nTraj trajectories

    # Degenrated problem - assuming 4 states and known hidden states
    nDim = 4
    vHiddenStates = np.array([2,3])
      
    mForwardTrajs = np.zeros([nTraj,lenTraj,2])
    len4Creation = 3*lenTraj # because we intersted in coarsed grained traj, we need sample much longer traj in order to cut lenTraj 
    fInd = 10 # first index
    lInd = fInd+lenTraj
    for iTraj in range(nTraj):
        mW = np.array([[-11.,2.,0.,1.],[3.,-52.2,2.,35.],[0.,50.,-77.,0.7],[8.,0.2,75.,-36.7]])
        mFTrajsTmp,nCgDim=pt.CreateCoarseGrainedTraj(nDim,len4Creation,mW,vHiddenStates,0.01)
        mForwardTrajs[iTraj,:,:] = mFTrajsTmp[fInd:lInd,:] 
        
    return mForwardTrajs
    
def CreateForwardBackwardSet(nTraj,lenTraj):
    # nTraj is the number of trajectories for backwrad or forward process - in total will be created 2*nTraj trajectories

    # Degenrated problem - assuming 4 states and known hidden states
    nDim = 4
    vHiddenStates = np.array([2,3])
      
    mForwardTrajs = np.zeros([nTraj,lenTraj,2])
    mBackwardTrajs = np.zeros([nTraj,lenTraj,2])
    len4Creation = 3*lenTraj # because we intersted in coarsed grained traj, we need sample much longer traj in order to cut lenTraj 
    fInd = lenTraj-int(4*lenTraj/5) # first index
    lInd = fInd+lenTraj
    for iTraj in range(nTraj):
        mW = GenRateMat(nDim)
        mFTrajsTmp,nCgDim=pt.CreateCoarseGrainedTraj(nDim,len4Creation,mW,vHiddenStates,0.01)
        mWdual = CalcDualRateMat(mW,mFTrajsTmp,nCgDim)
        mBTrajsTmp,nCgDim=pt.CreateCoarseGrainedTraj(nDim,len4Creation,mWdual,vHiddenStates,0.01)
        mForwardTrajs[iTraj,:,:] = mFTrajsTmp[fInd:lInd,:] 
        mBackwardTrajs[iTraj,:,:] = mBTrajsTmp[fInd:lInd,:] 
        
    return mForwardTrajs,mBackwardTrajs

def CreateBasicDataSet(lenTraj,nTrain,nValid,nTest):
    lenTraj = int(lenTraj)
    nTrain = int(nTrain)
    nValid = int(nValid)
    nTest = int(nTest)
    # Common inits
    dTrain = {'mData':[],'vLabels':[]}
    dValid = dict(dTrain)
    dTest = dict(dValid)
    
    # Create Train set
    mForwardTrajs,mBackwardTrajs = CreateForwardBackwardSet(nTrain,lenTraj)
    dTrain['mData'] = np.concatenate((mForwardTrajs,mBackwardTrajs))
    dTrain['vLabels'] = np.concatenate((np.zeros(nTrain),np.ones(nTrain)))
    
    # Create Valid set
    mForwardTrajs,mBackwardTrajs = CreateForwardBackwardSet(nValid,lenTraj)
    dValid['mData'] = np.concatenate((mForwardTrajs,mBackwardTrajs))
    dValid['vLabels'] = np.concatenate((np.zeros(nValid),np.ones(nValid)))
    
    # Create Test set
    mForwardTrajs,mBackwardTrajs = CreateForwardBackwardSet(nTest,lenTraj)
    dTest['mData'] = np.concatenate((mForwardTrajs,mBackwardTrajs))
    dTest['vLabels'] = np.concatenate((np.zeros(nTest),np.ones(nTest)))
    
    return dTrain,dValid,dTest
    
def CreateRnnDataSet(lenTraj):
    # For estimation of the entropy rate
    lenTraj = int(lenTraj)
    nTrain = 1
    nValid = 1
    nTest = 1
    # Common inits
    dTrain = {'mData':[],'vLabels':[]}
    dValid = dict(dTrain)
    dTest = dict(dValid)
    
    # Create Train set
    mForwardTrajs = CreateForwardSet(nTrain,lenTraj)
    sigmaDotKld,T,_,_,_,_ = pt.CalcKLDPartialEntropyProdRate(np.squeeze(mForwardTrajs),3)
    dTrain['mData'] = mForwardTrajs
    dTrain['vLabels'] = sigmaDotKld*T # per step
    
    # Create Valid set
    mForwardTrajs = CreateForwardSet(nValid,lenTraj)
    sigmaDotKld,T,_,_,_,_ = pt.CalcKLDPartialEntropyProdRate(np.squeeze(mForwardTrajs),3)
    dValid['mData'] = mForwardTrajs
    dValid['vLabels'] = sigmaDotKld*T # per step
    
    # Create Test set
    sigmaDotKld,T,_,_,_,_ = pt.CalcKLDPartialEntropyProdRate(np.squeeze(mForwardTrajs),3)
    dTest['mData'] = mForwardTrajs
    dTest['vLabels'] = sigmaDotKld*T # per step
  
    return dTrain,dValid,dTest


# This dataset used for analysing the RNEEP implementation on markov chains with rate matrix from Gili's 2019 paper
def CreateRNeepDataSet(nTimeStamps,mW,vX):
    # The Dataset will be created for different applied forces(vX) and will consist of mDataset which will hold the trajectories(size of nTimeStamps)
    # with their KLD per step.
    # Also there will be mapping vector(vX) whose elements are the applied force on i'th trajectory(mDataset[i][2:]) 
    
    # Base rate matrix
    nDim=4
    vHiddenStates = np.array([2,3]) # states 3 and 4 for 4-D state syte
    timeRes = 0.001
    dDataSet = {'mDataSet':[],'vLabels':vX,'vTimeFactors':[]} # TimeFactors saved for further debug or any need to convert per step->per time
    
    for x in vX:
        mWx = pt.CalcW4DrivingForce(mW,x) # Calculate W matrix after applying force
        # Passive partial entropy production rate
        vP0 = np.random.uniform(size=(nDim))
        vP0 = vP0/sum(vP0)
        mCgTrajectory,nCgDim = pt.CreateCoarseGrainedTraj(nDim,nTimeStamps,mWx,vHiddenStates,timeRes)
        sigmaDotKld,T,sigmaDotAff,sigmaWtd,dd1H2,dd2H1 = pt.CalcKLDPartialEntropyProdRate(mCgTrajectory,nCgDim)        
        vCgTrajectory = mCgTrajectory[:,0]
        dDataSet['mDataSet'].append(np.insert(vCgTrajectory,0,sigmaDotKld*T))
        dDataSet['vTimeFactors'].append(T)
    
    return dDataSet
##########################################################

if __name__ == '__main__':
    # Define estimated size of trajectories(this will be the size of full trajectory which will be coarse grained)
    nTimeStamps = int(4096*128*1e2) # batchSize X maxBatchSize X minNumBatches
    # Define our physical rate matrix
    mW = np.array([[-11.,2.,0.,1.],[3.,-52.2,2.,35.],[0.,50.,-77.,0.7],[8.,0.2,75.,-36.7]])
    vPiSt,xSt,r01,r10  = pt.CalcStallingData(mW) 
    
    # First we want data set for coarse grid of forces
    vX = np.concatenate((np.arange(-2.,xSt-0.05,0.5),np.arange(xSt,4.,0.5)))
    dDataSet=CreateRNeepDataSet(nTimeStamps,mW,vX)
    with open('RneepDbCoarse.pickle', 'wb') as handle:
        pickle.dump(dDataSet, handle)
    # Second we want data set for fine grid of forces around stalling force
    vX = np.concatenate((np.arange(xSt-0.005,xSt-0.0005,0.001),np.arange(xSt,xSt+0.005,0.001)))
    dDataSet=CreateRNeepDataSet(nTimeStamps,mW,vX)
    with open('RneepDbStalling.pickle', 'wb') as handle:
        pickle.dump(dDataSet, handle)    
    
    ###### Some old datasets that need "dust removal"
    # dTrain,dValid,dTest = CreateBasicDataSet(1e3,1e3,500,100)
    # (lenTraj,nTrain,nValid,nTest)

    # dTrain,dValid,dTest = CreateRnnDataSet(1e5)
    # %% Save data set
    # with open('TrainSet.pickle', 'wb') as handle:
    #     pickle.dump(dTrain, handle)
    # with open('ValidSet.pickle', 'wb') as handle:
    #     pickle.dump(dValid, handle)
    # with open('TestSet.pickle', 'wb') as handle:
    #     pickle.dump(dTest, handle)
        
    # with open('TrainSetRnn.pickle', 'wb') as handle:
    #     pickle.dump(dTrain, handle)
    # with open('ValidSetRnn.pickle', 'wb') as handle:
    #     pickle.dump(dValid, handle)
    # with open('TestSetRnn.pickle', 'wb') as handle:
    #     pickle.dump(dTest, handle)
        
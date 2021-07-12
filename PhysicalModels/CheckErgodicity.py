# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 16:03:38 2020

@title: Monte Carlo to prove Ergodicity in Steady-State

@author: Uri Kapustin

@description: This script checks if created trajectory (not coarse-grained) is eragodic
"""

import numpy as np
import random as rd
from PhysicalModels.UtilityTraj import *
from PhysicalModels.TrajectoryCreation import CreateTrajectory

# %% Define problem
nDim = 4 # dimension of the problem
nTimeStamps = int(1e3) # how much time stamps will be saved
initState = rd.randrange(nDim) # Define initial state in T=0
mW = GenRateMat(nDim) # transition matrix

nTimeStampTest = int(nTimeStamps*0.85) # this time stamp will be sampled for ergodicity proof
nTraj = 500 # number of trajectories for Monte Carlo
nCount0 = 0
nCount1 = 0
nCount2 = 0
nCount3 = 0

# %% Run single trajectory
mTrajectory, mWout = CreateTrajectory(nDim,nTimeStamps,initState,mW) # Run Create Trajectory
## Start of analysing portion of the trajectory
startInd = int(nTimeStamps*0.15)
endInd = nTimeStamps
mPartTraj = mTrajectory[startInd:endInd,:]
## Check if waiting times distributed as 1/lambda for each state
vIndState0 = np.where(mPartTraj[:,0] == 0)
vIndState1 = np.where(mPartTraj[:,0] == 1)
vIndState2 = np.where(mPartTraj[:,0] == 2)
vIndState3 = np.where(mPartTraj[:,0] == 3)
vWaitT0 = mPartTraj[vIndState0,1].T
vWaitT1 = mPartTraj[vIndState1,1].T
vWaitT2 = mPartTraj[vIndState2,1].T
vWaitT3 = mPartTraj[vIndState3,1].T
# Calculte steady-state from simulation by calculating dwell time on each state
vSimSteadyState = np.zeros(nDim) # init steady state
totTime = sum(vWaitT0)+sum(vWaitT1)+sum(vWaitT2)+sum(vWaitT3) # time of the portion of the trajectory
vSingleTrajSS = np.zeros(4)
vSingleTrajSS[0]= sum(vWaitT0)/totTime
vSingleTrajSS[1]= sum(vWaitT1)/totTime
vSingleTrajSS[2]= sum(vWaitT2)/totTime
vSingleTrajSS[3]= sum(vWaitT3)/totTime

for iTtraj in range(nTraj):
    mTrajectoryMC, mWout = CreateTrajectory(nDim,nTimeStamps,initState,mW) # Run Create Trajectory
    vTimeStamps = np.cumsum(mTrajectoryMC[:,1])
    approxTime = np.where(vTimeStamps>=nTimeStampTest)[0][0]
    if mTrajectoryMC[approxTime,0] == 0:
        nCount0 += 1
    elif mTrajectoryMC[approxTime,0] == 1:
        nCount1 += 1
    elif mTrajectoryMC[approxTime,0] == 2:
        nCount2 += 1
    else:
        nCount3 += 1

vMonteCarloSS = np.zeros(4)
vMonteCarloSS[0] = nCount0/(nCount0+nCount1+nCount2+nCount3)
vMonteCarloSS[1] = nCount1/(nCount0+nCount1+nCount2+nCount3)
vMonteCarloSS[2] = nCount2/(nCount0+nCount1+nCount2+nCount3)
vMonteCarloSS[3] = nCount3/(nCount0+nCount1+nCount2+nCount3)
print('Singe Trajectory SS:',vSingleTrajSS)
print('\nMonteCarlo SS:',vMonteCarloSS)
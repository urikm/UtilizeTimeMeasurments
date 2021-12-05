# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 20:36:13 2020

@title:  Utility Trajectory simulation

@author: Uri Kapustin

@description: self explanatory
"""
import numpy as np

# %% Generate Rate matrix 
def GenRateMat(nDim):
    mW = np.random.uniform(size=(nDim,nDim))
    mW = mW/nDim # Normalize
    for k in range(nDim):
        mW[k,k] = mW[k,k] - np.sum(mW[:,k])
    return mW

# Coarse grain rate matrix by combining hidden states to single state.
# Note: there is no importance for dweling rate for the hidden state because its not Poisson distributed
# TODO : write it more general by using the chosen hidden states
def CgRateMatrix(mW,vHidden):
    # Inits
    nDim = np.size(mW,0) # We can take the '1' dimension, doesnt matter for rate matrix
    nDimCg = nDim-(np.size(vHidden)-1)
    mWCg = np.zeros([nDimCg,nDimCg])
    mWCg[0:2,0:2] = mW[0:2,0:2] 
    mWCg[2,0] = mW[2,0] + mW[3,0] 
    mWCg[2,1] = mW[2,1] + mW[3,1]  
    mWCg[0,2] = mW[0,2] + mW[0,3] 
    mWCg[1,2] = mW[1,2] + mW[1,3]      
    # for iState in range(nDim):
    #     if iState in vHidden:
    #     else:
    #         mWCg[:,]
    return mWCg
           
# %% Proceed step of Master equation defined by 'timeJump'
def MasterEqStep(mW,vP,timeJump):
    vPdot = np.dot(mW,vP) # Pdot = W*P
    return vP + vPdot*timeJump


# %% Calculate Entropy Rate from Steady-State probabilities and rate matrix
def EntropyRateCalculation(nDim,mW,vPi):
    entropyRate = 0 
    for iRow in range(nDim-1):
        for iCol in range(iRow+1,nDim):
            Jij,J_j2i,J_i2j = CalcSteadyStateCurrent(mW,vPi,iRow,iCol)
            if np.abs(J_j2i) != 0 and np.abs(J_i2j) != 0:
                entropyRate += Jij*np.log(J_j2i/J_i2j)
    return entropyRate

def CalcSteadyStateCurrent(mW,vPi,iState,jState):
    # iState is inbound state and jState is outbound state
    # vPi - steady state distribution
    J_j2i = mW[iState,jState]*vPi[jState]
    J_i2j = mW[jState,iState]*vPi[iState]    
    Jij = J_j2i-J_i2j
    return Jij,J_j2i,J_i2j

# %% Calculate Kintec bound as mentioned in Rahav's 2021 "Thermodynamic uncertainty relation for first-passage times on Markov chains"
def CalcKineticBoundEntProdRate(mW,vPi):
    nDim = np.size(vPi)
    kineticBound = 0 # Init
    for iState in range(1):#range(nDim): 
        # kineticBound += np.sum(np.dot(np.delete(mW[:,iState],iState),vPi[iState]))
        kineticBound = mW[iState+1,iState]*vPi[iState]+mW[iState,iState+1]*vPi[iState+1]
    return kineticBound

